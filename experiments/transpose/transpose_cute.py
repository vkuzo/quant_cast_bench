"""CuTeDSL kernel for a 2-D transpose: y = x.t().

Mirrors the Triton kernel at a high level: a 2-D grid of TM x TN tiles, each staged through shared
memory (coalesced global load -> smem -> sync -> coalesced global store, reading smem transposed).

Two code paths, same algorithm:
  * `_transpose_vec_kernel` (fast, requires TM|M, TN|N): 128 threads per 32x32 tile (vs 1024), each
    thread moves a 128-bit VECTORIZED run of 8 bf16 on BOTH the coalesced global load and the
    coalesced global store. The smem tile uses a SWIZZLED layout (make_swizzle) so that the phase-1
    store stays vectorized *and* the transposed phase-2 column read is (near) conflict-free -- the
    same mechanism Triton's `tl.trans` uses. This lands within ~1 pt of the Triton kernel.
  * `_transpose_scalar_kernel` (general): one bf16 element per thread; handles ragged shapes with
    predication. Used as a fallback when the tile doesn't divide the shape.

bf16-specific for now.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

_TM = 32
_TN = 32
_VEC = 8                       # 8 bf16 = 128 bits
_THREADS = (_TM * _TN) // _VEC  # 128 threads/tile, 1 vector per thread per phase
_LD_BITS = _VEC * 16           # 128-bit copy
# Swizzle(BBits, MBase, SShift): MBase=3 leaves the low 3 bits (the 8-bf16 = 128-bit vector) untouched
# so the phase-1 store stays vectorized; the XOR scatters the column-read banks in phase 2. Swept.
_SW_B = 2
_SW_M = 3
_SW_S = 5

# cute.compile is slow; cache the compiled callable per (shape, path) and reuse it.
_COMPILE_CACHE: dict = {}


@cute.struct
class _TransposeSmem:
    tile: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _TM * _TN], 1024]


@cute.kernel
def _transpose_vec_kernel(mX: cute.Tensor, mY: cute.Tensor, smem_layout: cute.ComposedLayout):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()  # bx = m-tile, by = n-tile

    smem = utils.SmemAllocator()
    st = smem.allocate(_TransposeSmem)
    sT = st.tile.get_tensor(smem_layout)  # (TM, TN) row-major, swizzled

    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                               num_bits_per_copy=_LD_BITS)

    # ---- phase 1: coalesced 128-bit global load -> vectorized 128-bit smem store (swizzled) ----
    gX = cute.local_tile(mX, (_TM, _TN), (bx, by))         # (TM, TN), stride (N, 1)
    gXv = cute.zipped_divide(gX, (1, _VEC))                # ((1,VEC), (TM, TN//VEC))
    sTv = cute.zipped_divide(sT, (1, _VEC))                # ((1,VEC), (TM, TN//VEC))
    ncols = _TN // _VEC
    row = tid // ncols                                     # 0..TM-1
    vc = tid % ncols                                       # 0..TN//VEC-1  (adjacent tid -> adjacent n)
    cute.copy(atom, gXv[(0, None), (row, vc)], sTv[(0, None), (row, vc)])  # LDG.128 -> STS.128
    cute.arch.sync_threads()

    # ---- phase 2: swizzled smem column read -> registers -> coalesced 128-bit global store ----
    gY = cute.local_tile(mY, (_TN, _TM), (by, bx))         # (TN, TM), stride (M, 1)
    gYv = cute.zipped_divide(gY, (1, _VEC))                # ((1,VEC), (TN, TM//VEC))
    mcols = _TM // _VEC
    orow = tid // mcols                                    # 0..TN-1  (= n)
    mvc = tid % mcols                                      # 0..TM//VEC-1 (adjacent tid -> adjacent m)
    # y[orow, mvc*VEC + j] = x[mvc*VEC + j, orow] = sT[mvc*VEC + j, orow] -- strided column gather;
    # the swizzle makes these strided reads (near) conflict-free.
    frg2 = cute.make_rmem_tensor(cute.make_layout(_VEC), cutlass.BFloat16)
    for j in cutlass.range_constexpr(_VEC):
        frg2[j] = sT[mvc * _VEC + j, orow]
    cute.copy(atom, frg2, gYv[(0, None), (orow, mvc)])     # STG.128


@cute.jit
def _transpose_vec_jit(mX: cute.Tensor, mY: cute.Tensor, M: cutlass.Int32, N: cutlass.Int32):
    sw = cute.make_swizzle(_SW_B, _SW_M, _SW_S)
    smem_layout = cute.make_composed_layout(sw, 0, cute.make_layout((_TM, _TN), stride=(_TN, 1)))
    grid_m = M // _TM
    grid_n = N // _TN
    _transpose_vec_kernel(mX, mY, smem_layout).launch(
        grid=(grid_m, grid_n, 1), block=(_THREADS, 1, 1))


@cute.kernel
def _transpose_scalar_kernel(gX: cute.Tensor, gY: cute.Tensor, M: cutlass.Int32, N: cutlass.Int32,
                             smem_layout: cute.Layout):
    tx, ty, _ = cute.arch.thread_idx()   # tx in [0, TN) -> col of x; ty in [0, TM) -> row of x
    bx, by, _ = cute.arch.block_idx()
    tile_m = bx * _TM
    tile_n = by * _TN

    smem = utils.SmemAllocator()
    st = smem.allocate(_TransposeSmem)
    sT = st.tile.get_tensor(smem_layout)  # (TM, TN)

    m = tile_m + ty
    n = tile_n + tx
    if cutlass.dynamic_expr((m < M) & (n < N)):
        sT[ty, tx] = gX[m, n]
    cute.arch.sync_threads()

    m_out = tile_m + tx
    n_out = tile_n + ty
    if cutlass.dynamic_expr((m_out < M) & (n_out < N)):
        gY[n_out, m_out] = sT[tx, ty]


@cute.jit
def _transpose_scalar_jit(mX: cute.Tensor, mY: cute.Tensor, M: cutlass.Int32, N: cutlass.Int32):
    smem_layout = cute.make_layout((_TM, _TN), stride=(_TN, 1))
    grid_m = (M + _TM - 1) // _TM
    grid_n = (N + _TN - 1) // _TN
    _transpose_scalar_kernel(mX, mY, M, N, smem_layout).launch(
        grid=(grid_m, grid_n, 1), block=(_TN, _TM, 1))


def transpose_cute(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    M, N = x.shape
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)
    # Static layout (no mark_layout_dynamic): lets CuTe prove 128-bit alignment for the vectorized
    # copies. Safe because we cache/compile per shape anyway.
    mX = from_dlpack(x, assumed_align=16)
    mY = from_dlpack(y, assumed_align=16)
    vectorized = (M % _TM == 0) and (N % _TN == 0)
    jit_fn = _transpose_vec_jit if vectorized else _transpose_scalar_jit
    key = (M, N, vectorized)
    fn = _COMPILE_CACHE.get(key)
    if fn is None:
        fn = cute.compile(jit_fn, mX, mY, M, N)
        _COMPILE_CACHE[key] = fn
    fn(mX, mY, M, N)
    return y


# --- shared contract (test.py / benchmark.py rely on these names) ---
kernel_fn = transpose_cute


def reference_fn(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous()


def get_inputs():
    return [torch.randn(16384, 16384, dtype=torch.bfloat16, device="cuda")]
