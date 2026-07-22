"""CuTeDSL kernel v2 for a 2-D transpose: y = x.t().  (optimization playground)

v1 (transpose_cute.py) is frozen at ~66.5% of B200 peak -- it matched Triton's `tl.trans` by using
the exact same mechanism: a 32x32 tile staged through swizzled shared memory. Triton is pinned there
because `tl.trans` always lowers to a small register-level transpose over a fixed tile.

v2's thesis for beating that wall: the DRAM-side ceiling (~65.7% dram throughput in v1) comes from
SHORT write bursts. `y` is (N, M) row-major; a CTA writes a TN x TM tile, so each written row of `y`
is only TM contiguous bf16 = 64 bytes for TM=32. Short, misaligned-ish bursts underutilize the DRAM
row buffer. Going WIDER (bigger TM/TN, multiple 128-bit vectors per thread) makes each store a longer
contiguous run and gives more in-flight memory parallelism per CTA -- something `tl.trans` never does.

Same algorithm as v1 (coalesced vec load -> swizzled smem -> coalesced vec store, transposed column
read), just generalized so TM, TN, THREADS are free parameters and each thread moves several vectors.

Result (16384x16384 bf16, B200): 81.7% of peak (0.164 ms) vs Triton 67.6% and v1 66.5% -- a 1.21x
speedup. ncu confirms the mechanism: DRAM throughput 65.7% (v1) -> 80.7% (v2). Shared-memory load
conflicts actually go UP (0.5M -> 9.2M) but are fully hidden behind DRAM latency at 92% warp
occupancy, so the kernel is now genuinely DRAM-bound rather than smem/write-burst bound.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

# --- tile / launch config (swept) ---
# The win over v1/Triton: a big 128x128 tile makes each written row of y a 256-byte contiguous run
# (vs 64 bytes for a 32-wide tile), so DRAM write bursts are full-length. 512 threads (4 vectors each
# per phase) keep enough memory ops in flight; the wide smem row needs a LARGER swizzle shift (S=8) to
# scatter its column-read banks -- with S too small (v1's S=5) the wide tile collapses to ~25%.
_TM = 128
_TN = 128
_VEC = 8                          # 8 bf16 = 128 bits
_THREADS = 512
_LD_BITS = _VEC * 16              # 128-bit copy

# derived vector bookkeeping
_LOAD_NCOLS = _TN // _VEC         # 128-bit vectors per smem/global row
_LOAD_VECS = _TM * _LOAD_NCOLS    # total load vectors per tile
_LOAD_VPT = _LOAD_VECS // _THREADS
_STORE_MCOLS = _TM // _VEC        # vectors along M in a y-row
_STORE_VECS = _TN * _STORE_MCOLS
_STORE_VPT = _STORE_VECS // _THREADS

# Swizzle(BBits, MBase, SShift): MBase=3 protects the low 3 offset bits (the 8-bf16 = 128-bit vector)
# so phase-1 stores stay vectorized; the XOR scatters the phase-2 column-read banks. Swept per tile.
_SW_B = 4
_SW_M = 3
_SW_S = 8

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

    # ---- phase 1: coalesced 128-bit global loads -> vectorized 128-bit smem stores (swizzled) ----
    gX = cute.local_tile(mX, (_TM, _TN), (bx, by))         # (TM, TN), stride (N, 1)
    gXv = cute.zipped_divide(gX, (1, _VEC))                # ((1,VEC), (TM, TN//VEC))
    sTv = cute.zipped_divide(sT, (1, _VEC))
    for i in cutlass.range_constexpr(_LOAD_VPT):
        v = tid + i * _THREADS
        row = v // _LOAD_NCOLS
        vc = v % _LOAD_NCOLS
        cute.copy(atom, gXv[(0, None), (row, vc)], sTv[(0, None), (row, vc)])
    cute.arch.sync_threads()

    # ---- phase 2: swizzled smem column reads -> registers -> coalesced 128-bit global stores ----
    gY = cute.local_tile(mY, (_TN, _TM), (by, bx))         # (TN, TM), stride (M, 1)
    gYv = cute.zipped_divide(gY, (1, _VEC))                # ((1,VEC), (TN, TM//VEC))
    for i in cutlass.range_constexpr(_STORE_VPT):
        v = tid + i * _THREADS
        orow = v // _STORE_MCOLS                           # 0..TN-1  (= n)
        mvc = v % _STORE_MCOLS                             # 0..TM//VEC-1 (adjacent tid -> adjacent m)
        # y[orow, mvc*VEC + j] = x[mvc*VEC + j, orow] = sT[mvc*VEC + j, orow] -- strided column gather.
        frg = cute.make_rmem_tensor(cute.make_layout(_VEC), cutlass.BFloat16)
        for j in cutlass.range_constexpr(_VEC):
            frg[j] = sT[mvc * _VEC + j, orow]
        cute.copy(atom, frg, gYv[(0, None), (orow, mvc)])  # STG.128


@cute.jit
def _transpose_vec_jit(mX: cute.Tensor, mY: cute.Tensor, M: cutlass.Int32, N: cutlass.Int32):
    sw = cute.make_swizzle(_SW_B, _SW_M, _SW_S)
    smem_layout = cute.make_composed_layout(sw, 0, cute.make_layout((_TM, _TN), stride=(_TN, 1)))
    grid_m = M // _TM
    grid_n = N // _TN
    _transpose_vec_kernel(mX, mY, smem_layout).launch(
        grid=(grid_m, grid_n, 1), block=(_THREADS, 1, 1))


# --- scalar fallback for ragged shapes (identical to v1) ---
@cute.struct
class _ScalarSmem:
    tile: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, 32 * 32], 1024]


@cute.kernel
def _transpose_scalar_kernel(gX: cute.Tensor, gY: cute.Tensor, M: cutlass.Int32, N: cutlass.Int32,
                             smem_layout: cute.Layout):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    tile_m = bx * 32
    tile_n = by * 32

    smem = utils.SmemAllocator()
    st = smem.allocate(_ScalarSmem)
    sT = st.tile.get_tensor(smem_layout)

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
    smem_layout = cute.make_layout((32, 32), stride=(32, 1))
    grid_m = (M + 31) // 32
    grid_n = (N + 31) // 32
    _transpose_scalar_kernel(mX, mY, M, N, smem_layout).launch(
        grid=(grid_m, grid_n, 1), block=(32, 32, 1))


def transpose_cute_v2(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    M, N = x.shape
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)
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
kernel_fn = transpose_cute_v2


def reference_fn(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous()


def get_inputs():
    return [torch.randn(16384, 16384, dtype=torch.bfloat16, device="cuda")]
