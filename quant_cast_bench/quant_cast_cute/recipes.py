"""CuTeDSL (cutlass.cute) implementations of the quant_cast_gold recipes.

Mirrors quant_cast_triton: each recipe is a `QuantCastCuteRecipe` inheriting the gold reference
(`pt_ref_fn`/`correctness_fn`/`example_input_fn`/`perf_description`) from a
`QuantCastSingleKernelGold` and adding `cute_fn`, a CuTeDSL implementation of the same cast.
test.py grades each `cute_fn` against its gold `pt_ref_fn`.

Correctness-first: kernels are naive (tiling not tuned). `cute.compile` is cached per (recipe,
shape) since it is slow and the benchmark calls each fn many times.
"""

from dataclasses import dataclass
from typing import Callable

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm  # inline PTX for the hardware fp4/e4m3 cvts
from cutlass.cute.nvgpu import cpasync  # TMA (bulk-tensor) copy ops + tma_partition
from cutlass.cute.nvgpu import warp as warp_mma  # SM80 warp-level mma.sync atoms (m16n8k16)
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import _maybe_recast_from_f4  # packs an fp4 register vector to bytes
from cutlass.cutlass_dsl import T, dsl_user_op

from quant_cast_bench.quant_cast_gold.recipes import (
    ColwiseFp8Gold,
    ColwisePrecalcGold,
    Deepseek1x128DimKmGold,
    Deepseek1x128DimMGold,
    Deepseek1x128Gold,
    Deepseek128x128Gold,
    Float8TensorwiseGold,
    HadamardRht,
    Mxfp832x32FloorGold,
    Mxfp8FloorDimKmGold,
    Mxfp8FloorDimMGold,
    Mxfp8FloorGold,
    Mxfp8FloorSwizzleGold,
    Nvfp4BlockedOuterGold,
    Nvfp4GsSwizzleGold,
    QuantCastSingleKernelGold,
    RowwiseFp8Gold,
    RowwisePrecalcGold,
    SrF32ToBf16,
)


@dataclass(frozen=True)
class QuantCastCuteRecipe(QuantCastSingleKernelGold):
    """A gold recipe plus a CuTeDSL implementation of its `pt_ref_fn`. Mirrors flexquant_v3's
    RecipeV2 / quant_cast_triton's QuantCastTritonRecipe: inherits from the gold, adds `cute_fn`."""

    cute_fn: Callable | None = None

    @classmethod
    def from_gold(cls, gold: QuantCastSingleKernelGold, cute_fn: Callable) -> "QuantCastCuteRecipe":
        return cls(
            pt_ref_fn=gold.pt_ref_fn,
            correctness_fn=gold.correctness_fn,
            example_input_fn=gold.example_input_fn,
            perf_description=gold.perf_description,
            cute_fn=cute_fn,
        )


# cute.compile is slow; cache the compiled callable per (recipe, shape) and reuse it (called with
# fresh cute tensors of the same layout each invocation, like the cutlass benchmark examples).
_COMPILE_CACHE: dict = {}


def _compiled(key, jit_fn, *cute_args):
    fn = _COMPILE_CACHE.get(key)
    if fn is None:
        fn = cute.compile(jit_fn, *cute_args)
        _COMPILE_CACHE[key] = fn
    return fn


# ---------------------------------------------------------------------------
# fp8 tensorwise with a precomputed per-tensor scale (scalar). Elementwise: qdata = (x/scale)->e4m3.
# The cast is a pure elementwise memory-bound op (no reuse/reduction), so it just needs to move
# bytes at DRAM speed-of-light. We flatten to 1-D and tile it 128 threads x 16 elems, with each
# thread owning a contiguous 16-element run (tv stride (V,1)) so a warp reads 32 contiguous runs
# (coalesced) and each run is issued as a single 128-bit VECTORIZED transaction via
# `num_bits_per_copy` copy atoms (bf16 load 128b, fp8 store 128b). This is the crux: without the
# vectorized atoms CopyUniversalOp emits per-element strided loads (~4/32 bytes per sector used,
# L1TEX-bound ~48% peak); with them, ~90% DRAM throughput (ncu) = ~84% of B200 peak at 16384.
# `assumed_align=16` lets the 128-bit copies assume 16B alignment (torch allocations satisfy it).
# Because numel % VPT == 0, a thread's whole 16-vector is in-range iff its first element is, so the
# single-coordinate predicate correctly guards a partial final tile.
# ---------------------------------------------------------------------------
_TENSORWISE_THREADS = 128
_TENSORWISE_VPT = 16
_TENSORWISE_LD_BITS = min(128, _TENSORWISE_VPT * 16)  # bf16 input
_TENSORWISE_ST_BITS = min(128, _TENSORWISE_VPT * 8)   # fp8 output


@cute.kernel
def _tensorwise_kernel(gX: cute.Tensor, gY: cute.Tensor, gS: cute.Tensor, cX: cute.Tensor,
                       shape: cute.Shape, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    coord = (None, bidx)
    thrX = cute.composition(gX[coord], tv_layout)[(tidx, None)]
    thrY = cute.composition(gY[coord], tv_layout)[(tidx, None)]
    thrC = cute.composition(cX[coord], tv_layout)[(tidx, None)]

    if cute.elem_less(thrC[0], shape):
        ld_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type,
                                      num_bits_per_copy=_TENSORWISE_LD_BITS)
        st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type,
                                      num_bits_per_copy=_TENSORWISE_ST_BITS)
        frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
        cute.copy(ld_atom, thrX, frgX)
        frgS = cute.make_rmem_tensor(cute.make_layout(1), gS.element_type)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gS.element_type), gS, frgS)
        recip = 1.0 / frgS[0]
        frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
        frgY.store((frgX.load().to(cutlass.Float32) * recip).to(gY.element_type))
        cute.copy(st_atom, frgY, thrY)


@cute.jit
def _tensorwise_jit(mX, mY, mS):
    # mX/mY are 1-D (flattened); tile the single axis by 128*VPT.
    tv_layout = cute.make_layout((_TENSORWISE_THREADS, _TENSORWISE_VPT), stride=(_TENSORWISE_VPT, 1))
    tiler = (cute.size(tv_layout),)
    gX = cute.zipped_divide(mX, tiler)
    gY = cute.zipped_divide(mY, tiler)
    cX = cute.zipped_divide(cute.make_identity_tensor(mX.shape), tiler)
    _tensorwise_kernel(gX, gY, mS, cX, mX.shape, tv_layout).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def fp8_tensorwise_cute(x, scale, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert (M * N) % _TENSORWISE_VPT == 0, "tensorwise cute kernel needs numel % 16 == 0"
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    # assumed_align=16 enables the 128-bit vectorized copies (torch allocations are >=256B aligned).
    mX = from_dlpack(x.reshape(-1), assumed_align=16).mark_layout_dynamic()  # flatten (elementwise)
    mY = from_dlpack(y.reshape(-1), assumed_align=16).mark_layout_dynamic()
    mS = from_dlpack(scale.reshape(1))  # static (1,) for the in-kernel scalar copy
    fn = _compiled(("tensorwise", M, N), _tensorwise_jit, mX, mY, mS)
    fn(mX, mY, mS)
    return (y,)


FP8_TENSORWISE_PRECALC_SCALE = QuantCastCuteRecipe.from_gold(
    Float8TensorwiseGold, cute_fn=fp8_tensorwise_cute
)


# ---------------------------------------------------------------------------
# fp8 rowwise with a precomputed (M, 1) per-row scale. Flat (1, TileV) tiles lie within one row,
# so all of a thread's values share one scale = scale[row]; row is read from the coordinate tensor.
# Mirrors rowwise_precalc_f: qdata = (x / scale[row]) -> e4m3.
# ---------------------------------------------------------------------------
@cute.kernel
def _rowwise_precalc_kernel(gX: cute.Tensor, gY: cute.Tensor, gS: cute.Tensor, cX: cute.Tensor,
                            shape: cute.Shape, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    coord = (None, bidx)
    thrX = cute.composition(gX[coord], tv_layout)[(tidx, None)]
    thrY = cute.composition(gY[coord], tv_layout)[(tidx, None)]
    thrC = cute.composition(cX[coord], tv_layout)[(tidx, None)]

    if cute.elem_less(thrC[0], shape):
        row = thrC[0][0]
        frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type), thrX, frgX)
        frgS = cute.make_rmem_tensor(cute.make_layout(1), gS.element_type)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gS.element_type),
                  gS[(row, None)], frgS)
        sval = frgS[0]
        frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
        frgY.store((frgX.load().to(cutlass.Float32) / sval).to(gY.element_type))
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type), frgY, thrY)


@cute.jit
def _rowwise_precalc_jit(mX, mY, mS):
    tv_layout = cute.make_layout((128, 4), stride=(4, 1))
    tiler = [1] * cute.rank(mX.layout)
    tiler[1] = cute.size(tv_layout)
    gX = cute.zipped_divide(mX, tuple(tiler))
    gY = cute.zipped_divide(mY, tuple(tiler))
    cX = cute.zipped_divide(cute.make_identity_tensor(mX.shape), tuple(tiler))
    _rowwise_precalc_kernel(gX, gY, mS, cX, mX.shape, tv_layout).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def fp8_rowwise_precalc_cute(x, scale, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    mX = from_dlpack(x).mark_layout_dynamic()
    mY = from_dlpack(y).mark_layout_dynamic()
    mS = from_dlpack(scale)  # (M, 1) static so gS[(row, None)] -> static (1,)
    fn = _compiled(("rowwise_precalc", M, N), _rowwise_precalc_jit, mX, mY, mS)
    fn(mX, mY, mS)
    return (y,)


FP8_ROWWISE_PRECALC_SCALE = QuantCastCuteRecipe.from_gold(
    RowwisePrecalcGold, cute_fn=fp8_rowwise_precalc_cute
)


# ---------------------------------------------------------------------------
# fp8 colwise with a precomputed (1, N) per-column scale; transposed-contiguous output (N, M).
# The per-column scale is broadcast to (M, N) via a stride-0 layout and tiled like x, so each
# thread loads its own column-scales as a fragment; the output is a transposed (M, N)-strided view
# of the (N, M) buffer, so a normal store lands transposed. Mirrors colwise_precalc_f.
# ---------------------------------------------------------------------------
@cute.kernel
def _colwise_precalc_kernel(gX: cute.Tensor, gY: cute.Tensor, gS: cute.Tensor, cX: cute.Tensor,
                            shape: cute.Shape, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    coord = (None, bidx)
    thrX = cute.composition(gX[coord], tv_layout)[(tidx, None)]
    thrY = cute.composition(gY[coord], tv_layout)[(tidx, None)]
    thrS = cute.composition(gS[coord], tv_layout)[(tidx, None)]
    thrC = cute.composition(cX[coord], tv_layout)[(tidx, None)]

    if cute.elem_less(thrC[0], shape):
        frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type), thrX, frgX)
        frgS = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gS.element_type)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gS.element_type), thrS, frgS)
        frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
        frgY.store((frgX.load().to(cutlass.Float32) / frgS.load()).to(gY.element_type))
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type), frgY, thrY)


@cute.jit
def _colwise_precalc_jit(mX, mY, mS):
    tv_layout = cute.make_layout((128, 4), stride=(4, 1))
    tiler = [1] * cute.rank(mX.layout)
    tiler[1] = cute.size(tv_layout)
    # broadcast the (1, N) scale to (M, N) with a stride-0 row so it tiles exactly like x
    mS_bcast = cute.make_tensor(mS.iterator, cute.make_layout(mX.shape, stride=(0, 1)))
    gX = cute.zipped_divide(mX, tuple(tiler))
    gY = cute.zipped_divide(mY, tuple(tiler))
    gS = cute.zipped_divide(mS_bcast, tuple(tiler))
    cX = cute.zipped_divide(cute.make_identity_tensor(mX.shape), tuple(tiler))
    _colwise_precalc_kernel(gX, gY, gS, cX, mX.shape, tv_layout).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def fp8_colwise_precalc_cute(x, scale, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)  # transposed output
    mX = from_dlpack(x).mark_layout_dynamic()
    mY = from_dlpack(y.t()).mark_layout_dynamic()  # (M, N) strided view -> store lands transposed
    mS = from_dlpack(scale).mark_layout_dynamic()  # (1, N)
    fn = _compiled(("colwise_precalc", M, N), _colwise_precalc_jit, mX, mY, mS)
    fn(mX, mY, mS)
    return (y,)


FP8_COLWISE_PRECALC_SCALE = QuantCastCuteRecipe.from_gold(
    ColwisePrecalcGold, cute_fn=fp8_colwise_precalc_cute
)


# ---------------------------------------------------------------------------
# deepseek fp8 1x128 (vectorized dim-K path): the 85.9%-tensorwise / 67.6%-mxfp8_swizzle recipe
# applied to 1x128 blocks. 1-D flatten, 128 threads/CTA, VPT=32 (128-bit vectorized ld/st via
# assumed_align=16). A 1x128 block is 4 contiguous threads (4 x 32 = 128), so the per-thread abs-max
# is combined across the group with warp_reduction_max(threads_in_group=4). Because N % 128 == 0,
# every 128-aligned run stays within one row, so a 4-thread group IS exactly one 1x128 block; the
# group leader (tidx%4==0) scatters the fp32 scale to its (row, col-block) slot. Replaces the scalar
# one-warp-per-block kernel (~9% peak).
# ---------------------------------------------------------------------------
_DS_THREADS = 128
_DS_VPT = 32                              # 32 vals/thread; 4 threads (128 vals) = one 1x128 block
_DS_GROUP = 128 // _DS_VPT                # 4-thread group reduce width
_DS_CHUNK = _DS_THREADS * _DS_VPT         # 4096 elements per CTA (1-D tile) = 32 blocks
_DS_LD_BITS = min(128, _DS_VPT * 16)      # bf16 in  -> 128-bit vectorized load
_DS_ST_BITS = min(128, _DS_VPT * 8)       # fp8 out  -> 128-bit vectorized store


@cute.kernel
def _deepseek_1x128_opt_kernel(gX: cute.Tensor, gY: cute.Tensor, sflat: cute.Tensor,
                               cX: cute.Tensor, tv_layout: cute.Layout,
                               N: cutlass.Constexpr, ncb: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    thrX = cute.composition(gX[(None, bidx)], tv_layout)[(tidx, None)]
    thrY = cute.composition(gY[(None, bidx)], tv_layout)[(tidx, None)]
    thrC = cute.composition(cX[(None, bidx)], tv_layout)[(tidx, None)]
    ld_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type,
                                  num_bits_per_copy=_DS_LD_BITS)
    st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type,
                                  num_bits_per_copy=_DS_ST_BITS)
    frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
    cute.copy(ld_atom, thrX, frgX)
    v = frgX.load().to(cutlass.Float32)
    local = cutlass.max(v, -v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    amax = cutlass.max(cute.arch.warp_reduction_max(local, threads_in_group=_DS_GROUP),
                       cutlass.Float32(1e-12))
    scale = amax / 448.0
    frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
    frgY.store((v * (1.0 / scale)).to(gY.element_type))
    cute.copy(st_atom, frgY, thrY)
    # group leader writes the fp32 scale for its 1x128 block at (row, col-block)
    if tidx % _DS_GROUP == 0:
        p = thrC[0][0]                                   # flat index of the block's first element
        sflat[(p // N) * ncb + (p % N) // 128] = scale.to(sflat.element_type)


@cute.jit
def _deepseek_1x128_opt_jit(mX, mY, sflat, N: cutlass.Constexpr, ncb: cutlass.Constexpr):
    tv_layout = cute.make_layout((_DS_THREADS, _DS_VPT), stride=(_DS_VPT, 1))
    tiler = (_DS_CHUNK,)
    gX = cute.zipped_divide(mX, tiler)
    gY = cute.zipped_divide(mY, tiler)
    cX = cute.zipped_divide(cute.make_identity_tensor(mX.shape), tiler)
    _deepseek_1x128_opt_kernel(gX, gY, sflat, cX, tv_layout, N, ncb).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[_DS_THREADS, 1, 1])


def fp8_deepseek_1x128_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert N % 128 == 0 and (M * N) % _DS_CHUNK == 0, \
        f"deepseek_1x128 cute kernel needs N%128==0 and numel%{_DS_CHUNK}==0"
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    ncb = N // 128
    s = torch.empty(M, ncb, dtype=torch.float32, device=x.device)
    # flatten (elementwise-style); assumed_align=16 enables the 128-bit vectorized copies.
    mX = from_dlpack(x.reshape(-1), assumed_align=16).mark_layout_dynamic()
    mY = from_dlpack(y.reshape(-1), assumed_align=16).mark_layout_dynamic()
    sflat = from_dlpack(s.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("deepseek_1x128", M, N), _deepseek_1x128_opt_jit, mX, mY, sflat, N, ncb)
    fn(mX, mY, sflat)
    return y, s


FP8_DEEPSEEK_1X128 = QuantCastCuteRecipe.from_gold(
    Deepseek1x128Gold, cute_fn=fp8_deepseek_1x128_cute
)


# ---------------------------------------------------------------------------
# deepseek fp8 1x128 dim-M: reduce 128-row blocks down M, transposed outputs (N, M) / (N, M//128).
# The direct analog of `mxfp8_floor_dim_m` (warp-specialized TMA), with a 128-row block (not 32) and
# an fp32 amax/448 scale (not an e8m0 byte). Feeding a transposed x.t() to the scalar kernel makes
# every load uncoalesced (~7% peak); instead:
#   - TMA G2S loads a (TM, TN) row-major tile of x into smem (sIN), TM a multiple of 128;
#   - the (TM/128 x TN) scale-groups are split across threads; each owns one (128-row block, col),
#     scans its 128 rows down a column of sIN for the amax (scalar accumulate -> low registers, keeps
#     occupancy high), computes scale = max(amax,1e-12)/448, then re-reads the column to quantize and
#     write the 128 fp8 values as a CONTIGUOUS run into sOUT laid out (TN, TM) -- the transpose
#     happens in the register->smem write (no col-major TMA, which the DSL can't drive);
#   - TMA S2G stores sOUT into the (TN, TM) tile of the row-major (N, M) output at (n_tile, m_tile).
# The fp32 scale is scattered straight to gmem scales (N, M//128). Barrier follows the same
# arrive-and-expect-tx pattern (single arrival, warp-0 gated) required for a multi-warp block.
# ---------------------------------------------------------------------------
_DSM_TM, _DSM_TN, _DSM_WARPS = 128, 128, 4         # tuned on B200 @ 16384 (needs M%TM==0, N%TN==0)
_DSM_THREADS = _DSM_WARPS * 32
_DSM_RB = _DSM_TM // 128                            # 128-row blocks per tile
_DSM_CHUNKS = 128 // 32                              # 32-wide chunks per 128-row block (vectorize)
_DSM_GROUPS = _DSM_TN * _DSM_RB                     # (col, row-block) scale groups
_DSM_ITERS = (_DSM_GROUPS + _DSM_THREADS - 1) // _DSM_THREADS
_DSM_IN_BYTES = _DSM_TM * _DSM_TN * 2               # bf16 tile bytes for the TMA expect-tx


@cute.struct
class _DeepseekDimMSmem:
    bar: cute.struct.MemRange[cutlass.Int64, 1]
    sin: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _DSM_TM * _DSM_TN], 1024]
    sout: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, _DSM_TM * _DSM_TN], 1024]


@cute.kernel
def _deepseek_dim_m_kernel(atom_in: cute.CopyAtom, ten_in: cute.Tensor, atom_out: cute.CopyAtom,
                           ten_out: cute.Tensor, scales: cute.Tensor, sil: cute.Layout,
                           sol: cute.Layout, M: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()   # bidx = m_tile, bidy = n_tile
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    st = smem.allocate(_DeepseekDimMSmem)
    bar = st.bar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    sIN = st.sin.get_tensor(sil)                             # (TM, TN) row-major
    sOUT = st.sout.get_tensor(sol)                           # (TN, TM) row-major (transposed)
    gIN = cute.local_tile(ten_in, (_DSM_TM, _DSM_TN), (None, None))
    gOUT = cute.local_tile(ten_out, (_DSM_TN, _DSM_TM), (None, None))
    tAsA, tAgA = cpasync.tma_partition(atom_in, 0, cute.make_layout(1),
                                       cute.group_modes(sIN, 0, 2), cute.group_modes(gIN, 0, 2))
    tOsO, tBgB = cpasync.tma_partition(atom_out, 0, cute.make_layout(1),
                                       cute.group_modes(sOUT, 0, 2), cute.group_modes(gOUT, 0, 2))
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, _DSM_IN_BYTES)
        cute.copy(atom_in, tAgA[(None, bidx, bidy)], tAsA, tma_bar_ptr=bar)
    cute.arch.mbarrier_wait(bar, 0)

    m0 = bidx * _DSM_TM
    n0 = bidy * _DSM_TN
    mblk = M // 128
    for it in cutlass.range_constexpr(_DSM_ITERS):
        g = tidx + it * _DSM_THREADS
        if g < _DSM_GROUPS:
            col = g % _DSM_TN
            rb = g // _DSM_TN
            r0 = rb * 128
            # pass 1: amax over the 128 rows down this column, in 4 chunks of 32 (vector reduce,
            # only 32 f32 live at a time -> low registers, high occupancy).
            amax = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(_DSM_CHUNKS):
                frgIn = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                for r in cutlass.range_constexpr(32):
                    frgIn[r] = sIN[r0 + c * 32 + r, col].to(cutlass.Float32)
                v = frgIn.load()
                amax = cutlass.max(amax, cute.where(v < 0, -v, v).reduce(
                    cute.ReductionOp.MAX, cutlass.Float32(0.0), 0))
            scale = cutlass.max(amax, cutlass.Float32(1e-12)) / 448.0
            inv = 1.0 / scale                                 # reciprocal-mul, not per-element divs
            # pass 2: re-read (cheap smem), quantize each chunk (vectorized f32->fp8), transpose in
            # the register->smem write (contiguous run into sOUT).
            for c in cutlass.range_constexpr(_DSM_CHUNKS):
                frgIn = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                for r in cutlass.range_constexpr(32):
                    frgIn[r] = sIN[r0 + c * 32 + r, col].to(cutlass.Float32)
                frgOut = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
                frgOut.store((frgIn.load() * inv).to(cutlass.Float8E4M3FN))
                for r in cutlass.range_constexpr(32):
                    sOUT[col, r0 + c * 32 + r] = frgOut[r]
            scales[(n0 + col) * mblk + (m0 // 128 + rb)] = scale.to(scales.element_type)

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
    if warp == 0:
        cute.copy(atom_out, tOsO, tBgB[(None, bidy, bidx)])       # y tile (n_tile, m_tile)


@cute.jit
def _deepseek_dim_m_jit(mIN, mOUT, scales, M: cutlass.Constexpr):
    sil = cute.make_layout((_DSM_TM, _DSM_TN), stride=(_DSM_TN, 1))
    sol = cute.make_layout((_DSM_TN, _DSM_TM), stride=(_DSM_TM, 1))
    atom_in, ten_in = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mIN, sil, (_DSM_TM, _DSM_TN))
    atom_out, ten_out = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mOUT, sol, (_DSM_TN, _DSM_TM))
    M2, N2 = mIN.shape
    grid = ((M2 + _DSM_TM - 1) // _DSM_TM, (N2 + _DSM_TN - 1) // _DSM_TN, 1)
    _deepseek_dim_m_kernel(atom_in, ten_in, atom_out, ten_out, scales, sil, sol,
                           cutlass.Int64(M)).launch(grid=grid, block=(_DSM_THREADS, 1, 1),
                                                     cluster=(1, 1, 1))


def fp8_deepseek_1x128_dim_m_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % _DSM_TM == 0 and N % _DSM_TN == 0, \
        f"deepseek_1x128_dim_m cute kernel needs M%{_DSM_TM}==0 and N%{_DSM_TN}==0"
    y = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)  # transposed row-major output
    s = torch.empty(N, M // 128, dtype=torch.float32, device=x.device)
    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-elem aligned).
    mIN = (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
           .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mOUT = (from_dlpack(y, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=16))
    scales = from_dlpack(s.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("deepseek_1x128_dim_m", M, N), _deepseek_dim_m_jit, mIN, mOUT, scales, M)
    fn(mIN, mOUT, scales)
    return y, s


FP8_DEEPSEEK_1X128_DIM_M = QuantCastCuteRecipe.from_gold(
    Deepseek1x128DimMGold, cute_fn=fp8_deepseek_1x128_dim_m_cute
)


# ---------------------------------------------------------------------------
# deepseek fp8 1x128 in BOTH directions, one pass over x (mirrors deepseek_1x128_dim_km_f). The
# deepseek analog of mxfp8_floor_dim_km: same fused TMA BMxBN template, but a 128-block and an fp32
# max(amax,1e-12)/448 scale (not a 1x32 e8m0 byte). Four outputs: dim-K (qk (M,N), sk (M,N//128) --
# 1x128 blocks along columns, like deepseek_1x128) and dim-M (qm (N,M), sm (N,M//128) -- 128x1 blocks
# down rows, transposed, like deepseek_1x128_dim_m).
#
# TMA G2S loads one (TM, TN) row-major tile of x into smem (sIN), read once and reduced BOTH ways:
#   - dim-M (binding): each (col, 128-row-block) group scans its 128 rows DOWN a column of sIN for
#     the amax (4 chunks of 32 -> only 32 f32 live, low registers), scales, re-reads to quantize, and
#     writes the run into sOUT laid out (TN, TM) -- the transpose is the register->smem write -- for a
#     TMA store into the (N, M) output.
#   - dim-K (rides the loaded tile): each (row, 128-col-block) group scans its 128 cols ALONG a row,
#     and since qk keeps x's (M,N) layout it quantizes in registers and stores each 32-chunk DIRECTLY
#     to gmem with a 128-bit vectorized copy (a thread owns a whole 128-col block = one 128 B sector).
# fp32 scales scatter straight to gmem. Reciprocal-mul (v*(1/scale)) avoids per-element divides.
# ~41% of B200 peak at 16384 (beats compile 35.8%, below triton 57.8% and the mxfp8_floor_dim_km
# sibling's 57.4%). Two bottlenecks, BOTH worse than the 1x32 sibling and hard to fix here (ncu):
# (1) the dim-K row reads are 32-WAY bank-conflicted (vs 16-way at 1x32) -- a 128-col bf16 block is
# bank-aligned, so every lane of a thread-per-block read hits the same bank regardless of tile/mapping;
# (2) doing both 128-reductions per thread costs 154 reg/thread -> only 3 CTAs/SM (18.75% occ). But
# occupancy is NOT the lever (L1/TEX ~86% is), and NEITHER is the conflict -- FOUR restructurings all
# REGRESSED vs this simple interleaved-per-thread version, because they trade away its ILP/MLP:
# warp-split (1 dir/thread, ~72 reg, 2x occ) -> 35%; dynamic non-unrolled chunk loops -> 30%;
# warp-cooperative conflict-FREE dim-K (32 lanes read 128 consecutive cols, 4/lane + warp_reduction_max)
# -> 27%. An XOR-swizzled sIN (the textbook conflict fix) could not be wired through this repo's
# cpasync TMA path -- make_tiled_tma_atom accepts the composed swizzle layout but cpasync.tma_partition
# then fails to partition it against the gmem tile ("unable to partition input tensors for TMA").
# So 41% stands: this kernel is bound by per-thread ILP over the interleaved reductions, not by the
# bank conflicts the L1/TEX metric flags.
# ---------------------------------------------------------------------------
_DKKM_TM, _DKKM_TN, _DKKM_WARPS = 128, 128, 4      # tuned on B200 @ 16384 (needs M%TM==0, N%TN==0)
_DKKM_THREADS = _DKKM_WARPS * 32
_DKKM_KB = _DKKM_TN // 128                          # 128-col blocks per row (dim-K)
_DKKM_CHUNKS = 128 // 32                            # 32-wide chunks per 128 block (vectorize)
_DKKM_GROUPS = _DKKM_TM * _DKKM_TN // 128           # scale groups per tile (same count each direction)
_DKKM_ITERS = (_DKKM_GROUPS + _DKKM_THREADS - 1) // _DKKM_THREADS
_DKKM_IN_BYTES = _DKKM_TM * _DKKM_TN * 2            # bf16 tile bytes for the TMA expect-tx


@cute.struct
class _DeepseekDimKmSmem:
    bar: cute.struct.MemRange[cutlass.Int64, 1]
    sin: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _DKKM_TM * _DKKM_TN], 1024]
    soutm: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, _DKKM_TM * _DKKM_TN], 1024]


@cute.kernel
def _deepseek_dim_km_kernel(atom_in: cute.CopyAtom, ten_in: cute.Tensor,
                            atom_m: cute.CopyAtom, ten_m: cute.Tensor, mK: cute.Tensor,
                            sk: cute.Tensor, sm: cute.Tensor, sil: cute.Layout, solm: cute.Layout,
                            M: cutlass.Int64, N: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()   # bidx = m_tile, bidy = n_tile
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    st = smem.allocate(_DeepseekDimKmSmem)
    bar = st.bar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    sIN = st.sin.get_tensor(sil)             # (TM, TN) row-major
    sOUTm = st.soutm.get_tensor(solm)        # (TN, TM) row-major (dim-M transpose)
    gIN = cute.local_tile(ten_in, (_DKKM_TM, _DKKM_TN), (None, None))
    gM = cute.local_tile(ten_m, (_DKKM_TN, _DKKM_TM), (None, None))
    tAsA, tAgA = cpasync.tma_partition(atom_in, 0, cute.make_layout(1),
                                       cute.group_modes(sIN, 0, 2), cute.group_modes(gIN, 0, 2))
    tMsM, tMgM = cpasync.tma_partition(atom_m, 0, cute.make_layout(1),
                                       cute.group_modes(sOUTm, 0, 2), cute.group_modes(gM, 0, 2))
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, _DKKM_IN_BYTES)
        cute.copy(atom_in, tAgA[(None, bidx, bidy)], tAsA, tma_bar_ptr=bar)
    cute.arch.mbarrier_wait(bar, 0)

    m0 = bidx * _DKKM_TM
    n0 = bidy * _DKKM_TN
    mblk = M // 128                          # sm is (N, M//128)
    nblk = N // 128                          # sk is (M, N//128)
    st_k = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mK.element_type, num_bits_per_copy=128)
    for it in cutlass.range_constexpr(_DKKM_ITERS):
        g = tidx + it * _DKKM_THREADS
        if g < _DKKM_GROUPS:
            # --- dim-M: 128-row column block -> transposed store into sOUTm (TMA-stored at the end)
            col = g % _DKKM_TN
            rb = g // _DKKM_TN
            r0 = rb * 128
            amax_m = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(_DKKM_CHUNKS):     # amax over 128 rows, 4x32
                frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                for r in cutlass.range_constexpr(32):
                    frg[r] = sIN[r0 + c * 32 + r, col].to(cutlass.Float32)
                v = frg.load()
                amax_m = cutlass.max(amax_m, cute.where(v < 0, -v, v).reduce(
                    cute.ReductionOp.MAX, cutlass.Float32(0.0), 0))
            scale_m = cutlass.max(amax_m, cutlass.Float32(1e-12)) / 448.0
            inv_m = 1.0 / scale_m
            for c in cutlass.range_constexpr(_DKKM_CHUNKS):     # re-read + quantize + transpose write
                frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                for r in cutlass.range_constexpr(32):
                    frg[r] = sIN[r0 + c * 32 + r, col].to(cutlass.Float32)
                frgO = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
                frgO.store((frg.load() * inv_m).to(cutlass.Float8E4M3FN))
                for r in cutlass.range_constexpr(32):
                    sOUTm[col, r0 + c * 32 + r] = frgO[r]
            sm[(n0 + col) * mblk + (m0 // 128 + rb)] = scale_m.to(sm.element_type)

            # --- dim-K: 128-col row block. qk keeps x's layout, so quantize in registers and store
            # each 32-chunk directly to gmem with a 128-bit vectorized copy (no transpose, no smem).
            kb = g % _DKKM_KB
            row = g // _DKKM_KB
            c0 = kb * 128
            amax_k = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(_DKKM_CHUNKS):     # amax over 128 cols, 4x32
                frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                for j in cutlass.range_constexpr(32):
                    frg[j] = sIN[row, c0 + c * 32 + j].to(cutlass.Float32)
                v = frg.load()
                amax_k = cutlass.max(amax_k, cute.where(v < 0, -v, v).reduce(
                    cute.ReductionOp.MAX, cutlass.Float32(0.0), 0))
            scale_k = cutlass.max(amax_k, cutlass.Float32(1e-12)) / 448.0
            inv_k = 1.0 / scale_k
            base = (m0 + row) * N + (n0 + c0)
            for c in cutlass.range_constexpr(_DKKM_CHUNKS):     # re-read + quantize + direct gmem store
                frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                for j in cutlass.range_constexpr(32):
                    frg[j] = sIN[row, c0 + c * 32 + j].to(cutlass.Float32)
                frgO = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
                frgO.store((frg.load() * inv_k).to(cutlass.Float8E4M3FN))
                off = cute.assume(base + c * 32, divby=32)
                gk = cute.make_tensor(mK.iterator + off, cute.make_layout(32))
                cute.copy(st_k, frgO, gk)
            sk[(m0 + row) * nblk + (n0 // 128 + kb)] = scale_k.to(sk.element_type)

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
    if warp == 0:
        cute.copy(atom_m, tMsM, tMgM[(None, bidy, bidx)])         # qm tile (n_tile, m_tile)


@cute.jit
def _deepseek_dim_km_jit(mIN, mM, mK, sk, sm, M: cutlass.Constexpr, N: cutlass.Constexpr):
    sil = cute.make_layout((_DKKM_TM, _DKKM_TN), stride=(_DKKM_TN, 1))
    solm = cute.make_layout((_DKKM_TN, _DKKM_TM), stride=(_DKKM_TM, 1))   # dim-M transposed
    atom_in, ten_in = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mIN, sil, (_DKKM_TM, _DKKM_TN))
    atom_m, ten_m = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mM, solm, (_DKKM_TN, _DKKM_TM))
    M2, N2 = mIN.shape
    grid = ((M2 + _DKKM_TM - 1) // _DKKM_TM, (N2 + _DKKM_TN - 1) // _DKKM_TN, 1)
    _deepseek_dim_km_kernel(atom_in, ten_in, atom_m, ten_m, mK, sk, sm, sil, solm,
                            cutlass.Int64(M), cutlass.Int64(N)).launch(
        grid=grid, block=(_DKKM_THREADS, 1, 1), cluster=(1, 1, 1))


def fp8_deepseek_1x128_dim_km_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % _DKKM_TM == 0 and N % _DKKM_TN == 0, \
        f"deepseek_1x128_dim_km cute kernel needs M%{_DKKM_TM}==0 and N%{_DKKM_TN}==0"
    yk = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)              # dim-K (M,N)
    sk_t = torch.empty(M, N // 128, dtype=torch.float32, device=x.device)
    ym = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)             # dim-M (N,M) transp
    sm_t = torch.empty(N, M // 128, dtype=torch.float32, device=x.device)
    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-elem aligned).
    mIN = (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
           .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mM = (from_dlpack(ym, assumed_align=16).mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mK = from_dlpack(yk.reshape(-1), assumed_align=16).mark_layout_dynamic()  # direct gmem store
    sk = from_dlpack(sk_t.reshape(-1)).mark_layout_dynamic()
    sm = from_dlpack(sm_t.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("deepseek_1x128_dim_km", M, N), _deepseek_dim_km_jit, mIN, mM, mK, sk, sm, M, N)
    fn(mIN, mM, mK, sk, sm)
    return yk, sk_t, ym, sm_t


FP8_DEEPSEEK_1X128_DIM_KM = QuantCastCuteRecipe.from_gold(
    Deepseek1x128DimKmGold, cute_fn=fp8_deepseek_1x128_dim_km_cute
)


# ---------------------------------------------------------------------------
# e8m0 FLOOR helper (device): amax (f32 scalar) -> (sfp, biased_uint8). Replicates gold's
# _amax_to_e8m0_floor + _e8m0_to_fp32 exactly via bit extraction (recast_tensor bitcast), because
# `.to(Float8E8M0FNU)` rounds UP. `biased` is the e8m0 byte; `sfp` is the exact fp32 pow2 factor.
# ---------------------------------------------------------------------------
@cute.jit
def _e8m0_floor(amax):
    fbits = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)
    fbits[0] = amax
    ibits = cute.recast_tensor(fbits, dtype=cutlass.Int32)
    extracted = ((ibits[0] >> 23) & 0xFF) - 127
    unbiased = cutlass.min(cutlass.max(extracted - 8, cutlass.Int32(-127)), cutlass.Int32(128))
    biased = unbiased + 127
    ib = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Int32)
    ib[0] = biased << 23
    fb = cute.recast_tensor(ib, dtype=cutlass.Float32)
    return cutlass.max(fb[0], cutlass.Float32(2.0 ** -126)), biased


# ---------------------------------------------------------------------------
# mxfp8 FLOOR 1x32: one e8m0 scale per 1x32 block. One thread per block (owns 32 vals -> fp8
# aligned, intra-thread abs-max reduce). Tile width 512 = BPT blocks; scale (M,N//32) tiled (1,BPT)
# shares bidx with x tiled (1,512). Mirrors mxfp8_floor_f.
# ---------------------------------------------------------------------------
_MXFP8_1X32_BPT = 16  # 512 // 32


@cute.kernel
def _mxfp8_floor_kernel(gX: cute.Tensor, gY: cute.Tensor, gScale: cute.Tensor,
                        tv_layout: cute.Layout, s_tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    thrX = cute.composition(gX[(None, bidx)], tv_layout)[(tidx, None)]
    thrY = cute.composition(gY[(None, bidx)], tv_layout)[(tidx, None)]
    thrS = cute.composition(gScale[(None, bidx)], s_tv_layout)[(tidx, None)]
    frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type), thrX, frgX)
    v = frgX.load().to(cutlass.Float32)
    amax = cutlass.max(v, -v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    sfp, biased = _e8m0_floor(amax)
    frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
    frgY.store((v / sfp).to(gY.element_type))
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type), frgY, thrY)
    frgS = cute.make_rmem_tensor(cute.make_layout(1), gScale.element_type)
    frgS[0] = biased.to(gScale.element_type)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gScale.element_type), frgS, thrS)


@cute.jit
def _mxfp8_floor_jit(mX, mY, mScale):
    bpt = _MXFP8_1X32_BPT
    tv_layout = cute.make_layout((bpt, 32), stride=(32, 1))
    s_tv_layout = cute.make_layout((bpt, 1), stride=(1, 1))
    gX = cute.zipped_divide(mX, (1, bpt * 32))
    gY = cute.zipped_divide(mY, (1, bpt * 32))
    gScale = cute.zipped_divide(mScale, (1, bpt))
    _mxfp8_floor_kernel(gX, gY, gScale, tv_layout, s_tv_layout).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[bpt, 1, 1])


def mxfp8_floor_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    s_u8 = torch.empty(M, N // 32, dtype=torch.uint8, device=x.device)
    mX = from_dlpack(x).mark_layout_dynamic()
    mY = from_dlpack(y).mark_layout_dynamic()
    mS = from_dlpack(s_u8).mark_layout_dynamic()
    fn = _compiled(("mxfp8_floor", M, N), _mxfp8_floor_jit, mX, mY, mS)
    fn(mX, mY, mS)
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_FLOOR = QuantCastCuteRecipe.from_gold(Mxfp8FloorGold, cute_fn=mxfp8_floor_cute)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR dim-M: 32-row blocks down M, transposed outputs (N, M) / (N, M//32). The reduction is
# down columns of x, and the output is transposed -- feeding x.t() to mxfp8_floor makes EVERY load
# uncoalesced. A hand-rolled coalesced-load + smem-transpose + coalesced-store kernel (1 warp/32x32
# tile) only reached ~35% of peak: the memory transactions were scalar CopyUniversalOp, and more
# warps/block did NOT help (occupancy was not the binding lever). The win is TMA: this kernel uses
# the canonical warp-specialized bulk-tensor path (like torchao's cutedsl quantize_2d) --
#   - TMA G2S loads a (TM, TN) row-major tile of x into smem (sIN);
#   - the (TM/32 x TN) scale-groups are split across all threads; each owns one (32-row block, col),
#     reads its 32 rows down a column of sIN (warp-coalesced, conflict-free), reduces to a per-column
#     amax, computes the e8m0 FLOOR scale, quantizes its 32 values, and writes them as a CONTIGUOUS
#     run into sOUT laid out (TN, TM) row-major -- i.e. the transpose happens in the register->smem
#     write, so no smem-transpose/padding dance and no col-major TMA (which the DSL can't drive);
#   - TMA S2G stores sOUT into the (TN, TM) tile of the row-major (N, M) output at (n_tile, m_tile).
# The e8m0 byte is scattered straight to gmem scales (N, M//32). Both TMA transfers are standard
# row-major, and the transposed store rides the TMA engine at DRAM speed. ~62% of B200 peak at
# 16384 (vs ~35% hand-rolled, ~60% triton, ~68% CUDA SOL). Barrier follows the arrive-and-expect-tx
# pattern (single arrival, warp-0 gated) required for a multi-warp block.
# ---------------------------------------------------------------------------
_DIMM_TM, _DIMM_TN, _DIMM_WARPS = 64, 256, 8      # tuned on B200 @ 16384 (needs M%TM==0, N%TN==0)
_DIMM_THREADS = _DIMM_WARPS * 32
_DIMM_RB = _DIMM_TM // 32                          # 32-row blocks per tile
_DIMM_GROUPS = _DIMM_TN * _DIMM_RB                 # (col, row-block) scale groups
_DIMM_ITERS = (_DIMM_GROUPS + _DIMM_THREADS - 1) // _DIMM_THREADS
_DIMM_IN_BYTES = _DIMM_TM * _DIMM_TN * 2           # bf16 tile bytes for the TMA expect-tx


@cute.struct
class _Mxfp8DimMSmem:
    bar: cute.struct.MemRange[cutlass.Int64, 1]
    sin: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _DIMM_TM * _DIMM_TN], 1024]
    sout: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, _DIMM_TM * _DIMM_TN], 1024]


@cute.kernel
def _mxfp8_dim_m_kernel(atom_in: cute.CopyAtom, ten_in: cute.Tensor, atom_out: cute.CopyAtom,
                        ten_out: cute.Tensor, scales: cute.Tensor, sil: cute.Layout,
                        sol: cute.Layout, M: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()   # bidx = m_tile, bidy = n_tile
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    st = smem.allocate(_Mxfp8DimMSmem)
    bar = st.bar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    sIN = st.sin.get_tensor(sil)                             # (TM, TN) row-major
    sOUT = st.sout.get_tensor(sol)                           # (TN, TM) row-major (transposed)
    gIN = cute.local_tile(ten_in, (_DIMM_TM, _DIMM_TN), (None, None))
    gOUT = cute.local_tile(ten_out, (_DIMM_TN, _DIMM_TM), (None, None))
    tAsA, tAgA = cpasync.tma_partition(atom_in, 0, cute.make_layout(1),
                                       cute.group_modes(sIN, 0, 2), cute.group_modes(gIN, 0, 2))
    tOsO, tBgB = cpasync.tma_partition(atom_out, 0, cute.make_layout(1),
                                       cute.group_modes(sOUT, 0, 2), cute.group_modes(gOUT, 0, 2))
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, _DIMM_IN_BYTES)
        cute.copy(atom_in, tAgA[(None, bidx, bidy)], tAsA, tma_bar_ptr=bar)
    cute.arch.mbarrier_wait(bar, 0)

    m0 = bidx * _DIMM_TM
    n0 = bidy * _DIMM_TN
    mblk = M // 32
    for it in cutlass.range_constexpr(_DIMM_ITERS):
        g = tidx + it * _DIMM_THREADS
        if g < _DIMM_GROUPS:
            col = g % _DIMM_TN
            rb = g // _DIMM_TN
            r0 = rb * 32
            frgIn = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
            for r in cutlass.range_constexpr(32):
                frgIn[r] = sIN[r0 + r, col].to(cutlass.Float32)   # column read (warp-coalesced)
            v = frgIn.load()
            amax = cute.where(v < 0, -v, v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
            sfp, biased = _e8m0_floor(amax)
            frgOut = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
            frgOut.store((v * (1.0 / sfp)).to(cutlass.Float8E4M3FN))  # reciprocal-mul, not 32 divs
            for r in cutlass.range_constexpr(32):
                sOUT[col, r0 + r] = frgOut[r]                     # contiguous run = the transpose
            scales[(n0 + col) * mblk + (m0 // 32 + rb)] = biased.to(scales.element_type)

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
    if warp == 0:
        cute.copy(atom_out, tOsO, tBgB[(None, bidy, bidx)])       # y tile (n_tile, m_tile)


@cute.jit
def _mxfp8_dim_m_jit(mIN, mOUT, scales, M: cutlass.Constexpr):
    sil = cute.make_layout((_DIMM_TM, _DIMM_TN), stride=(_DIMM_TN, 1))
    sol = cute.make_layout((_DIMM_TN, _DIMM_TM), stride=(_DIMM_TM, 1))
    atom_in, ten_in = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mIN, sil, (_DIMM_TM, _DIMM_TN))
    atom_out, ten_out = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mOUT, sol, (_DIMM_TN, _DIMM_TM))
    M2, N2 = mIN.shape
    grid = ((M2 + _DIMM_TM - 1) // _DIMM_TM, (N2 + _DIMM_TN - 1) // _DIMM_TN, 1)
    _mxfp8_dim_m_kernel(atom_in, ten_in, atom_out, ten_out, scales, sil, sol,
                        cutlass.Int64(M)).launch(grid=grid, block=(_DIMM_THREADS, 1, 1),
                                                 cluster=(1, 1, 1))


def mxfp8_floor_dim_m_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % _DIMM_TM == 0 and N % _DIMM_TN == 0, \
        f"mxfp8_floor_dim_m cute kernel needs M%{_DIMM_TM}==0 and N%{_DIMM_TN}==0"
    y = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)  # transposed row-major output
    s_u8 = torch.empty(N, M // 32, dtype=torch.uint8, device=x.device)
    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-elem aligned).
    mIN = (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
           .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mOUT = (from_dlpack(y, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=16))
    scales = from_dlpack(s_u8.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("mxfp8_floor_dim_m", M, N), _mxfp8_dim_m_jit, mIN, mOUT, scales, M)
    fn(mIN, mOUT, scales)
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_FLOOR_DIM_M = QuantCastCuteRecipe.from_gold(
    Mxfp8FloorDimMGold, cute_fn=mxfp8_floor_dim_m_cute
)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR in BOTH directions, one pass over x (mirrors mxfp8_floor_dim_km_f / the fused triton
# kernel). Four outputs: dim-K (qk (M,N), sk (M,N//32) -- 1x32 blocks along columns, like mxfp8_floor)
# and dim-M (qm (N,M), sm (N,M//32) -- 32x1 blocks down rows, transposed, like mxfp8_floor_dim_m).
#
# The fused version of the TMA BMxBN template: TMA G2S loads one (TM, TN) row-major tile of x into
# smem (sIN), read once and reduced BOTH ways, then two TMA S2G stores emit the two quantized tiles:
#   - dim-M (the binding cost): the TM*TN/32 (col, 32-row-block) groups are split across all threads;
#     each reads its 32 rows DOWN a column of sIN, reduces to the per-column amax, e8m0-FLOOR scales,
#     quantizes, and writes the 32 values as a CONTIGUOUS run into sOUT_M laid out (TN, TM) -- the
#     transpose happens in the register->smem write (no col-major TMA). TMA-store sOUT_M into the
#     (TN, TM) tile of the row-major (N, M) output = the dim-M transpose.
#   - dim-K (rides the already-loaded tile ~free): the TM*TN/32 (row, 32-col-block) groups are split
#     the same way; each reads its 32 ALONG a row of sIN, reduces to the per-row amax, e8m0 scales,
#     quantizes, and -- since qk keeps x's (M,N) layout (no transpose) -- writes the contiguous 32-run
#     DIRECTLY to gmem with a 128-bit vectorized copy (adjacent threads = adjacent col-blocks of the
#     same row -> coalesced). Keeping qk out of smem is what lifted this from 52% to 57%: it frees
#     16 KB -> +1 CTA/SM (occupancy 37.5% -> 50%) and drops the dim-K transpose-store bank conflicts.
# Both scale bytes scatter straight to gmem. e8m0 is pow2 so v*(1/sfp) == v/sfp bit-exactly. This
# replaces a naive 32x32-tile / 1-warp kernel (18.9%, instruction-bound on 32 serial warp-reduces +
# per-element stores). ~57% of B200 peak at 16384 (beats triton 47.1%, nears standalone dim_m 60.3%).
# The remaining ceiling is L1/TEX (ncu ~82%): the dim-K row reads are >=16-way bank-conflicted because
# a 32-col bf16 block is exactly 16 banks wide, so thread-per-block reads collapse to 2 bank groups
# regardless of tile/mapping (bank depends only on column when TN is a multiple of 64). Killing that
# needs a SWIZZLED smem layout for sIN (XOR swizzle, as CUTLASS GEMM uses) -- the real next step, but
# it would also have to keep the dim-M column reads conflict-free, so it is left as future work.
# ---------------------------------------------------------------------------
_DKM_TM, _DKM_TN, _DKM_WARPS = 64, 256, 8         # tuned on B200 @ 16384 (needs M%TM==0, N%TN==0)
_DKM_THREADS = _DKM_WARPS * 32
_DKM_GROUPS = _DKM_TM * _DKM_TN // 32             # scale groups per tile (same count each direction)
_DKM_ITERS = (_DKM_GROUPS + _DKM_THREADS - 1) // _DKM_THREADS
_DKM_KB = _DKM_TN // 32                           # 32-col blocks per row (dim-K)
_DKM_IN_BYTES = _DKM_TM * _DKM_TN * 2             # bf16 tile bytes for the TMA expect-tx


@cute.struct
class _Mxfp8DimKmSmem:
    bar: cute.struct.MemRange[cutlass.Int64, 1]
    sin: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _DKM_TM * _DKM_TN], 1024]
    soutm: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, _DKM_TM * _DKM_TN], 1024]


@cute.kernel
def _mxfp8_dim_km_kernel(atom_in: cute.CopyAtom, ten_in: cute.Tensor,
                         atom_m: cute.CopyAtom, ten_m: cute.Tensor, mK: cute.Tensor,
                         sk: cute.Tensor, sm: cute.Tensor, sil: cute.Layout, solm: cute.Layout,
                         M: cutlass.Int64, N: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()   # bidx = m_tile, bidy = n_tile
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    st = smem.allocate(_Mxfp8DimKmSmem)
    bar = st.bar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    sIN = st.sin.get_tensor(sil)             # (TM, TN) row-major
    sOUTm = st.soutm.get_tensor(solm)        # (TN, TM) row-major (dim-M transpose)
    gIN = cute.local_tile(ten_in, (_DKM_TM, _DKM_TN), (None, None))
    gM = cute.local_tile(ten_m, (_DKM_TN, _DKM_TM), (None, None))
    tAsA, tAgA = cpasync.tma_partition(atom_in, 0, cute.make_layout(1),
                                       cute.group_modes(sIN, 0, 2), cute.group_modes(gIN, 0, 2))
    tMsM, tMgM = cpasync.tma_partition(atom_m, 0, cute.make_layout(1),
                                       cute.group_modes(sOUTm, 0, 2), cute.group_modes(gM, 0, 2))
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, _DKM_IN_BYTES)
        cute.copy(atom_in, tAgA[(None, bidx, bidy)], tAsA, tma_bar_ptr=bar)
    cute.arch.mbarrier_wait(bar, 0)

    m0 = bidx * _DKM_TM
    n0 = bidy * _DKM_TN
    mblk = M // 32                           # sm is (N, M//32)
    nblk = N // 32                           # sk is (M, N//32)
    st_k = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mK.element_type, num_bits_per_copy=128)
    for it in cutlass.range_constexpr(_DKM_ITERS):
        g = tidx + it * _DKM_THREADS
        if g < _DKM_GROUPS:
            # --- dim-M: 32-row column block -> transposed store into sOUTm (TMA-stored at the end)
            col = g % _DKM_TN
            rb = g // _DKM_TN
            r0 = rb * 32
            frgM = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
            for r in cutlass.range_constexpr(32):
                frgM[r] = sIN[r0 + r, col].to(cutlass.Float32)   # column read (down the tile)
            vm = frgM.load()
            amax_m = cute.where(vm < 0, -vm, vm).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
            sfp_m, biased_m = _e8m0_floor(amax_m)
            frgOm = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
            frgOm.store((vm * (1.0 / sfp_m)).to(cutlass.Float8E4M3FN))
            for r in cutlass.range_constexpr(32):
                sOUTm[col, r0 + r] = frgOm[r]                     # contiguous run = the transpose
            sm[(n0 + col) * mblk + (m0 // 32 + rb)] = biased_m.to(sm.element_type)

            # --- dim-K: 32-col row block. qk keeps x's layout (no transpose), so quantize in
            # registers and store the contiguous 32-run DIRECTLY to gmem with a 128-bit vectorized
            # copy (adjacent threads = adjacent col-blocks of the same row -> coalesced). Keeping qk
            # out of smem frees 16 KB -> +1 CTA/SM and drops the sOUTk transpose bank conflicts.
            kb = g % _DKM_KB
            row = g // _DKM_KB
            c0 = kb * 32
            frgK = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
            for c in cutlass.range_constexpr(32):
                frgK[c] = sIN[row, c0 + c].to(cutlass.Float32)    # row read (along the tile)
            vk = frgK.load()
            amax_k = cute.where(vk < 0, -vk, vk).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
            sfp_k, biased_k = _e8m0_floor(amax_k)
            frgOk = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
            frgOk.store((vk * (1.0 / sfp_k)).to(cutlass.Float8E4M3FN))
            off = cute.assume((m0 + row) * N + (n0 + c0), divby=32)
            gk = cute.make_tensor(mK.iterator + off, cute.make_layout(32))
            cute.copy(st_k, frgOk, gk)
            sk[(m0 + row) * nblk + (n0 // 32 + kb)] = biased_k.to(sk.element_type)

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
    if warp == 0:
        cute.copy(atom_m, tMsM, tMgM[(None, bidy, bidx)])         # qm tile (n_tile, m_tile)


@cute.jit
def _mxfp8_dim_km_jit(mIN, mM, mK, sk, sm, M: cutlass.Constexpr, N: cutlass.Constexpr):
    sil = cute.make_layout((_DKM_TM, _DKM_TN), stride=(_DKM_TN, 1))
    solm = cute.make_layout((_DKM_TN, _DKM_TM), stride=(_DKM_TM, 1))   # dim-M transposed
    atom_in, ten_in = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mIN, sil, (_DKM_TM, _DKM_TN))
    atom_m, ten_m = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mM, solm, (_DKM_TN, _DKM_TM))
    M2, N2 = mIN.shape
    grid = ((M2 + _DKM_TM - 1) // _DKM_TM, (N2 + _DKM_TN - 1) // _DKM_TN, 1)
    _mxfp8_dim_km_kernel(atom_in, ten_in, atom_m, ten_m, mK, sk, sm, sil, solm,
                         cutlass.Int64(M), cutlass.Int64(N)).launch(
        grid=grid, block=(_DKM_THREADS, 1, 1), cluster=(1, 1, 1))


def mxfp8_floor_dim_km_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % _DKM_TM == 0 and N % _DKM_TN == 0, \
        f"mxfp8_floor_dim_km cute kernel needs M%{_DKM_TM}==0 and N%{_DKM_TN}==0"
    yk = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)              # dim-K (M,N)
    sk_u8 = torch.empty(M, N // 32, dtype=torch.uint8, device=x.device)
    ym = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)             # dim-M (N,M) transp
    sm_u8 = torch.empty(N, M // 32, dtype=torch.uint8, device=x.device)
    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-elem aligned).
    mIN = (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
           .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mM = (from_dlpack(ym, assumed_align=16).mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mK = from_dlpack(yk.reshape(-1), assumed_align=16).mark_layout_dynamic()  # direct gmem store
    sk = from_dlpack(sk_u8.reshape(-1)).mark_layout_dynamic()
    sm = from_dlpack(sm_u8.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("mxfp8_floor_dim_km", M, N), _mxfp8_dim_km_jit, mIN, mM, mK, sk, sm, M, N)
    fn(mIN, mM, mK, sk, sm)
    return yk, sk_u8.view(torch.float8_e8m0fnu), ym, sm_u8.view(torch.float8_e8m0fnu)


MXFP8_FLOOR_DIM_KM = QuantCastCuteRecipe.from_gold(
    Mxfp8FloorDimKmGold, cute_fn=mxfp8_floor_dim_km_cute
)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR 32x32: one e8m0 scale per 32x32 block (amax over the whole block, e8m0-FLOOR pow2
# scale). Like deepseek_128x128 this is a NON-transposing block reduction, so it takes the same TMA
# path (both tiles row-major (TM,TN), no register->smem transpose). Went 37.1% -> 70.9% of B200 peak
# (beats compile 26.5%, matches the deepseek_128x128 sibling's ~70% ceiling; triton 76.5%). The
# 32x32 block maps perfectly to a warp: 32 lanes x 32 rows = 1024 = a full block, so a WARP OWNS A
# BLOCK and the block amax is a single warp_reduction_max -- no cross-warp scratch, no inter-phase
# sync (simpler than deepseek_128x128). Tile size is decoupled from warp count (a small 64x64 tile
# is TMA-starved, 45%): a big (TM,TN)=128x128 tile (32KB, matching deepseek's sweet spot) holds
# NBLK=16 blocks, and each of the WARPS=8 warps LOOPS over NBLK//WARPS blocks.
#   - TMA G2S loads the (TM,TN) tile into sIN;
#   - warp w handles blocks {w, w+WARPS, ...}; for each, lane `l` owns COLUMN l of the block (rows
#     0..31) -> consecutive lanes hit consecutive smem cols = conflict-free (bank note in
#     [[cute-tma-transpose-quant]]); lane reduces its 32-row column, warp_reduction_max -> block
#     amax -> e8m0 floor scale; each lane quantizes its column (vectorized f32->fp8) into sOUT;
#     lane0 writes the e8m0 byte;
#   - TMA S2G stores sOUT into the (TM,TN) tile of the row-major output.
# TWO decisive findings (both took the kernel from ~38% -> 70%): (1) `v / sfp` on a 32-wide vector
# emits 32 per-element DIVISIONS (168M insts, 38%); `inv = 1/sfp; v * inv` is bit-exact for a pow2
# e8m0 scale and cuts to 95M insts (matches deepseek) -> 70%. (2) WARPS=8 (not 4): smem caps
# occupancy at 4 CTAs/SM regardless of thread count, so 8 warps/CTA doubles resident warps to hide
# the TMA-load latency (long_scoreboard-bound). Tried but REJECTED: a direct coalesced fp8 GLOBAL
# store (drop sOUT smem, halve footprint, raise occupancy) -> only 50.5%; TMA store of the whole
# tile beats scattered 32-byte fp8 sectors even at lower occupancy. Bigger tiles (128x256, 256x256)
# are smem-occupancy-starved (63%, 47%).
# ---------------------------------------------------------------------------
_M32_TM, _M32_TN = 128, 128                    # TMA tile (tuned on B200 @ 16384)
_M32_WARPS = 8                                 # warps/CTA (tuned; more warps hide TMA-load latency)
_M32_THREADS = _M32_WARPS * 32
_M32_BR = _M32_TM // 32                         # block-rows per tile
_M32_BC = _M32_TN // 32                         # block-cols per tile
_M32_NBLK = _M32_BR * _M32_BC                   # 32x32 blocks per tile
_M32_BPW = _M32_NBLK // _M32_WARPS              # blocks each warp loops over
_M32_BYTES = _M32_TM * _M32_TN * 2              # bf16 tile bytes for TMA expect-tx


@cute.struct
class _Mxfp832Smem:
    bar: cute.struct.MemRange[cutlass.Int64, 1]
    sin: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _M32_TM * _M32_TN], 1024]
    sout: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, _M32_TM * _M32_TN], 1024]


@cute.kernel
def _mxfp8_32x32_kernel(atom_in: cute.CopyAtom, ten_in: cute.Tensor, atom_out: cute.CopyAtom,
                        ten_out: cute.Tensor, scales: cute.Tensor, sil: cute.Layout,
                        sol: cute.Layout, ncb: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()   # bidx = m_tile, bidy = n_tile
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    smem = utils.SmemAllocator()
    st = smem.allocate(_Mxfp832Smem)
    bar = st.bar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    sIN = st.sin.get_tensor(sil)
    sOUT = st.sout.get_tensor(sol)
    gIN = cute.local_tile(ten_in, (_M32_TM, _M32_TN), (None, None))
    gOUT = cute.local_tile(ten_out, (_M32_TM, _M32_TN), (None, None))
    tAsA, tAgA = cpasync.tma_partition(atom_in, 0, cute.make_layout(1),
                                       cute.group_modes(sIN, 0, 2), cute.group_modes(gIN, 0, 2))
    tOsO, tBgB = cpasync.tma_partition(atom_out, 0, cute.make_layout(1),
                                       cute.group_modes(sOUT, 0, 2), cute.group_modes(gOUT, 0, 2))
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, _M32_BYTES)
        cute.copy(atom_in, tAgA[(None, bidx, bidy)], tAsA, tma_bar_ptr=bar)
    cute.arch.mbarrier_wait(bar, 0)

    # warp handles blocks {warp, warp+WARPS, ...}; lane l owns column l of a block (32 rows deep).
    for bi in cutlass.range_constexpr(_M32_BPW):
        blk = warp + bi * _M32_WARPS
        br = blk // _M32_BC
        bc = blk % _M32_BC
        r0 = br * 32
        c0 = bc * 32 + lane
        frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
        for i in cutlass.range_constexpr(32):
            frg[i] = sIN[r0 + i, c0].to(cutlass.Float32)
        v = frg.load()
        local = cute.where(v < 0, -v, v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
        amax = cute.arch.warp_reduction_max(local)       # across the 32 columns -> whole-block amax
        sfp, biased = _e8m0_floor(amax)
        inv = 1.0 / sfp                                   # pow2 scale: v*(1/sfp) is bit-exact vs v/sfp
        if lane == 0:
            gbr = bidx * _M32_BR + br
            gbc = bidy * _M32_BC + bc
            scales[gbr * ncb + gbc] = biased.to(scales.element_type)
        frgOut = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
        frgOut.store((v * inv).to(cutlass.Float8E4M3FN))  # reciprocal-mul, not 32 per-element divs
        for i in cutlass.range_constexpr(32):
            sOUT[r0 + i, c0] = frgOut[i]

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
    if warp == 0:
        cute.copy(atom_out, tOsO, tBgB[(None, bidx, bidy)])


@cute.jit
def _mxfp8_32x32_jit(mIN, mOUT, scales, ncb: cutlass.Constexpr):
    sil = cute.make_layout((_M32_TM, _M32_TN), stride=(_M32_TN, 1))
    sol = cute.make_layout((_M32_TM, _M32_TN), stride=(_M32_TN, 1))
    atom_in, ten_in = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mIN, sil, (_M32_TM, _M32_TN))
    atom_out, ten_out = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mOUT, sol, (_M32_TM, _M32_TN))
    M2, N2 = mIN.shape
    grid = ((M2 + _M32_TM - 1) // _M32_TM, (N2 + _M32_TN - 1) // _M32_TN, 1)
    _mxfp8_32x32_kernel(atom_in, ten_in, atom_out, ten_out, scales, sil, sol,
                        ncb).launch(grid=grid, block=(_M32_THREADS, 1, 1), cluster=(1, 1, 1))


def mxfp8_32x32_floor_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % _M32_TM == 0 and N % _M32_TN == 0, \
        f"mxfp8_32x32 cute kernel needs M%{_M32_TM}==0 and N%{_M32_TN}==0"
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    ncb = N // 32
    s_u8 = torch.empty(M // 32, ncb, dtype=torch.uint8, device=x.device)
    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-elem aligned).
    mX = (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mY = (from_dlpack(y, assumed_align=16).mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mS = from_dlpack(s_u8.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("mxfp8_32x32", M, N), _mxfp8_32x32_jit, mX, mY, mS, ncb)
    fn(mX, mY, mS)
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_32X32_FLOOR = QuantCastCuteRecipe.from_gold(
    Mxfp832x32FloorGold, cute_fn=mxfp8_32x32_floor_cute
)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR 1x32 with the e8m0 scale written into the NVIDIA-swizzled 4D grid (nrb, ncb, 32, 16).
# Same quant as mxfp8_floor (one thread per 1x32 block); the scale for pre-swizzle (row, col) lands
# at flat offset ((br*ncb+bc)*32 + b)*16 + (a*4+c4) where br=row//128, r128=row%128, a=r128//32,
# b=r128%32, bc=col//4, c4=col%4 (from _to_blocked_4d). We compute (row, col) from the flat block
# index and scatter the biased byte to a flat uint8 scale buffer. Mirrors mxfp8_floor_swizzle_f.
#
# The qdata output is NOT transposed (same (M, N) layout as x), so this is a streaming cast: read
# bf16, write fp8, plus a tiny scattered scale-byte write (numel/32 bytes ~= 0.4% of traffic). So it
# wants the SAME recipe that took fp8_tensorwise to DRAM speed-of-light -- FLATTEN to 1-D, 128
# threads/CTA, each owning a CONTIGUOUS run loaded/stored via 128-bit VECTORIZED copy atoms
# (`num_bits_per_copy` + `assumed_align=16`). We can't tile 2-D here: with a (TR, TN) tile,
# composition flattens the ROW mode fastest, so a linear thread-layout gives each thread a strided
# (multi-row) fragment that can't be vectorized. The 1-D flatten keeps fragments contiguous+compact;
# we recover the block's (row, col) arithmetically from its flat position (N is a per-shape compile
# constant, so p//N, p%N fold to shifts for power-of-two N) for the swizzled scale offset.
#
# Each thread owns one full 1x32 block (VPT=32), so the block reduction is thread-local (no cross-
# thread reduce). Because N % 32 == 0, every 32-aligned run stays within one row, so a thread's run
# IS exactly one 1x32 mxfp8 block. The old kernel launched only 16 threads/CTA with scalar copies
# (~9% peak); this is the mxfp8_floor analog of the 85.9% tensorwise kernel.
# ---------------------------------------------------------------------------
_SWZ_THREADS = 128
_SWZ_VPT = 32                             # one full 1x32 block per thread (fp8-aligned, local reduce)
_SWZ_CHUNK = _SWZ_THREADS * _SWZ_VPT      # 4096 elements per CTA (1-D tile); 128 blocks
_SWZ_LD_BITS = min(128, _SWZ_VPT * 16)    # bf16 input  -> 128-bit vectorized load
_SWZ_ST_BITS = min(128, _SWZ_VPT * 8)     # fp8 output  -> 128-bit vectorized store


# swizzle offset (device): pre-swizzle scale position (row, col) -> flat offset into the 4D
# (nrb, ncb, 32, 16) block grid, i.e. ((row//128 * ncb + col//4) * 32 + row%128%32) * 16
# + ((row%128)//32 * 4 + col%4). Exact port of _to_blocked_4d's index math (mirrors the triton
# swizzle kernels). `col` is the block-column index (32-group for mxfp8, 16-group for nvfp4).
@cute.jit
def _swizzle_flat(row, col, ncb: cutlass.Int32):
    br = row // 128
    r128 = row % 128
    a = r128 // 32
    b = r128 % 32
    bc = col // 4
    c4 = col % 4
    return ((br * ncb + bc) * 32 + b) * 16 + (a * 4 + c4)


@cute.kernel
def _mxfp8_floor_swizzle_kernel(gX: cute.Tensor, gY: cute.Tensor, sflat: cute.Tensor,
                                cX: cute.Tensor, tv_layout: cute.Layout,
                                N: cutlass.Constexpr, ncb: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    thrX = cute.composition(gX[(None, bidx)], tv_layout)[(tidx, None)]
    thrY = cute.composition(gY[(None, bidx)], tv_layout)[(tidx, None)]
    thrC = cute.composition(cX[(None, bidx)], tv_layout)[(tidx, None)]
    ld_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type,
                                  num_bits_per_copy=_SWZ_LD_BITS)
    st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type,
                                  num_bits_per_copy=_SWZ_ST_BITS)
    frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
    cute.copy(ld_atom, thrX, frgX)
    v = frgX.load().to(cutlass.Float32)
    amax = cutlass.max(v, -v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    sfp, biased = _e8m0_floor(amax)
    frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
    frgY.store((v * (1.0 / sfp)).to(gY.element_type))  # reciprocal-mul (pow2 scale -> bit-identical)
    cute.copy(st_atom, frgY, thrY)
    # pre-swizzle (row, 32-group col) from the block's flat position; scatter the e8m0 byte.
    p = thrC[0][0]                                       # flat index of this block's first element
    flat = _swizzle_flat(p // N, (p % N) // 32, cutlass.Int32(ncb))
    sflat[flat] = biased.to(sflat.element_type)


@cute.jit
def _mxfp8_floor_swizzle_jit(mX, mY, sflat, N: cutlass.Constexpr, ncb: cutlass.Constexpr):
    # 1-D tile: thread t owns the contiguous 32-run [t*32, t*32+32) of the CTA's 4096-element chunk.
    tv_layout = cute.make_layout((_SWZ_THREADS, _SWZ_VPT), stride=(_SWZ_VPT, 1))
    tiler = (_SWZ_CHUNK,)
    gX = cute.zipped_divide(mX, tiler)
    gY = cute.zipped_divide(mY, tiler)
    cX = cute.zipped_divide(cute.make_identity_tensor(mX.shape), tiler)
    _mxfp8_floor_swizzle_kernel(gX, gY, sflat, cX, tv_layout, N, ncb).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[_SWZ_THREADS, 1, 1])


def mxfp8_floor_swizzle_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert N % 32 == 0 and (M * N) % _SWZ_CHUNK == 0, \
        f"mxfp8_floor_swizzle cute kernel needs N%32==0 and numel%{_SWZ_CHUNK}==0"
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    ngc = N // 32
    nrb = (M + 127) // 128
    ncb = (ngc + 3) // 4
    s_u8 = torch.zeros(nrb, ncb, 32, 16, dtype=torch.uint8, device=x.device)  # padding stays 0
    # flatten (elementwise-style); assumed_align=16 enables the 128-bit vectorized copies.
    mX = from_dlpack(x.reshape(-1), assumed_align=16).mark_layout_dynamic()
    mY = from_dlpack(y.reshape(-1), assumed_align=16).mark_layout_dynamic()
    sflat = from_dlpack(s_u8.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("mxfp8_floor_swizzle", M, N), _mxfp8_floor_swizzle_jit, mX, mY, sflat, N, ncb)
    fn(mX, mY, sflat)
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_FLOOR_SWIZZLE = QuantCastCuteRecipe.from_gold(
    Mxfp8FloorSwizzleGold, cute_fn=mxfp8_floor_swizzle_cute
)


# ---------------------------------------------------------------------------
# deepseek fp8 128x128: one fp32 scale per 128x128 block (amax over the whole block). The block is a
# strided 2D region (128 rows N apart), so the old 32-thread/warp kernel with a (32,512) tv-layout had
# uncoalesced loads across the warp (~4.5% peak). Like `..._dim_m`, the fix is TMA -- but this cast is
# NON-transposing (y is (M,N) like x), so it's simpler: no register->smem transpose, both tiles
# row-major (128,128). One CTA per block:
#   - TMA G2S loads the (128,128) tile into smem (sIN);
#   - each thread reduces its VPT contiguous elements to a local amax (32-wide vector reduce), then a
#     warp-reduce + smem block-reduce yields the whole-block amax; scale = max(amax,1e-12)/448;
#   - each thread re-reads its chunk (cheap smem), quantizes (vectorized f32->fp8), writes to sOUT;
#   - TMA S2G stores sOUT into the (128,128) tile of the row-major output.
# The fp32 scale is written once per block by thread 0. Mirrors deepseek_128x128_f.
# ---------------------------------------------------------------------------
_D128_TILE = 128
_D128_THREADS = 128                                      # tuned on B200 @ 16384 (128 best: 256->45%, 64->56%)
_D128_VPT = (_D128_TILE * _D128_TILE) // _D128_THREADS   # elems/thread (strided, coalesced)
_D128_WARPS = _D128_THREADS // 32
_D128_CHUNKS = _D128_VPT // 32                            # 32-wide chunks per thread (vectorize)
_D128_BYTES = _D128_TILE * _D128_TILE * 2                # bf16 tile bytes for the TMA expect-tx


@cute.struct
class _Deepseek128Smem:
    bar: cute.struct.MemRange[cutlass.Int64, 1]
    red: cute.struct.MemRange[cutlass.Float32, _D128_WARPS]     # per-warp amax scratch
    sin: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _D128_TILE * _D128_TILE], 1024]
    sout: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, _D128_TILE * _D128_TILE], 1024]


@cute.kernel
def _deepseek_128x128_kernel(atom_in: cute.CopyAtom, ten_in: cute.Tensor, atom_out: cute.CopyAtom,
                             ten_out: cute.Tensor, scales: cute.Tensor, sil: cute.Layout,
                             sol: cute.Layout, ncb: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()   # bidx = m_tile, bidy = n_tile
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    st = smem.allocate(_Deepseek128Smem)
    bar = st.bar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    sIN = st.sin.get_tensor(sil)
    sOUT = st.sout.get_tensor(sol)
    red = st.red.get_tensor(cute.make_layout(_D128_WARPS))
    gIN = cute.local_tile(ten_in, (_D128_TILE, _D128_TILE), (None, None))
    gOUT = cute.local_tile(ten_out, (_D128_TILE, _D128_TILE), (None, None))
    tAsA, tAgA = cpasync.tma_partition(atom_in, 0, cute.make_layout(1),
                                       cute.group_modes(sIN, 0, 2), cute.group_modes(gIN, 0, 2))
    tOsO, tBgB = cpasync.tma_partition(atom_out, 0, cute.make_layout(1),
                                       cute.group_modes(sOUT, 0, 2), cute.group_modes(gOUT, 0, 2))
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, _D128_BYTES)
        cute.copy(atom_in, tAgA[(None, bidx, bidy)], tAsA, tma_bar_ptr=bar)
    cute.arch.mbarrier_wait(bar, 0)

    # Thread t owns the strided (coalesced) positions {t + i*THREADS}: at each step lanes 0..31 hit
    # consecutive smem addresses -> conflict-free. (Contiguous VPT-runs put all 32 lanes on one bank.)
    # pass 1: per-thread amax over its VPT elems (32-wide chunks), then block-reduce.
    local = cutlass.Float32(0.0)
    for c in cutlass.range_constexpr(_D128_CHUNKS):
        frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
        for r in cutlass.range_constexpr(32):
            fp = tidx + (c * 32 + r) * _D128_THREADS
            frg[r] = sIN[fp // 128, fp % 128].to(cutlass.Float32)
        v = frg.load()
        local = cutlass.max(local, cute.where(v < 0, -v, v).reduce(
            cute.ReductionOp.MAX, cutlass.Float32(0.0), 0))
    wmax = cute.arch.warp_reduction_max(local)
    if tidx % 32 == 0:
        red[warp] = wmax
    cute.arch.sync_threads()
    bmax = red[0]
    for w in cutlass.range_constexpr(1, _D128_WARPS):
        bmax = cutlass.max(bmax, red[w])
    scale = cutlass.max(bmax, cutlass.Float32(1e-12)) / 448.0
    inv = 1.0 / scale                                         # reciprocal-mul, not per-element divs
    if tidx == 0:
        scales[bidx * ncb + bidy] = scale.to(scales.element_type)
    # pass 2: re-read (cheap smem), quantize (vectorized f32->fp8), write to sOUT.
    for c in cutlass.range_constexpr(_D128_CHUNKS):
        frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
        for r in cutlass.range_constexpr(32):
            fp = tidx + (c * 32 + r) * _D128_THREADS
            frg[r] = sIN[fp // 128, fp % 128].to(cutlass.Float32)
        frgOut = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
        frgOut.store((frg.load() * inv).to(cutlass.Float8E4M3FN))
        for r in cutlass.range_constexpr(32):
            fp = tidx + (c * 32 + r) * _D128_THREADS
            sOUT[fp // 128, fp % 128] = frgOut[r]

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
    if warp == 0:
        cute.copy(atom_out, tOsO, tBgB[(None, bidx, bidy)])


@cute.jit
def _deepseek_128x128_jit(mIN, mOUT, scales, ncb: cutlass.Constexpr):
    sil = cute.make_layout((_D128_TILE, _D128_TILE), stride=(_D128_TILE, 1))
    sol = cute.make_layout((_D128_TILE, _D128_TILE), stride=(_D128_TILE, 1))
    atom_in, ten_in = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mIN, sil, (_D128_TILE, _D128_TILE))
    atom_out, ten_out = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mOUT, sol, (_D128_TILE, _D128_TILE))
    M2, N2 = mIN.shape
    grid = ((M2 + _D128_TILE - 1) // _D128_TILE, (N2 + _D128_TILE - 1) // _D128_TILE, 1)
    _deepseek_128x128_kernel(atom_in, ten_in, atom_out, ten_out, scales, sil, sol,
                             ncb).launch(grid=grid, block=(_D128_THREADS, 1, 1), cluster=(1, 1, 1))


def fp8_deepseek_128x128_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % _D128_TILE == 0 and N % _D128_TILE == 0, \
        f"deepseek_128x128 cute kernel needs M%{_D128_TILE}==0 and N%{_D128_TILE}==0"
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    ncb = N // 128
    s = torch.empty(M // 128, ncb, dtype=torch.float32, device=x.device)
    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-elem aligned).
    mIN = (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
           .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mOUT = (from_dlpack(y, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=16))
    scales = from_dlpack(s.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("deepseek_128x128", M, N), _deepseek_128x128_jit, mIN, mOUT, scales, ncb)
    fn(mIN, mOUT, scales)
    return y, s


FP8_DEEPSEEK_128X128 = QuantCastCuteRecipe.from_gold(
    Deepseek128x128Gold, cute_fn=fp8_deepseek_128x128_cute
)


# ---------------------------------------------------------------------------
# fp8 rowwise (full-span), optimized: one fp32 scale per row, amax over ALL columns. One CTA per row,
# but it LOOPS over the row in BN-wide blocks (small, few-warp CTA) instead of holding the whole row
# live -- the register-resident variant was capped at 50% occupancy by 58 regs/thr. Pass 1 streams the
# row's blocks accumulating a per-thread abs-max (only VPT elems live/iter), then a warp-reduce (row in
# one warp) or warp+smem block-reduce (wide row) gives the row amax = one fp32 scale; pass 2 re-reads
# each block (hits L2, warm from pass 1 -- like triton's evict_last/evict_first hints), quantizes and
# stores. Two DRAM->L2 read passes trade a small extra read for high occupancy. Mirrors rowwise_fp8_f.
# Needs N % BN == 0 with BN = THREADS*VPT, and THREADS <= 32 (one warp) or a multiple of 32.
# ---------------------------------------------------------------------------
_RW_VPT = 16                               # vals/thread (128-bit bf16 load, 128-bit fp8 store)
_RW_THREADS = 256                          # target CTA width (BN = THREADS*VPT = 4096), tuned on B200
_RW_LD_BITS = min(128, _RW_VPT * 16)       # bf16 in  -> 128-bit vectorized load
_RW_ST_BITS = min(128, _RW_VPT * 8)        # fp8 out  -> 128-bit vectorized store
_RW_MAXWARPS = 32                          # smem scratch upper bound


@cute.struct
class _RowwiseSmem:
    red: cute.struct.MemRange[cutlass.Float32, _RW_MAXWARPS]   # per-warp amax scratch


@cute.kernel
def _rowwise_opt_kernel(gX: cute.Tensor, gY: cute.Tensor, sflat: cute.Tensor, tv_layout: cute.Layout,
                        threads: cutlass.Constexpr, warps: cutlass.Constexpr, nbpr: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()                       # bidx = row
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    ld_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type,
                                  num_bits_per_copy=_RW_LD_BITS)
    st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type,
                                  num_bits_per_copy=_RW_ST_BITS)
    # pass 1: stream the row's blocks, accumulating a per-thread abs-max (only VPT elems live per iter)
    local = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(nbpr):
        thrX = cute.composition(gX[(None, bidx * nbpr + j)], tv_layout)[(tidx, None)]
        frg = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
        cute.copy(ld_atom, thrX, frg)
        v = frg.load().to(cutlass.Float32)
        local = cutlass.max(local, cutlass.max(v, -v).reduce(
            cute.ReductionOp.MAX, cutlass.Float32(0.0), 0))
    if cutlass.const_expr(warps == 1):
        # row fits one (possibly partial) warp: single butterfly reduce broadcasts amax to all threads
        amax = cutlass.max(cute.arch.warp_reduction_max(local, threads_in_group=threads),
                           cutlass.Float32(1e-12))
    else:
        # wide row: warp-reduce, lane0 -> smem scratch, then every thread maxes the WARPS entries
        smem = utils.SmemAllocator()
        st = smem.allocate(_RowwiseSmem)
        red = st.red.get_tensor(cute.make_layout(warps))
        wmax = cute.arch.warp_reduction_max(local)
        if tidx % 32 == 0:
            red[warp] = wmax
        cute.arch.sync_threads()
        bmax = red[0]
        for w in cutlass.range_constexpr(1, warps):
            bmax = cutlass.max(bmax, red[w])
        amax = cutlass.max(bmax, cutlass.Float32(1e-12))
    scale = amax / 448.0
    inv = 1.0 / scale                                        # reciprocal-mul, not per-element divs
    # pass 2: re-read each block (warm in L2), quantize (vectorized f32->fp8), store.
    for j in cutlass.range_constexpr(nbpr):
        thrX = cute.composition(gX[(None, bidx * nbpr + j)], tv_layout)[(tidx, None)]
        thrY = cute.composition(gY[(None, bidx * nbpr + j)], tv_layout)[(tidx, None)]
        frg = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
        cute.copy(ld_atom, thrX, frg)
        frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
        frgY.store((frg.load().to(cutlass.Float32) * inv).to(gY.element_type))
        cute.copy(st_atom, frgY, thrY)
    if tidx == 0:
        sflat[bidx] = scale.to(sflat.element_type)


@cute.jit
def _rowwise_opt_jit(mX, mY, sflat, threads: cutlass.Constexpr, warps: cutlass.Constexpr,
                     nbpr: cutlass.Constexpr):
    tv_layout = cute.make_layout((threads, _RW_VPT), stride=(_RW_VPT, 1))
    tiler = (threads * _RW_VPT,)                              # one BN-wide block per tile; nbpr per row
    gX = cute.zipped_divide(mX, tiler)
    gY = cute.zipped_divide(mY, tiler)
    _rowwise_opt_kernel(gX, gY, sflat, tv_layout, threads, warps, nbpr).launch(
        grid=[cute.size(gX, mode=[1]) // nbpr, 1, 1], block=[threads, 1, 1])


def fp8_rowwise_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert N % _RW_VPT == 0, f"fp8_rowwise cute kernel needs N % {_RW_VPT} == 0"
    # BN-wide blocks (BN = THREADS*VPT); the CTA loops nbpr = N//BN blocks over its row. Use the target
    # CTA width when it divides N, else fall back to one block spanning the whole (short) row.
    bn = _RW_THREADS * _RW_VPT
    if N % bn == 0:
        threads, nbpr = _RW_THREADS, N // bn
    else:
        threads, nbpr = N // _RW_VPT, 1
    assert threads <= 1024 and (threads <= 32 or threads % 32 == 0), \
        f"fp8_rowwise cute kernel needs a valid block width for N={N} (got threads={threads})"
    warps = (threads + 31) // 32
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(M, 1, dtype=torch.float32, device=x.device)
    # flatten to 1-D (blocks are contiguous BN-runs); assumed_align=16 enables the vectorized copies
    # (torch allocations are >=256B aligned).
    mX = from_dlpack(x.reshape(-1), assumed_align=16).mark_layout_dynamic()
    mY = from_dlpack(y.reshape(-1), assumed_align=16).mark_layout_dynamic()
    sflat = from_dlpack(s.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("fp8_rowwise", M, N), _rowwise_opt_jit, mX, mY, sflat, threads, warps, nbpr)
    fn(mX, mY, sflat)
    return y, s


FP8_ROWWISE = QuantCastCuteRecipe.from_gold(RowwiseFp8Gold, cute_fn=fp8_rowwise_cute)


# ---------------------------------------------------------------------------
# fp8 colwise (full-span): one fp32 scale per column, amax over ALL rows; transposed outputs
# (N, M) / (N, 1). The reduction is down a column (the strided axis of row-major x) and the output is
# transposed, so a single naive kernel is forced into uncoalesced reads (~1.6%). Mirror triton's
# split into two coalesced passes (see quant_cast_triton):
#   pass 1 (amax): TMA-load (TM, TN) row-major tiles of x (the TMA engine streams these strided tiles
#     at DRAM speed -- a hand-rolled strided row-segment read caps at ~42%); each thread owns a column,
#     reduces its TM rows in smem to a partial abs-max, then one atomic_max per column into a (N,) fp32
#     scratch -- combines the per-column amax across the M-grid.
#   pass 2 (quant): TMA-load a (TM, TN) row-major tile of x, quantize each column with the precomputed
#     scale (amax/448), transpose in the register->smem write (like mxfp8_floor_dim_m -- the DSL can't
#     drive a col-major TMA store), and TMA-store the (TN, TM) tile into the row-major (N, M) output;
#     the (N, 1) scale is written once (from the m_tile=0 row of blocks). Both DRAM passes ride the TMA
#     engine and the transpose happens in the register->smem write.
# ~1.6% -> 46.0% of B200 peak (beats triton 43.8%, compile 25.6%). The ceiling is ~51%: a full-column
# amax forces reading x TWICE (amax then quant) and, unlike rowwise, the quant re-read MISSES L2 (the
# whole 512 MB of x streams between the two kernels; a full column is 32 KB * M, far larger than L2, so
# nothing stays warm). L2-panel tiling (amax+quant per column-panel that fits in L2) does NOT help --
# separate TMA kernels don't retain the panel, and per-CTA reuse needs <6 concurrent CTAs (occupancy).
# ---------------------------------------------------------------------------
_CWA_TM, _CWA_TN, _CWA_WARPS = 128, 256, 8     # amax tile (tuned on B200; needs M%TM==0, N%TN==0)
_CWA_THREADS = _CWA_WARPS * 32
_CWA_ITERS = (_CWA_TN + _CWA_THREADS - 1) // _CWA_THREADS
_CWA_IN_BYTES = _CWA_TM * _CWA_TN * 2          # bf16 tile bytes for the TMA expect-tx


@cute.struct
class _ColwiseAmaxSmem:
    bar: cute.struct.MemRange[cutlass.Int64, 1]
    sin: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _CWA_TM * _CWA_TN], 1024]


@cute.kernel
def _colwise_amax_kernel(atom_in: cute.CopyAtom, ten_in: cute.Tensor, mA: cute.Tensor,
                         sil: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()   # bidx = m_tile, bidy = n_tile
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    st = smem.allocate(_ColwiseAmaxSmem)
    bar = st.bar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    sIN = st.sin.get_tensor(sil)                             # (TM, TN) row-major
    gIN = cute.local_tile(ten_in, (_CWA_TM, _CWA_TN), (None, None))
    tAsA, tAgA = cpasync.tma_partition(atom_in, 0, cute.make_layout(1),
                                       cute.group_modes(sIN, 0, 2), cute.group_modes(gIN, 0, 2))
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, _CWA_IN_BYTES)
        cute.copy(atom_in, tAgA[(None, bidx, bidy)], tAsA, tma_bar_ptr=bar)
    cute.arch.mbarrier_wait(bar, 0)

    n0 = bidy * _CWA_TN
    for it in cutlass.range_constexpr(_CWA_ITERS):
        col = tidx + it * _CWA_THREADS
        if col < _CWA_TN:
            frg = cute.make_rmem_tensor(cute.make_layout(_CWA_TM), cutlass.Float32)
            for r in cutlass.range_constexpr(_CWA_TM):
                frg[r] = sIN[r, col].to(cutlass.Float32)     # column read (warp-coalesced)
            v = frg.load()
            amax = cutlass.Float32(
                cute.where(v < 0, -v, v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0))
            cute.arch.atomic_max_float32(ptr=(mA.iterator + (n0 + col)).llvm_ptr, value=amax)


@cute.jit
def _colwise_amax_jit(mX, mA):
    sil = cute.make_layout((_CWA_TM, _CWA_TN), stride=(_CWA_TN, 1))
    atom_in, ten_in = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mX, sil, (_CWA_TM, _CWA_TN))
    M2, N2 = mX.shape
    grid = ((M2 + _CWA_TM - 1) // _CWA_TM, (N2 + _CWA_TN - 1) // _CWA_TN, 1)
    _colwise_amax_kernel(atom_in, ten_in, mA, sil).launch(
        grid=grid, block=(_CWA_THREADS, 1, 1), cluster=(1, 1, 1))


_CWQ_TM, _CWQ_TN, _CWQ_WARPS = 64, 256, 8      # tuned on B200 (needs M%TM==0, N%TN==0)
_CWQ_THREADS = _CWQ_WARPS * 32
_CWQ_ITERS = (_CWQ_TN + _CWQ_THREADS - 1) // _CWQ_THREADS
_CWQ_IN_BYTES = _CWQ_TM * _CWQ_TN * 2          # bf16 tile bytes for the TMA expect-tx


@cute.struct
class _ColwiseQuantSmem:
    bar: cute.struct.MemRange[cutlass.Int64, 1]
    sin: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _CWQ_TM * _CWQ_TN], 1024]
    sout: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, _CWQ_TM * _CWQ_TN], 1024]


@cute.kernel
def _colwise_quant_kernel(atom_in: cute.CopyAtom, ten_in: cute.Tensor, atom_out: cute.CopyAtom,
                          ten_out: cute.Tensor, mA: cute.Tensor, sflat: cute.Tensor,
                          sil: cute.Layout, sol: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()   # bidx = m_tile, bidy = n_tile
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    st = smem.allocate(_ColwiseQuantSmem)
    bar = st.bar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(bar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    sIN = st.sin.get_tensor(sil)                             # (TM, TN) row-major
    sOUT = st.sout.get_tensor(sol)                           # (TN, TM) row-major (transposed)
    gIN = cute.local_tile(ten_in, (_CWQ_TM, _CWQ_TN), (None, None))
    gOUT = cute.local_tile(ten_out, (_CWQ_TN, _CWQ_TM), (None, None))
    tAsA, tAgA = cpasync.tma_partition(atom_in, 0, cute.make_layout(1),
                                       cute.group_modes(sIN, 0, 2), cute.group_modes(gIN, 0, 2))
    tOsO, tBgB = cpasync.tma_partition(atom_out, 0, cute.make_layout(1),
                                       cute.group_modes(sOUT, 0, 2), cute.group_modes(gOUT, 0, 2))
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, _CWQ_IN_BYTES)
        cute.copy(atom_in, tAgA[(None, bidx, bidy)], tAsA, tma_bar_ptr=bar)
    cute.arch.mbarrier_wait(bar, 0)

    n0 = bidy * _CWQ_TN
    for it in cutlass.range_constexpr(_CWQ_ITERS):
        col = tidx + it * _CWQ_THREADS
        if col < _CWQ_TN:
            amax = cutlass.max(mA[n0 + col], cutlass.Float32(1e-12))
            scale = amax / 448.0
            inv = 1.0 / scale                                # reciprocal-mul, not TM divs
            frgIn = cute.make_rmem_tensor(cute.make_layout(_CWQ_TM), cutlass.Float32)
            for r in cutlass.range_constexpr(_CWQ_TM):
                frgIn[r] = sIN[r, col].to(cutlass.Float32)   # column read (warp-coalesced)
            frgOut = cute.make_rmem_tensor(cute.make_layout(_CWQ_TM), cutlass.Float8E4M3FN)
            frgOut.store((frgIn.load() * inv).to(cutlass.Float8E4M3FN))
            for r in cutlass.range_constexpr(_CWQ_TM):
                sOUT[col, r] = frgOut[r]                      # contiguous run = the transpose
            if bidx == 0:
                sflat[n0 + col] = scale.to(sflat.element_type)

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
    if warp == 0:
        cute.copy(atom_out, tOsO, tBgB[(None, bidy, bidx)])   # y tile (n_tile, m_tile)


@cute.jit
def _colwise_quant_jit(mX, mY, mA, sflat):
    sil = cute.make_layout((_CWQ_TM, _CWQ_TN), stride=(_CWQ_TN, 1))
    sol = cute.make_layout((_CWQ_TN, _CWQ_TM), stride=(_CWQ_TM, 1))
    atom_in, ten_in = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mX, sil, (_CWQ_TM, _CWQ_TN))
    atom_out, ten_out = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mY, sol, (_CWQ_TN, _CWQ_TM))
    M2, N2 = mX.shape
    grid = ((M2 + _CWQ_TM - 1) // _CWQ_TM, (N2 + _CWQ_TN - 1) // _CWQ_TN, 1)
    _colwise_quant_kernel(atom_in, ten_in, atom_out, ten_out, mA, sflat, sil, sol).launch(
        grid=grid, block=(_CWQ_THREADS, 1, 1), cluster=(1, 1, 1))


def fp8_colwise_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % _CWA_TM == 0 and M % _CWQ_TM == 0 and N % _CWA_TN == 0 and N % _CWQ_TN == 0, \
        f"fp8_colwise cute kernel needs M%{_CWQ_TM}==0 and N%{_CWQ_TN}==0"
    y = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)  # transposed row-major output
    s = torch.empty(N, 1, dtype=torch.float32, device=x.device)
    a = torch.zeros(N, dtype=torch.float32, device=x.device)           # per-column amax scratch (>=0)
    mA = from_dlpack(a).mark_layout_dynamic()
    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-elem aligned).
    mX = (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mY = (from_dlpack(y, assumed_align=16).mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, divisibility=16))
    sflat = from_dlpack(s.reshape(-1)).mark_layout_dynamic()
    # --- pass 1: TMA-tile per-column amax via atomic_max ---
    amax_fn = _compiled(("fp8_colwise_amax", M, N), _colwise_amax_jit, mX, mA)
    amax_fn(mX, mA)
    # --- pass 2: TMA-tile quant + transposed store (reads the precomputed per-column amax) ---
    quant_fn = _compiled(("fp8_colwise_quant", M, N), _colwise_quant_jit, mX, mY, mA, sflat)
    quant_fn(mX, mY, mA, sflat)
    return y, s


FP8_COLWISE = QuantCastCuteRecipe.from_gold(ColwiseFp8Gold, cute_fn=fp8_colwise_cute)


# ---------------------------------------------------------------------------
# nvfp4 shared device helper: given a 1x16 block fragment `v` (f32) and its `outer` scale, compute
# the inner e4m3 block scale + the packed fp4 qdata (8 bytes). Mirrors nvfp4_gs_swizzle_f's per-block
# math: inner = clamp((amax/6)/outer, eps, 448)->e4m3; data = v/(outer*inner) -> fp4 (RN, saturating,
# so the explicit +-6 clamp is redundant). Returns (packed_bytes_ssa, inner_e4_scalar). Scalar->e4m3
# and e4m3->f32 both need a >=32-bit-aligned vector, so a 4-lane broadcast fragment is used.
_NVFP4_EPS = 0.015625  # torch.finfo(float8_e4m3fn).tiny (E4M3_EPS)


@cute.jit
def _nvfp4_block(v, data_val_layout, outer):
    amax = cutlass.max(v, -v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    inner_val = cutlass.min(
        cutlass.max((amax / 6.0) / outer, cutlass.Float32(_NVFP4_EPS)), cutlass.Float32(448.0)
    )
    frgI = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    frgI[0], frgI[1], frgI[2], frgI[3] = inner_val, inner_val, inner_val, inner_val
    frgIe = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float8E4M3FN)
    frgIe.store(frgI.load().to(cutlass.Float8E4M3FN))  # RN to e4m3 (vector conversion)
    frgIf = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    frgIf.store(frgIe.load().to(cutlass.Float32))  # e4m3 -> f32 (vector conversion)
    recip = (1.0 / outer) / frgIf[0]
    frgD = cute.make_rmem_tensor(data_val_layout, cutlass.Float32)  # 16 scaled f32 values
    frgD.store(v * recip)
    packed = _maybe_recast_from_f4(frgD.load().to(cutlass.Float4E2M1FN), cutlass.Float4E2M1FN)
    return packed, frgIe[0]


# Hardware fp4 / e4m3 conversions via inline PTX (Blackwell SM10.x), ported verbatim from the
# human-optimized torchao unified fp4 CuTeDSL cast (pytorch/ao#4517, cute_utils.py). The fast
# nvfp4_swizzle kernel uses these single hardware instructions instead of `_nvfp4_block`'s
# 4-lane-broadcast e4m3 fragment conversions + `_maybe_recast_from_f4` fp4 packing -- far fewer
# registers and instructions (the kernel was compute/register-bound at 56 reg/thread, 44% occ).
@dsl_user_op
def _cvt_rn_satfinite_e2m1x2_f32(hi, lo, *, loc=None, ip=None):
    """Pack two f32 into one E2M1x2 byte = (e2m1(hi) << 4) | e2m1(lo) (RN, saturating to +-6). The
    e2m1x2 cvt output is a .b8, but inline-asm outputs must be >=16-bit, so (like cutlass's
    nvvm_wrappers) we cvt into a .reg .b8 and assemble a .b16 via mov.b16 with a zero high byte."""
    packed = cutlass.Uint16(
        llvm.inline_asm(
            T.i16(),
            [cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
             cutlass.Float32(lo).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b8 d, z, w;\n\t"
            ".reg .b16 zero16;\n\t"
            "mov.u16 zero16, 0;\n\t"
            "mov.b16 {z, w}, zero16;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 d, $1, $2;\n\t"
            "mov.b16 $0, {d, z};\n\t"
            "}",
            "=h,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return cutlass.Uint8(packed & cutlass.Uint16(0xFF))


@dsl_user_op
def _cvt_rn_satfinite_e2m1x2_f32_x4(v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None):
    """Pack eight f32 into four E2M1x2 bytes as one Uint32 (byte k = (e2m1(v[2k+1])<<4)|e2m1(v[2k])),
    assembled with a single mov.b32 -- no per-byte masking. Mirrors MSLK's convert_fp32_to_fp4_packed
    (the triton nvfp4 kernel); amortizes the fp4-pack ALU over 4 bytes/call instead of 1."""
    args = [cutlass.Float32(x).ir_value(loc=loc, ip=ip)
            for x in (v0, v1, v2, v3, v4, v5, v6, v7)]
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            args,
            "{\n\t"
            ".reg .b8 b0, b1, b2, b3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;\n\t"
            "mov.b32 $0, {b0, b1, b2, b3};\n\t"
            "}",
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cvt_rn_satfinite_e4m3x2_f32(hi, lo, *, loc=None, ip=None):
    """Convert one f32 to its E4M3 byte (RN, saturating to 448): pass hi == lo, return the low byte
    (e4m3x2 cvt already yields a .b16, low byte = e4m3(lo))."""
    packed = cutlass.Uint16(
        llvm.inline_asm(
            T.i16(),
            [cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
             cutlass.Float32(lo).ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
            "=h,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return cutlass.Uint8(packed & cutlass.Uint16(0xFF))


@dsl_user_op
def _cvt_e4m3_byte_to_f32(byte, *, loc=None, ip=None):
    """Dequantize a single E4M3 byte to Float32 (== float8_e4m3fn(byte).to(torch.float32)). Zero-
    extend to a .b16 and feed it straight to cvt.rn.f16x2.e4m3x2; the low f16 = e4m3(byte)."""
    src_i16 = cutlass.Uint16(byte)
    rst_i32 = cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [src_i16.ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.e4m3x2 $0, $1;",
            "=r,h",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    lo_f16_bits = cutlass.Uint16(rst_i32 & cutlass.Uint32(0xFFFF))
    f16_val = cutlass.Float16(llvm.bitcast(T.f16(), lo_f16_bits.ir_value(loc=loc, ip=ip)))
    return cutlass.Float32(f16_val)


@cute.jit
def _nvfp4_scale_e4m3(amax, inv_outer):
    """NVFP4 two-level inner scale via hardware cvts: returns (e4m3 scale byte, fp32 data recip).
    Matches nvfp4_gs_swizzle_f: local = clamp((amax/6)/outer, eps, 448) -> e4m3;
    recip = (1/outer) / dequant(e4m3) (what each element is multiplied by before the fp4 cast).
    Takes the hoisted inv_outer = 1/outer so the per-block `/outer` is a multiply (nvfp4 is a
    coarse 4-bit / SQNR-graded cast, so the reassociated rounding is well within tolerance)."""
    local = cutlass.min(
        cutlass.max((amax / 6.0) * inv_outer, cutlass.Float32(_NVFP4_EPS)), cutlass.Float32(448.0)
    )
    e4m3_byte = _cvt_rn_satfinite_e4m3x2_f32(local, local)
    recip = inv_outer / _cvt_e4m3_byte_to_f32(e4m3_byte)
    return e4m3_byte, recip


# nvfp4 with a per-tensor (global) outer scale: 1x16 inner blocks, e4m3 inner scale, fp4-packed
# qdata, inner scale in the swizzled 4D grid. Mirrors nvfp4_gs_swizzle_f. Went 7.9% -> ~62% of B200
# peak (beats compile 23.5%, edges triton 62.7% at peak).
#
# ATTRIBUTION: the design and much of the machinery are adapted from the human-optimized torchao
# unified FP4 (NVFP4 + MXFP4 +/- RHT) CuTeDSL quantize cast, pytorch/ao PR #4517
# (https://github.com/pytorch/ao/pull/4517), files cutedsl/cute_utils.py + fp4_unified_quantize.py.
# Taken from it: the three inline-PTX conversion helpers (e2m1x2 / e4m3x2 / f16x2.e4m3x2 -- see
# _cvt_* below, ported ~verbatim), the two-level nvfp4 scale recipe, the "group of 32 = two blocks =
# one 128-bit store" unit of work, and the warp-per-row ("wpr") thread mapping. Differences: this is
# a single-purpose nvfp4-only kernel (no MXFP4, no fused RHT, one scale layout = the blocked swizzle
# our gold recipe needs) rather than that PR's unified multi-format/multi-layout kernel; the batched
# 8-f32 -> 4-byte packer (_cvt_rn_satfinite_e2m1x2_f32_x4) instead follows the PR's sibling *triton*
# kernel's pack=4 asm; and the swizzle-offset hoisting (fix 2 below) is not in the PR.
#
# Like mxfp8_floor_swizzle this is a non-transposing streaming cast (qdata is (M, N//2), same row
# order as x), so it uses the SAME DRAM-speed recipe: FLATTEN to 1-D, 128 threads/CTA, each thread
# owning CONTIGUOUS runs loaded/stored via 128-bit VECTORIZED copy atoms. The old kernel launched
# only 8 threads/CTA with scalar (per-element) CopyUniversalOp loads (~7.9%). The unit of work is a
# "group" of 32 input elements = TWO 1x16 nvfp4 blocks = exactly one 128-bit fp4 store (32 fp4 -> 16
# bytes). Per block: thread-local amax reduce -> e4m3 inner scale -> fp4 pack; the 16 output bytes
# are assembled into one 128-bit store. The two e4m3 scale bytes are scattered to their swizzled 4D
# positions (numel/16 bytes ~= 0.4%). assumed_align=16 enables the 128-bit copies.
#
# The bandwidth is unlocked by three profiler-guided fixes (the kernel is ALU-pipe bound, not DRAM
# bound -- ncu showed ALU ~73% while DRAM climbed 58 -> 66% as each was applied):
#  1. HARDWARE cvts (inline PTX, ported from ao#4517): cvt.rn.satfinite.e2m1x2.f32 packs 8 f32 ->
#     4 fp4 bytes/call (one mov.b32, no per-byte masking), and cvt e4m3x2 / f16x2.e4m3x2 do the
#     two-level scale as single instructions -- vs _nvfp4_block's 4-lane-broadcast e4m3 fragments
#     + _maybe_recast_from_f4 (which cost registers + ALU). (blocked_outer still uses _nvfp4_block.)
#  2. HOISTED swizzle offset: the 4D (nrb,ncb,32,16) flatten factors as row_base + (col//4)*512 +
#     (col%4); the per-row div/mods are computed once/group (not per block), and because a group's
#     start is a multiple of 32 -> col16 is even -> its NB blocks share bc and land at flat0, +1.
#     (Recomputing the full swizzle per block cost ~7 pts: it was the dominant ALU term.)
#  3. WARP-PER-ROW mapping (ao#4517's "wpr", which beat its own 1-D "striped" mapping on this
#     swizzle layout): warp w owns row bidy*WARPS+w; its 32 lanes + a grid.x column split + ILP
#     stripe that row's 32-elem groups. Because the row is FIXED per warp, the row-dependent swizzle
#     math (row_base) is computed ONCE and reused across all ~GPR/(XSPLIT*32) groups a lane visits
#     (vs 2 in a 1-D-flatten mapping), and a whole row's scale bytes land in one 128-row swizzle
#     atom. All ILP loads are issued first so global-load latency overlaps via MLP. This took the
#     1-D striped mapping (~58%, long-scoreboard-bound at 44% occ) to ~62% (peaks at ao#4517's own
#     wpr ~63%, above the repo's triton 62.7%). Tuned WARPS=2, XSPLIT=4, ILP=4 (the 128-bit store
#     still pins the group at 32 elems = 2 blocks).
# Requires N % 32 == 0 (whole 32-groups per row); a 32-group gc holds the two consecutive 16-groups
# 2*gc and 2*gc+1 (2*gc even), so their swizzle offsets are flat0 and flat0+1.
_NVSWZ_WARPS = 2                             # warps/CTA; warp w owns row bidy*WARPS+w
_NVSWZ_XSPLIT = 4                            # grid.x column split within a row
_NVSWZ_ILP = 4                               # groups/thread per loop iter (loads issued together)
_NVSWZ_THREADS = 32 * _NVSWZ_WARPS
_NVSWZ_LDWIDTH = 8                           # bf16 per 128-bit load (NLD = 32//LDWIDTH sub-loads)
_NVSWZ_LD_BITS = _NVSWZ_LDWIDTH * 16         # bf16 input  -> 128-bit vectorized load
_NVSWZ_ST_BITS = 128                         # 16 uint8 output -> 128-bit vectorized store


@cute.kernel
def _nvfp4_swizzle_kernel(gX: cute.Tensor, gQ: cute.Tensor, sflat: cute.Tensor, mOuter: cute.Tensor,
                          M: cutlass.Constexpr, N: cutlass.Constexpr, ncb: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_id = tidx // 32
    lane = tidx % 32
    ld_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type,
                                  num_bits_per_copy=_NVSWZ_LD_BITS)
    st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gQ.element_type,
                                  num_bits_per_copy=_NVSWZ_ST_BITS)
    frgO = cute.make_rmem_tensor(cute.make_layout(1), mOuter.element_type)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mOuter.element_type), mOuter, frgO)
    inv_outer = 1.0 / frgO[0]                    # loop-invariant; per-block `/outer` becomes a mul
    blk_layout = cute.make_layout(16)
    ld_layout = cute.make_layout(_NVSWZ_LDWIDTH)
    GPR = N // 32                                # 32-elem groups per row (constexpr)
    NLD = 32 // _NVSWZ_LDWIDTH                   # 128-bit sub-loads per group

    row = bidy * _NVSWZ_WARPS + warp_id          # warp owns one row
    if row < M:
        # Row-invariant swizzle offset (bc=0, c4=0) -- computed ONCE per warp-row and reused across
        # every group this lane visits: the 4D (nrb,ncb,32,16) flatten factors as
        # row_base + (col//4)*512 + (col%4); this per-row div/mod chain was the ALU-pipe bottleneck.
        r128 = row % 128
        row_base = ((row // 128) * ncb * 32 + (r128 % 32)) * 16 + (r128 // 32) * 4
        nthreads_row = _NVSWZ_XSPLIT * 32
        base = bidx * 32 + lane
        while base < GPR:
            # issue all ILP loads first (each group = one contiguous 32-run -> NLD 128-bit loads;
            # the ILP groups are independent -> overlap) so global-load latency is hidden via MLP.
            fragbuf = cute.make_rmem_tensor(cute.make_layout(_NVSWZ_ILP * 32), gX.element_type)
            for jj in cutlass.range_constexpr(_NVSWZ_ILP):
                gc = base + jj * nthreads_row
                if gc < GPR:
                    off = cute.assume(row * N + gc * 32, divby=32)
                    for w in cutlass.range_constexpr(NLD):
                        cute.copy(
                            ld_atom,
                            cute.make_tensor(gX.iterator + off + w * _NVSWZ_LDWIDTH, ld_layout),
                            cute.make_tensor(fragbuf.iterator + jj * 32 + w * _NVSWZ_LDWIDTH,
                                             ld_layout))
            for jj in cutlass.range_constexpr(_NVSWZ_ILP):
                gc = base + jj * nthreads_row
                if gc < GPR:
                    frgQ = cute.make_rmem_tensor(cute.make_layout(16), gQ.element_type)
                    frgQ32 = cute.recast_tensor(frgQ, dtype=cutlass.Uint32)
                    col16 = gc * 2               # two 16-groups per 32-group; first (even) at 2*gc
                    flat0 = row_base + (col16 // 4) * 512 + (col16 % 4)
                    for b in cutlass.range_constexpr(2):
                        blk = cute.make_rmem_tensor(blk_layout, cutlass.Float32)
                        blk.store(cute.make_tensor(fragbuf.iterator + jj * 32 + b * 16,
                                                   blk_layout).load().to(cutlass.Float32))
                        v = blk.load()
                        amax = cutlass.max(v, -v).reduce(
                            cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
                        e4m3_byte, recip = _nvfp4_scale_e4m3(amax, inv_outer)
                        # pack 16 scaled values -> 8 bytes = two uint32 (even col -> low nibble, odd
                        # -> high, matching gold _f32_to_packed_fp4); 8 values/call amortizes ALU.
                        for c in cutlass.range_constexpr(2):
                            o = c * 8
                            frgQ32[b * 2 + c] = _cvt_rn_satfinite_e2m1x2_f32_x4(
                                blk[o] * recip, blk[o + 1] * recip, blk[o + 2] * recip,
                                blk[o + 3] * recip, blk[o + 4] * recip, blk[o + 5] * recip,
                                blk[o + 6] * recip, blk[o + 7] * recip)
                        sflat[flat0 + b] = e4m3_byte
                    offq = cute.assume(row * (N // 2) + gc * 16, divby=16)
                    cute.copy(st_atom, frgQ,
                              cute.make_tensor(gQ.iterator + offq, cute.make_layout(16)))
            base = base + nthreads_row * _NVSWZ_ILP


@cute.jit
def _nvfp4_swizzle_jit(mX, mQ, sflat, mOuter,
                       M: cutlass.Constexpr, N: cutlass.Constexpr, ncb: cutlass.Constexpr):
    # warp-per-row grid: grid.y covers M in steps of WARPS, grid.x is the XSPLIT column split.
    grid_y = (M + _NVSWZ_WARPS - 1) // _NVSWZ_WARPS
    _nvfp4_swizzle_kernel(mX, mQ, sflat, mOuter, M, N, ncb).launch(
        grid=[_NVSWZ_XSPLIT, grid_y, 1], block=[_NVSWZ_THREADS, 1, 1])


def nvfp4_swizzle_cute(x, outer_scale, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert N % 32 == 0, "nvfp4_swizzle cute kernel needs N % 32 == 0 (whole 32-groups per row)"
    q = torch.empty(M, N // 2, dtype=torch.uint8, device=x.device)
    ngc = N // 16
    nrb = (M + 127) // 128
    ncb = (ngc + 3) // 4
    s_u8 = torch.zeros(nrb, ncb, 32, 16, dtype=torch.uint8, device=x.device)  # padding stays 0
    # flatten (elementwise-style); assumed_align=16 enables the 128-bit vectorized copies.
    mX = from_dlpack(x.reshape(-1), assumed_align=16).mark_layout_dynamic()
    mQ = from_dlpack(q.reshape(-1), assumed_align=16).mark_layout_dynamic()
    sflat = from_dlpack(s_u8.reshape(-1)).mark_layout_dynamic()
    mOuter = from_dlpack(outer_scale.reshape(1))  # per-tensor scalar (static)
    fn = _compiled(("nvfp4_swizzle", M, N), _nvfp4_swizzle_jit, mX, mQ, sflat, mOuter, M, N, ncb)
    fn(mX, mQ, sflat, mOuter)
    return q.view(torch.float4_e2m1fn_x2), s_u8.view(torch.float8_e4m3fn)


NVFP4_SWIZZLE = QuantCastCuteRecipe.from_gold(Nvfp4GsSwizzleGold, cute_fn=nvfp4_swizzle_cute)


# nvfp4 with a 128x128-blocked outer scale (Mb, Nb): the outer scale is looked up per block from
# outer_blocked[row//128, (col*16)//128]. Mirrors nvfp4_blocked_outer_f. (Still the naive one-block-
# per-thread kernel -- unlike nvfp4_swizzle it wasn't the optimization target here.)
_NVFP4_BPT = 8  # 1x16 blocks per CTA (blocked_outer naive kernel)


@cute.kernel
def _nvfp4_blocked_kernel(gX: cute.Tensor, gQ: cute.Tensor, sflat: cute.Tensor, mOuter: cute.Tensor,
                          cX: cute.Tensor, tv_layout: cute.Layout, q_tv_layout: cute.Layout,
                          ncb: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    thrX = cute.composition(gX[(None, bidx)], tv_layout)[(tidx, None)]
    thrQ = cute.composition(gQ[(None, bidx)], q_tv_layout)[(tidx, None)]
    thrC = cute.composition(cX[(None, bidx)], tv_layout)[(tidx, None)]
    frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type), thrX, frgX)
    row = thrC[0][0]
    col = thrC[0][1] // 16  # 16-group index
    outer = mOuter[(row // 128, (col * 16) // 128)]  # per-128x128-block outer scale

    packed, inner_e4 = _nvfp4_block(frgX.load().to(cutlass.Float32),
                                    cute.get(tv_layout, mode=[1]), outer)
    frgQ = cute.make_rmem_tensor(cute.get(q_tv_layout, mode=[1]), gQ.element_type)
    frgQ.store(packed)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gQ.element_type), frgQ, thrQ)
    flat = _swizzle_flat(row, col, ncb)
    fib = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float8E4M3FN)
    fib[0] = inner_e4
    sflat[flat] = cute.recast_tensor(fib, dtype=cutlass.Uint8)[0]


@cute.jit
def _nvfp4_blocked_jit(mX, mQ, sflat, mOuter, ncb: cutlass.Constexpr):
    tv_layout = cute.make_layout((_NVFP4_BPT, 16), stride=(16, 1))
    q_tv_layout = cute.make_layout((_NVFP4_BPT, 8), stride=(8, 1))
    gX = cute.zipped_divide(mX, (1, _NVFP4_BPT * 16))
    gQ = cute.zipped_divide(mQ, (1, _NVFP4_BPT * 8))
    cX = cute.zipped_divide(cute.make_identity_tensor(mX.shape), (1, _NVFP4_BPT * 16))
    _nvfp4_blocked_kernel(gX, gQ, sflat, mOuter, cX, tv_layout, q_tv_layout,
                          cutlass.Int32(ncb)).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[_NVFP4_BPT, 1, 1])


def nvfp4_blocked_outer_cute(x, outer_blocked, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    q = torch.empty(M, N // 2, dtype=torch.uint8, device=x.device)
    ngc = N // 16
    nrb = (M + 127) // 128
    ncb = (ngc + 3) // 4
    s_u8 = torch.zeros(nrb, ncb, 32, 16, dtype=torch.uint8, device=x.device)
    mX = from_dlpack(x).mark_layout_dynamic()
    mQ = from_dlpack(q).mark_layout_dynamic()
    sflat = from_dlpack(s_u8.reshape(-1)).mark_layout_dynamic()
    mOuter = from_dlpack(outer_blocked.contiguous())  # (Mb, Nb) static
    fn = _compiled(("nvfp4_blocked_outer", M, N), _nvfp4_blocked_jit, mX, mQ, sflat, mOuter, ncb)
    fn(mX, mQ, sflat, mOuter)
    return q.view(torch.float4_e2m1fn_x2), s_u8.view(torch.float8_e4m3fn)


NVFP4_BLOCKED_OUTER = QuantCastCuteRecipe.from_gold(
    Nvfp4BlockedOuterGold, cute_fn=nvfp4_blocked_outer_cute
)


# ---------------------------------------------------------------------------
# 16x16 randomized Hadamard transform (bf16 in, bf16 out, no scale). Mirrors hadamard_rht_f:
# reshape the last dim into groups of 16 and right-multiply each group by the 16x16 RHT matrix
# (`out = x.reshape(..., 16) @ rht`). Memory-bound (4 bytes/element moved), but the per-group 16x16
# matmul must NOT be run on the fp32 CUDA cores: a scalar dot-product kernel is compute-bound at ~36%
# (256 fp32 MACs/group saturate the CUDA cores before DRAM), and torch.compile's cuBLAS GEMM stalls at
# ~29% (skinny K=N=16 tiles terribly). The fix is tensor cores: we run the transform as a batched GEMM
# on the SM80 warp-level bf16 `mma.sync` atom (m16n8k16), K=16 = a single k-step, exactly like the
# Triton `tl.dot` kernel.
#
# Layout: flatten x to (n_groups, 16); one m16n8k16 MMA consumes a 16-group x 16 tile. `cute.gemm`
# computes D[m,n] = sum_k A[m,k]*B[n,k], so with A = x tile (M=16 groups, K=16) and B[n,k] = rht[k,n]
# (i.e. rht^T, N=16) we get out[g,j] = sum_k x[g,k]*rht[k,j]. Each warp handles _RHT_TPW tiles; global
# <-> smem transfers are coalesced 128-bit vectorized copies (a warp streams _RHT_TPW*512 contiguous
# bf16), and smem <-> MMA register fragments go through the tiled-MMA partition (fast, smem-only). The
# 16x16 rht is staged in smem once per block, transposed on the fly (sB[n,k] = rht[k,n]) so the wrapper
# passes rht row-major with no host/runtime transpose. Reaches ~69% (fastest RHT path, ~triton parity,
# near the ~75% relu ceiling; ~2.3x compile, ~1.9x the scalar cute kernel).
#
# Bit-exactness vs the torch reference: bf16*bf16 is exact in fp32 (8+8 < 24 mantissa bits), so the
# tensor-core fp32 accumulation reproduces torch's bf16 matmul (fp32 products, fp32 accumulate,
# round-to-bf16) bit-for-bit -- verified against `x.reshape(...,16) @ rht` and against Triton's tl.dot.
# This matters because the cute test compares bf16 outputs to ~1 ULP fp32, i.e. it demands exactness.
# ---------------------------------------------------------------------------
_RHT_WARPS = 4                          # warps per block
_RHT_TPW = 2                            # 16-group tiles per warp (swept best: WARPSxTPW = 4x2)
_RHT_VW = 8                             # bf16 per vectorized copy (8 * 16b = 128b)
_RHT_THREADS = _RHT_WARPS * 32
_RHT_WT = _RHT_WARPS * _RHT_TPW         # tiles per block
_RHT_SMEM_ITERS = (256 + _RHT_THREADS - 1) // _RHT_THREADS  # threads-strided fill of the 256 rht elems


@cute.struct
class _RhtSmem:
    sA: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _RHT_WT * 256], 1024]
    sC: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _RHT_WT * 256], 1024]
    sB: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, 256], 1024]


@cute.kernel
def _rht_kernel(tiled_mma: cute.TiledMma, gX: cute.Tensor, gY: cute.Tensor, gR: cute.Tensor,
                ntiles: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    wp = tidx // 32
    smem = utils.SmemAllocator()
    st = smem.allocate(_RhtSmem)
    # smem views (pulled out before the branches so no python struct is live across an `if`).
    sBflat = st.sB.get_tensor(cute.make_layout(256))
    sBt = st.sB.get_tensor(cute.make_layout((16, 16), stride=(16, 1)))            # (N, K) = rht^T
    sA2 = st.sA.get_tensor(cute.make_layout((_RHT_WT, 256), stride=(256, 1)))     # per-tile flat (copy)
    sC2 = st.sC.get_tensor(cute.make_layout((_RHT_WT, 256), stride=(256, 1)))
    sA3 = st.sA.get_tensor(cute.make_layout((_RHT_WT, 16, 16), stride=(256, 16, 1)))  # (M, K) (MMA)
    sC3 = st.sC.get_tensor(cute.make_layout((_RHT_WT, 16, 16), stride=(256, 16, 1)))  # (M, N)
    tv = cute.make_layout((32, _RHT_VW), stride=(_RHT_VW, 1))                     # coalesced 128b copy
    cp = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=_RHT_VW * 16)

    # stage rht^T in smem once per block: sBt[n,k] = rht[k,n] = gR[k*16+n] (gR is rht row-major).
    for it in cutlass.range_constexpr(_RHT_SMEM_ITERS):
        p = tidx + it * _RHT_THREADS
        if p < 256:
            sBflat[p] = gR[(p % 16) * 16 + (p // 16)]
    cute.arch.sync_threads()

    base = bidx * _RHT_WT
    # pass 1: coalesced global -> smem for all of this warp's tiles.
    for t in cutlass.range_constexpr(_RHT_TPW):
        slot = wp * _RHT_TPW + t
        tile = base + slot
        if tile < ntiles:
            gAt = cute.composition(cute.local_tile(gX, (256,), (tile,)), tv)[(lane, None)]
            sAw = cute.composition(sA2[slot, None], tv)[(lane, None)]
            cute.copy(cp, gAt, sAw)
    cute.arch.sync_threads()

    # pass 2: tensor-core MMA per tile (B fragment loaded once, reused across the warp's tiles).
    thr_mma = tiled_mma.get_slice(lane)
    tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sBt))
    cute.autovec_copy(thr_mma.partition_B(sBt), tCrB)
    for t in cutlass.range_constexpr(_RHT_TPW):
        slot = wp * _RHT_TPW + t
        tile = base + slot
        if tile < ntiles:
            sA = sA3[slot, None, None]
            sC = sC3[slot, None, None]
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(sC))
            cute.autovec_copy(thr_mma.partition_A(sA), tCrA)
            for i in cutlass.range_constexpr(cute.size(tCrC)):
                tCrC[i] = cutlass.Float32(0.0)
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
            tCsC = thr_mma.partition_C(sC)
            for i in cutlass.range_constexpr(cute.size(tCrC)):
                tCsC[i] = tCrC[i].to(cutlass.BFloat16)
    cute.arch.sync_threads()

    # pass 3: coalesced smem -> global for all of this warp's tiles.
    for t in cutlass.range_constexpr(_RHT_TPW):
        slot = wp * _RHT_TPW + t
        tile = base + slot
        if tile < ntiles:
            sCw = cute.composition(sC2[slot, None], tv)[(lane, None)]
            gCt = cute.composition(cute.local_tile(gY, (256,), (tile,)), tv)[(lane, None)]
            cute.copy(cp, sCw, gCt)


@cute.jit
def _rht_jit(mX, mY, mR, ntiles: cutlass.Constexpr):
    mma_atom = cute.make_mma_atom(
        warp_mma.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)))
    tiled_mma = cute.make_tiled_mma(mma_atom)  # default atom layout = one warp (32 threads)
    nblocks = (ntiles + _RHT_WT - 1) // _RHT_WT
    _rht_kernel(tiled_mma, mX, mY, mR, ntiles).launch(
        grid=[nblocks, 1, 1], block=[_RHT_THREADS, 1, 1])


def rht_cute(x, rht, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert N % 16 == 0, f"rht cute kernel needs N % 16 == 0, got {N}"
    assert (M * N) % 256 == 0, "rht cute kernel needs numel % 256 == 0 (whole 16-group MMA tiles)"
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    ntiles = (M * N) // 256  # number of 16-group x 16 MMA tiles
    # assumed_align=16 enables the 128-bit vectorized copies (torch allocations are >=256B aligned).
    mX = from_dlpack(x.reshape(-1), assumed_align=16).mark_layout_dynamic()  # flatten (per-16-group)
    mY = from_dlpack(y.reshape(-1), assumed_align=16).mark_layout_dynamic()
    mR = from_dlpack(rht.reshape(-1)).mark_layout_dynamic()  # (256,) rht row-major; transposed in smem
    fn = _compiled(("rht", M, N), _rht_jit, mX, mY, mR, ntiles)
    fn(mX, mY, mR)
    return (y,)


BF16_RHT = QuantCastCuteRecipe.from_gold(HadamardRht, cute_fn=rht_cute)


# ---------------------------------------------------------------------------
# Stochastic-rounding fp32 -> bf16 (mirrors sr_bf16_f). SR add-then-truncate: dither the 16 mantissa
# bits fp32->bf16 drops with a uniform 16-bit value, then mask them off (& 0xFFFF0000); the low bits
# are then zero so f32->bf16 is an exact truncation. bf16 shares fp32's 8-bit exponent, so this is
# the simplest SR target -- no exponent rebias, no packing, no scale.
#
# RNG: the gold reference draws the dither from torch's Philox and the Triton kernel from its own
# `tl.randint4x`, so neither bit-matches the other -- only the SR *property* is well-defined (unbiased,
# every output lands on one of the two bracketing bf16 grid points), and that's all the test checks
# for the `_sr` recipes. CuTeDSL exposes no counter-based PRNG intrinsic (unlike Triton's
# `tl.randint4x`), so we implement Philox-4x32-10 by hand (`_philox_4x32`) out of the integer ops the
# DSL does have -- the same generator Triton/PyTorch use. It's a stateless pure function
# `(counter, key) -> 4 uniform uint32`, so every thread computes its own draws with no shared state.
# We key it like Triton's global SR kernel: counter = (flat index // 4), and the four outputs feed the
# four consecutive elements of that group (top 16 bits each = a uniform dither in [0, 2**16)). One
# Philox call per 4 elements amortizes the 10-round mix. E[SR(x)] = x holds (mean error ~1e-5 vs the
# 1e-3 tolerance on the 512x512 constant-input test).
#
# It's a pure elementwise streaming cast (read fp32, write bf16, no reduction), so it wants the same
# DRAM-speed-of-light recipe as fp8_tensorwise: FLATTEN to 1-D, each thread owning a CONTIGUOUS run
# loaded/stored via 128-bit VECTORIZED copy atoms (fp32 load 128b = 4 elems, bf16 store 128b = 8
# elems; assumed_align=16). The seed is loaded from a device (1,) int32 (the first word of the Philox
# key) so there's no host sync. numel % VPT == 0, so the single-coordinate predicate guards the tail.
# ---------------------------------------------------------------------------
_SR_THREADS = 256
_SR_VPT = 8                             # elems/thread; 8 fp32 = 2x128b load, 8 bf16 = 1x128b store
_SR_CHUNK = _SR_THREADS * _SR_VPT
_SR_LD_BITS = min(128, _SR_VPT * 32)    # fp32 input  -> 128-bit vectorized load
_SR_ST_BITS = min(128, _SR_VPT * 16)    # bf16 output -> 128-bit vectorized store

# Philox-4x32 round constants (Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"); the
# same values Random123 / Triton's tl.randint4x / PyTorch's _philox_uniform use. 10 rounds.
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9                 # Weyl key bump, word 0
_PHILOX_W1 = 0xBB67AE85                 # Weyl key bump, word 1
_PHILOX_ROUNDS = 10


@cute.jit
def _philox_4x32(c0, c1, c2, c3, k0, k1):
    """Philox-4x32-10: a stateless counter-based PRNG mapping a 128-bit counter (c0..c3) + 64-bit key
    (k0,k1) to four uniform uint32 outputs.

    We implement it by hand because CuTeDSL ships NO random-number generator -- there is no
    counter-based (or any) PRNG intrinsic in the device API (unlike Triton's `tl.randint4x` or
    PyTorch's `aten._philox_uniform`), so to draw the SR dither on-device we build the same standard
    generator out of the integer ops the DSL does expose (XOR, add, shift, and a Uint64-widened
    multiply for the mulhilo step).

    Each round multiplies two counter words by the M constants, keeps the hi/lo halves of the 64-bit
    products (mulhilo via the Uint64 widen), and mixes them with the other two counter words and the
    key; the key is bumped by the Weyl constants between rounds. Bit-for-bit the standard Philox4x32
    (verified against the Random123 reference)."""
    for _ in cutlass.range_constexpr(_PHILOX_ROUNDS):
        p0 = cutlass.Uint64(c0) * cutlass.Uint64(_PHILOX_M0)
        p1 = cutlass.Uint64(c2) * cutlass.Uint64(_PHILOX_M1)
        hi0 = cutlass.Uint32(p0 >> 32)
        lo0 = cutlass.Uint32(p0 & cutlass.Uint64(0xFFFFFFFF))
        hi1 = cutlass.Uint32(p1 >> 32)
        lo1 = cutlass.Uint32(p1 & cutlass.Uint64(0xFFFFFFFF))
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
        k0 = k0 + cutlass.Uint32(_PHILOX_W0)
        k1 = k1 + cutlass.Uint32(_PHILOX_W1)
    return c0, c1, c2, c3


@cute.kernel
def _sr_bf16_kernel(gX: cute.Tensor, gY: cute.Tensor, gSeed: cute.Tensor, cX: cute.Tensor,
                    shape: cute.Shape, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    coord = (None, bidx)
    thrX = cute.composition(gX[coord], tv_layout)[(tidx, None)]
    thrY = cute.composition(gY[coord], tv_layout)[(tidx, None)]
    thrC = cute.composition(cX[coord], tv_layout)[(tidx, None)]

    if cute.elem_less(thrC[0], shape):
        ld_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type,
                                      num_bits_per_copy=_SR_LD_BITS)
        st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type,
                                      num_bits_per_copy=_SR_ST_BITS)
        frgSeed = cute.make_rmem_tensor(cute.make_layout(1), gSeed.element_type)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gSeed.element_type), gSeed, frgSeed)
        seed = cute.recast_tensor(frgSeed, dtype=cutlass.Uint32)[0]  # bitcast the int32 key word
        frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
        cute.copy(ld_atom, thrX, frgX)
        p0 = thrC[0][0]                                   # flat index of this thread's first element
        frgXi = cute.recast_tensor(frgX, dtype=cutlass.Int32)  # reinterpret the f32 bits in place
        zero = cutlass.Uint32(0)
        # p0 is a multiple of VPT (hence of 4), so the thread's run splits into VPT//4 aligned groups
        # of 4 global-consecutive elements; one Philox call (counter = flat//4) dithers each group.
        for g in cutlass.range_constexpr(_SR_VPT // 4):
            ctr = cutlass.Uint32(p0 // 4) + cutlass.Uint32(g)
            r0, r1, r2, r3 = _philox_4x32(ctr, zero, zero, zero, seed, zero)
            b = g * 4
            # add the top-16-bit dither, then truncate the low 16 mantissa bits (-65536 == 0xFFFF0000).
            frgXi[b + 0] = (frgXi[b + 0] + cutlass.Int32(r0 >> 16)) & cutlass.Int32(-65536)
            frgXi[b + 1] = (frgXi[b + 1] + cutlass.Int32(r1 >> 16)) & cutlass.Int32(-65536)
            frgXi[b + 2] = (frgXi[b + 2] + cutlass.Int32(r2 >> 16)) & cutlass.Int32(-65536)
            frgXi[b + 3] = (frgXi[b + 3] + cutlass.Int32(r3 >> 16)) & cutlass.Int32(-65536)
        frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
        frgY.store(frgX.load().to(cutlass.BFloat16))      # exact: low 16 bits are zero
        cute.copy(st_atom, frgY, thrY)


@cute.jit
def _sr_bf16_jit(mX, mY, mSeed):
    tv_layout = cute.make_layout((_SR_THREADS, _SR_VPT), stride=(_SR_VPT, 1))
    tiler = (cute.size(tv_layout),)
    gX = cute.zipped_divide(mX, tiler)
    gY = cute.zipped_divide(mY, tiler)
    cX = cute.zipped_divide(cute.make_identity_tensor(mX.shape), tiler)
    _sr_bf16_kernel(gX, gY, mSeed, cX, mX.shape, tv_layout).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[cute.size(tv_layout, mode=[0]), 1, 1])


def sr_bf16_cute(x, key, **kwargs):
    """Matches sr_bf16_f: fp32 -> bf16 stochastic rounding. `key` is a Philox key tensor; its first
    32-bit word seeds the in-kernel fmix32 dither (loaded on-device, no host sync). Returns `(out,)`."""
    assert x.dtype == torch.float32, f"SR bf16 expects fp32 input, got {x.dtype}"
    assert x.is_contiguous() and x.dim() == 2
    assert _SR_VPT % 4 == 0, "SR Philox groups 4 elements per call, so VPT must be a multiple of 4"
    M, N = x.shape
    assert (M * N) % _SR_VPT == 0, f"sr_bf16 cute kernel needs numel % {_SR_VPT} == 0"
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    seed = key.reshape(-1)[:1].view(torch.int32).reshape(-1)[:1]  # first 32 bits of the key (device)
    # assumed_align=16 enables the 128-bit vectorized copies (torch allocations are >=256B aligned).
    mX = from_dlpack(x.reshape(-1), assumed_align=16).mark_layout_dynamic()  # flatten (elementwise)
    mY = from_dlpack(y.reshape(-1), assumed_align=16).mark_layout_dynamic()
    mSeed = from_dlpack(seed)
    fn = _compiled(("sr_bf16", M, N), _sr_bf16_jit, mX, mY, mSeed)
    fn(mX, mY, mSeed)
    return (y,)


SR_F32_TO_BF16 = QuantCastCuteRecipe.from_gold(SrF32ToBf16, cute_fn=sr_bf16_cute)


ALL_RECIPES = [
    # elementwise
    ("fp8_tensorwise_precalc_scale", FP8_TENSORWISE_PRECALC_SCALE),
    ("fp8_rowwise_precalc_scale", FP8_ROWWISE_PRECALC_SCALE),
    ("fp8_colwise_precalc_scale", FP8_COLWISE_PRECALC_SCALE),
    # 1x32, 8-bit
    ("mxfp8_floor", MXFP8_FLOOR),
    ("mxfp8_floor_swizzle", MXFP8_FLOOR_SWIZZLE),
    ("mxfp8_floor_dim_m", MXFP8_FLOOR_DIM_M),
    ("mxfp8_floor_dim_km", MXFP8_FLOOR_DIM_KM),
    ("mxfp8_32x32_floor", MXFP8_32X32_FLOOR),
    # 1x128, 8-bit
    ("fp8_deepseek_1x128", FP8_DEEPSEEK_1X128),
    ("fp8_deepseek_1x128_dim_m", FP8_DEEPSEEK_1X128_DIM_M),
    ("fp8_deepseek_1x128_dim_km", FP8_DEEPSEEK_1X128_DIM_KM),
    # 128x128, 8-bit
    ("fp8_deepseek_128x128", FP8_DEEPSEEK_128X128),
    # rowwise/colwise, 8-bit
    ("fp8_rowwise", FP8_ROWWISE),
    ("fp8_colwise", FP8_COLWISE),
    # 1x16, 4-bit
    ("nvfp4_swizzle", NVFP4_SWIZZLE),
    ("nvfp4_blocked_outer", NVFP4_BLOCKED_OUTER),
    # RHT
    ("bf16_rht", BF16_RHT),
    # stochastic rounding
    ("fp32_to_bf16_sr", SR_F32_TO_BF16),
]
