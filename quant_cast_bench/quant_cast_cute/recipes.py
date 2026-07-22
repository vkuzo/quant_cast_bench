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
from cutlass.cute.nvgpu import cpasync  # TMA (bulk-tensor) copy ops + tma_partition
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import _maybe_recast_from_f4  # packs an fp4 register vector to bytes

from quant_cast_bench.quant_cast_gold.recipes import (
    ColwiseFp8Gold,
    ColwisePrecalcGold,
    Deepseek1x128DimMGold,
    Deepseek1x128Gold,
    Deepseek128x128Gold,
    Float8TensorwiseGold,
    Mxfp832x32FloorGold,
    Mxfp8FloorDimMGold,
    Mxfp8FloorGold,
    Mxfp8FloorSwizzleGold,
    Nvfp4BlockedOuterGold,
    Nvfp4GsSwizzleGold,
    QuantCastSingleKernelGold,
    RowwiseFp8Gold,
    RowwisePrecalcGold,
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
# mxfp8 FLOOR 32x32: one e8m0 scale per 32x32 block. One warp (32 threads x 32 vals) per block:
# thread r owns row r of the block (32 contiguous cols), intra-thread abs-max then warp_reduction
# over the 32 rows -> block amax. Grid = one warp per (M//32, N//32) block. Mirrors mxfp8_32x32.
# ---------------------------------------------------------------------------
@cute.kernel
def _mxfp8_32x32_kernel(gX: cute.Tensor, gY: cute.Tensor, gScale: cute.Tensor, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    thrX = cute.composition(gX[(None, bidx)], tv_layout)[(tidx, None)]
    thrY = cute.composition(gY[(None, bidx)], tv_layout)[(tidx, None)]
    frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type), thrX, frgX)
    v = frgX.load().to(cutlass.Float32)
    local = cutlass.max(v, -v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    amax = cute.arch.warp_reduction_max(local)  # across the 32 rows of the block
    sfp, biased = _e8m0_floor(amax)
    frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
    frgY.store((v / sfp).to(gY.element_type))
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type), frgY, thrY)
    if tidx == 0:
        gScale[(None, bidx)][0] = biased.to(gScale.element_type)


@cute.jit
def _mxfp8_32x32_jit(mX, mY, mScale):
    tv_layout = cute.make_layout((32, 32), stride=(32, 1))  # 32 threads (rows) x 32 vals (cols)
    gX = cute.zipped_divide(mX, (32, 32))
    gY = cute.zipped_divide(mY, (32, 32))
    gScale = cute.zipped_divide(mScale, (1, 1))
    _mxfp8_32x32_kernel(gX, gY, gScale, tv_layout).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[32, 1, 1])


def mxfp8_32x32_floor_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    s_u8 = torch.empty(M // 32, N // 32, dtype=torch.uint8, device=x.device)
    mX = from_dlpack(x).mark_layout_dynamic()
    mY = from_dlpack(y).mark_layout_dynamic()
    mS = from_dlpack(s_u8).mark_layout_dynamic()
    fn = _compiled(("mxfp8_32x32", M, N), _mxfp8_32x32_jit, mX, mY, mS)
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
# fp8 rowwise (full-span): one fp32 scale per row, amax over ALL columns. One warp per row: thread t
# owns a contiguous N//32 chunk of the row, intra-thread abs-max then warp_reduction over 32 threads
# -> row amax; then quantize its chunk and store. Mirrors rowwise_fp8_f. Requires N % 32 == 0.
# (Correctness-first: one warp/row and a single-pass N//32 fragment; large-N perf is a later pass.)
# ---------------------------------------------------------------------------
@cute.kernel
def _fp8_rowwise_kernel(gX: cute.Tensor, gY: cute.Tensor, gScale: cute.Tensor, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    thrX = cute.composition(gX[(None, bidx)], tv_layout)[(tidx, None)]
    thrY = cute.composition(gY[(None, bidx)], tv_layout)[(tidx, None)]
    frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type), thrX, frgX)
    v = frgX.load().to(cutlass.Float32)
    local = cutlass.max(v, -v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    amax = cutlass.max(cute.arch.warp_reduction_max(local), cutlass.Float32(1e-12))
    scale = amax / 448.0
    frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
    frgY.store((v * (1.0 / scale)).to(gY.element_type))
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type), frgY, thrY)
    if tidx == 0:
        gScale[(None, bidx)][0] = scale


@cute.jit
def _fp8_rowwise_jit(mX, mY, mScale, vpt: cutlass.Constexpr):
    tv_layout = cute.make_layout((32, vpt), stride=(vpt, 1))  # 32 threads x N//32 vals = one row
    n = cute.size(mX, mode=[1])
    gX = cute.zipped_divide(mX, (1, n))
    gY = cute.zipped_divide(mY, (1, n))
    gScale = cute.zipped_divide(mScale, (1, 1))
    _fp8_rowwise_kernel(gX, gY, gScale, tv_layout).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[32, 1, 1])


def fp8_rowwise_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert N % 32 == 0, "fp8_rowwise cute kernel needs N % 32 == 0"
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(M, 1, dtype=torch.float32, device=x.device)
    mX = from_dlpack(x).mark_layout_dynamic()
    mY = from_dlpack(y).mark_layout_dynamic()
    mS = from_dlpack(s)  # (M, 1) stride (1,1): keep static (ambiguous leading dim otherwise)
    fn = _compiled(("fp8_rowwise", M, N), _fp8_rowwise_jit, mX, mY, mS, N // 32)
    fn(mX, mY, mS)
    return y, s


FP8_ROWWISE = QuantCastCuteRecipe.from_gold(RowwiseFp8Gold, cute_fn=fp8_rowwise_cute)


# ---------------------------------------------------------------------------
# fp8 colwise (full-span): one fp32 scale per column, amax over ALL rows; transposed outputs
# (N, M) / (N, 1). Exactly fp8_rowwise on the transposed view x.t() (reduce its last dim = all rows
# of x; the (N, M)-contiguous output view is the transpose). Reuse the rowwise kernel. Needs M%32==0.
# ---------------------------------------------------------------------------
def fp8_colwise_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0, "fp8_colwise cute kernel needs M % 32 == 0"
    y = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)  # transposed output
    s = torch.empty(N, 1, dtype=torch.float32, device=x.device)
    mX = from_dlpack(x.t()).mark_layout_dynamic()  # (N, M) strided view; reduce its last dim
    mY = from_dlpack(y).mark_layout_dynamic()
    mS = from_dlpack(s)  # (N, 1) stride (1,1): keep static (ambiguous leading dim otherwise)
    fn = _compiled(("fp8_colwise", M, N), _fp8_rowwise_jit, mX, mY, mS, M // 32)
    fn(mX, mY, mS)
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


# nvfp4 with a per-tensor (global) outer scale: 1x16 inner blocks, e4m3 inner scale, fp4-packed
# qdata, inner scale in the swizzled 4D grid. One thread per 1x16 block. Mirrors nvfp4_gs_swizzle_f.
@cute.kernel
def _nvfp4_swizzle_kernel(gX: cute.Tensor, gQ: cute.Tensor, sflat: cute.Tensor, mOuter: cute.Tensor,
                          cX: cute.Tensor, tv_layout: cute.Layout, q_tv_layout: cute.Layout,
                          ncb: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    thrX = cute.composition(gX[(None, bidx)], tv_layout)[(tidx, None)]
    thrQ = cute.composition(gQ[(None, bidx)], q_tv_layout)[(tidx, None)]
    thrC = cute.composition(cX[(None, bidx)], tv_layout)[(tidx, None)]
    frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type), thrX, frgX)
    frgO = cute.make_rmem_tensor(cute.make_layout(1), mOuter.element_type)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mOuter.element_type), mOuter, frgO)

    packed, inner_e4 = _nvfp4_block(frgX.load().to(cutlass.Float32),
                                    cute.get(tv_layout, mode=[1]), frgO[0])
    frgQ = cute.make_rmem_tensor(cute.get(q_tv_layout, mode=[1]), gQ.element_type)
    frgQ.store(packed)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gQ.element_type), frgQ, thrQ)
    flat = _swizzle_flat(thrC[0][0], thrC[0][1] // 16, ncb)  # col = 16-group index
    fib = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float8E4M3FN)
    fib[0] = inner_e4
    sflat[flat] = cute.recast_tensor(fib, dtype=cutlass.Uint8)[0]


_NVFP4_BPT = 8  # 1x16 blocks per CTA


@cute.jit
def _nvfp4_swizzle_jit(mX, mQ, sflat, mOuter, ncb: cutlass.Constexpr):
    tv_layout = cute.make_layout((_NVFP4_BPT, 16), stride=(16, 1))
    q_tv_layout = cute.make_layout((_NVFP4_BPT, 8), stride=(8, 1))  # 16 fp4 -> 8 bytes
    gX = cute.zipped_divide(mX, (1, _NVFP4_BPT * 16))
    gQ = cute.zipped_divide(mQ, (1, _NVFP4_BPT * 8))
    cX = cute.zipped_divide(cute.make_identity_tensor(mX.shape), (1, _NVFP4_BPT * 16))
    _nvfp4_swizzle_kernel(gX, gQ, sflat, mOuter, cX, tv_layout, q_tv_layout,
                          cutlass.Int32(ncb)).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[_NVFP4_BPT, 1, 1])


def nvfp4_swizzle_cute(x, outer_scale, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    q = torch.empty(M, N // 2, dtype=torch.uint8, device=x.device)
    ngc = N // 16
    nrb = (M + 127) // 128
    ncb = (ngc + 3) // 4
    s_u8 = torch.zeros(nrb, ncb, 32, 16, dtype=torch.uint8, device=x.device)  # padding stays 0
    mX = from_dlpack(x).mark_layout_dynamic()
    mQ = from_dlpack(q).mark_layout_dynamic()
    sflat = from_dlpack(s_u8.reshape(-1)).mark_layout_dynamic()
    mOuter = from_dlpack(outer_scale.reshape(1))  # per-tensor scalar (static)
    fn = _compiled(("nvfp4_swizzle", M, N), _nvfp4_swizzle_jit, mX, mQ, sflat, mOuter, ncb)
    fn(mX, mQ, sflat, mOuter)
    return q.view(torch.float4_e2m1fn_x2), s_u8.view(torch.float8_e4m3fn)


NVFP4_SWIZZLE = QuantCastCuteRecipe.from_gold(Nvfp4GsSwizzleGold, cute_fn=nvfp4_swizzle_cute)


# nvfp4 with a 128x128-blocked outer scale (Mb, Nb): identical to nvfp4_swizzle but the outer scale
# is looked up per block from outer_blocked[row//128, (col*16)//128]. Mirrors nvfp4_blocked_outer_f.
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


ALL_RECIPES = [
    # elementwise
    ("fp8_tensorwise_precalc_scale", FP8_TENSORWISE_PRECALC_SCALE),
    ("fp8_rowwise_precalc_scale", FP8_ROWWISE_PRECALC_SCALE),
    ("fp8_colwise_precalc_scale", FP8_COLWISE_PRECALC_SCALE),
    # 1x32, 8-bit
    ("mxfp8_floor", MXFP8_FLOOR),
    ("mxfp8_floor_swizzle", MXFP8_FLOOR_SWIZZLE),
    ("mxfp8_floor_dim_m", MXFP8_FLOOR_DIM_M),
    ("mxfp8_32x32_floor", MXFP8_32X32_FLOOR),
    # 1x128, 8-bit
    ("fp8_deepseek_1x128", FP8_DEEPSEEK_1X128),
    ("fp8_deepseek_1x128_dim_m", FP8_DEEPSEEK_1X128_DIM_M),
    # 128x128, 8-bit
    ("fp8_deepseek_128x128", FP8_DEEPSEEK_128X128),
    # rowwise/colwise, 8-bit
    ("fp8_rowwise", FP8_ROWWISE),
    ("fp8_colwise", FP8_COLWISE),
    # 1x16, 4-bit
    ("nvfp4_swizzle", NVFP4_SWIZZLE),
    ("nvfp4_blocked_outer", NVFP4_BLOCKED_OUTER),
]
