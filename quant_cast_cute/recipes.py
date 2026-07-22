"""CuTeDSL (cutlass.cute) implementations of the quant_cast_gold recipes.

Mirrors quant_cast_triton: each recipe is a `QuantCastCuteRecipe` inheriting the gold reference
(`pt_ref_fn`/`correctness_fn`/`example_input_fn`/`perf_description`) from a
`QuantCastSingleKernelGold` and adding `cute_fn`, a CuTeDSL implementation of the same cast.
test.py grades each `cute_fn` against its gold `pt_ref_fn`.

Correctness-first: kernels are naive (tiling not tuned). `cute.compile` is cached per (recipe,
shape) since it is slow and the benchmark calls each fn many times.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import _maybe_recast_from_f4  # packs an fp4 register vector to bytes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_gold.recipes import (
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
# deepseek fp8 1x128: one fp32 scale per 1x128 block. One warp (32 threads x 4 vals) per block;
# abs-max via intra-thread reduce + warp_reduction_max. x is tiled (1,128) and the scale (M,N//128)
# tiled (1,1) so both share the same block index (bidx) -- no scatter. Mirrors deepseek_1x128_f.
# ---------------------------------------------------------------------------
@cute.kernel
def _deepseek_1x128_kernel(gX: cute.Tensor, gY: cute.Tensor, gScale: cute.Tensor, tv_layout: cute.Layout):
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
def _deepseek_1x128_jit(mX, mY, mScale):
    tv_layout = cute.make_layout((32, 4), stride=(4, 1))  # 32 threads x 4 vals = one 1x128 block
    gX = cute.zipped_divide(mX, (1, 128))
    gY = cute.zipped_divide(mY, (1, 128))
    gScale = cute.zipped_divide(mScale, (1, 1))
    _deepseek_1x128_kernel(gX, gY, gScale, tv_layout).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[32, 1, 1]
    )


def fp8_deepseek_1x128_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(M, N // 128, dtype=torch.float32, device=x.device)
    mX = from_dlpack(x).mark_layout_dynamic()
    mY = from_dlpack(y).mark_layout_dynamic()
    mS = from_dlpack(s).mark_layout_dynamic()
    fn = _compiled(("deepseek_1x128", M, N), _deepseek_1x128_jit, mX, mY, mS)
    fn(mX, mY, mS)
    return y, s


FP8_DEEPSEEK_1X128 = QuantCastCuteRecipe.from_gold(
    Deepseek1x128Gold, cute_fn=fp8_deepseek_1x128_cute
)


# ---------------------------------------------------------------------------
# deepseek fp8 1x128 dim-M: reduce 128-row blocks down M, transposed outputs (N, M) / (N, M//128).
# This is exactly deepseek_1x128 applied to the TRANSPOSED view x.t(): reducing "1x128 along the
# last dim of x.t()" reduces 128 rows of x, and the (N, M)-contiguous output view IS the transpose.
# So we reuse `_deepseek_1x128_jit` verbatim, just feeding it x.t() and (N, M)/(N, M//128) buffers.
# ---------------------------------------------------------------------------
def fp8_deepseek_1x128_dim_m_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)  # transposed output
    s = torch.empty(N, M // 128, dtype=torch.float32, device=x.device)
    mX = from_dlpack(x.t()).mark_layout_dynamic()  # (N, M) strided view; reduce its last dim
    mY = from_dlpack(y).mark_layout_dynamic()
    mS = from_dlpack(s).mark_layout_dynamic()
    fn = _compiled(("deepseek_1x128_dim_m", M, N), _deepseek_1x128_jit, mX, mY, mS)
    fn(mX, mY, mS)
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
# uncoalesced (ncu: DRAM 4.6%, store 32 sectors/request). Instead we read x NATIVELY in coalesced
# (32,32) tiles, do a SHARED-MEMORY TRANSPOSE, and store coalesced:
#   - thread t loads column t of the tile (warp reads 32 contiguous rows -> coalesced), computes
#     that column's e8m0 scale intra-thread (no cross-thread reduce), quantizes its 32 rows;
#   - each thread writes its quantized column into a padded (32x33) smem tile (col write is
#     conflict-free; 33 pad makes the row read conflict-free too), __syncthreads;
#   - thread t reads row t of smem and does a COALESCED transposed store (tv stride (1,32): adjacent
#     threads -> adjacent output columns). qdata lands in y.t() (M,N)-view = the (N,M) transpose.
# Result: ~35% of B200 peak at 16384 (7x over the x.t() reuse), ld/st ~2 sectors/request (coalesced),
# on par with mxfp8_32x32. Remaining ceiling is ~50% occupancy (64 regs/thread, 1 warp/block).
# ---------------------------------------------------------------------------
@cute.kernel
def _mxfp8_dim_m_kernel(gX: cute.Tensor, gY: cute.Tensor, sflat: cute.Tensor, cX: cute.Tensor,
                        tv_col: cute.Layout, tv_store: cute.Layout, m_blocks: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = cutlass.utils.SmemAllocator()
    sT = smem.allocate_tensor(gY.element_type, cute.make_layout((32, 33), stride=(33, 1)),
                              byte_alignment=16)
    # phase 1: coalesced column load, per-column amax + e8m0 quantize into a register column
    thrX = cute.composition(gX[(None, bidx)], tv_col)[(tidx, None)]
    thrC = cute.composition(cX[(None, bidx)], tv_col)[(tidx, None)]
    frgX = cute.make_rmem_tensor(cute.get(tv_col, mode=[1]), gX.element_type)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type), thrX, frgX)
    v = frgX.load().to(cutlass.Float32)
    amax = cutlass.max(v, -v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    sfp, biased = _e8m0_floor(amax)
    sflat[thrC[0][1] * m_blocks + (thrC[0][0] // 32)] = biased.to(sflat.element_type)  # s[col, rb]
    frgYc = cute.make_rmem_tensor(cute.get(tv_col, mode=[1]), gY.element_type)
    frgYc.store((v / sfp).to(gY.element_type))
    # transpose through smem: thread t writes its column, __sync, then reads its row
    for r in cutlass.range_constexpr(32):
        sT[r, tidx] = frgYc[r]
    cute.arch.sync_threads()
    frgYr = cute.make_rmem_tensor(cute.get(tv_store, mode=[1]), gY.element_type)
    for c in cutlass.range_constexpr(32):
        frgYr[c] = sT[tidx, c]
    # phase 2: coalesced transposed store (into the y.t() view)
    thrY = cute.composition(gY[(None, bidx)], tv_store)[(tidx, None)]
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type), frgYr, thrY)


@cute.jit
def _mxfp8_dim_m_jit(mX, mY, sflat, m_blocks: cutlass.Constexpr):
    tv_col = cute.make_layout((32, 32), stride=(32, 1))    # thread t -> column t (32 rows)
    tv_store = cute.make_layout((32, 32), stride=(1, 32))  # thread t -> row t; value c -> col c
    gX = cute.zipped_divide(mX, (32, 32))
    gY = cute.zipped_divide(mY, (32, 32))
    cX = cute.zipped_divide(cute.make_identity_tensor(mX.shape), (32, 32))
    _mxfp8_dim_m_kernel(gX, gY, sflat, cX, tv_col, tv_store, cutlass.Int32(m_blocks)).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[32, 1, 1])


def mxfp8_floor_dim_m_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0 and N % 32 == 0, "mxfp8_floor_dim_m cute kernel needs M,N % 32 == 0"
    y = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)  # transposed output
    s_u8 = torch.empty(N, M // 32, dtype=torch.uint8, device=x.device)
    mX = from_dlpack(x, assumed_align=16).mark_layout_dynamic()      # read x natively (coalesced)
    mY = from_dlpack(y.t(), assumed_align=16).mark_layout_dynamic()  # (M,N)-view of the (N,M) output
    sflat = from_dlpack(s_u8.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("mxfp8_floor_dim_m", M, N), _mxfp8_dim_m_jit, mX, mY, sflat, M // 32)
    fn(mX, mY, sflat)
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
# ---------------------------------------------------------------------------
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
                                cX: cute.Tensor, tv_layout: cute.Layout, ncb: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    thrX = cute.composition(gX[(None, bidx)], tv_layout)[(tidx, None)]
    thrY = cute.composition(gY[(None, bidx)], tv_layout)[(tidx, None)]
    thrC = cute.composition(cX[(None, bidx)], tv_layout)[(tidx, None)]
    frgX = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gX.element_type)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type), thrX, frgX)
    v = frgX.load().to(cutlass.Float32)
    amax = cutlass.max(v, -v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    sfp, biased = _e8m0_floor(amax)
    frgY = cute.make_rmem_tensor(cute.get(tv_layout, mode=[1]), gY.element_type)
    frgY.store((v / sfp).to(gY.element_type))
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type), frgY, thrY)
    # pre-swizzle (row, 32-group col) from the coordinate tensor; scatter the e8m0 byte to the grid.
    flat = _swizzle_flat(thrC[0][0], thrC[0][1] // 32, ncb)
    sflat[flat] = biased.to(sflat.element_type)


@cute.jit
def _mxfp8_floor_swizzle_jit(mX, mY, sflat, ncb: cutlass.Constexpr):
    bpt = _MXFP8_1X32_BPT
    tv_layout = cute.make_layout((bpt, 32), stride=(32, 1))
    gX = cute.zipped_divide(mX, (1, bpt * 32))
    gY = cute.zipped_divide(mY, (1, bpt * 32))
    cX = cute.zipped_divide(cute.make_identity_tensor(mX.shape), (1, bpt * 32))
    _mxfp8_floor_swizzle_kernel(gX, gY, sflat, cX, tv_layout, cutlass.Int32(ncb)).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[bpt, 1, 1])


def mxfp8_floor_swizzle_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    ngc = N // 32
    nrb = (M + 127) // 128
    ncb = (ngc + 3) // 4
    s_u8 = torch.zeros(nrb, ncb, 32, 16, dtype=torch.uint8, device=x.device)  # padding stays 0
    mX = from_dlpack(x).mark_layout_dynamic()
    mY = from_dlpack(y).mark_layout_dynamic()
    sflat = from_dlpack(s_u8.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("mxfp8_floor_swizzle", M, N), _mxfp8_floor_swizzle_jit, mX, mY, sflat, ncb)
    fn(mX, mY, sflat)
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_FLOOR_SWIZZLE = QuantCastCuteRecipe.from_gold(
    Mxfp8FloorSwizzleGold, cute_fn=mxfp8_floor_swizzle_cute
)


# ---------------------------------------------------------------------------
# deepseek fp8 128x128: one fp32 scale per 128x128 block (amax over the whole block). One warp per
# block; thread r owns 4 chunks of 128 cols across 4 of the 128 rows... simpler: 32 threads each own
# 512 vals (128*128/32 = 512) via a (32, 512) tv-layout over the flattened block, intra-thread reduce
# then warp_reduction over the 32 threads. Mirrors deepseek_128x128_f. Grid = (M//128, N//128).
# ---------------------------------------------------------------------------
@cute.kernel
def _deepseek_128x128_kernel(gX: cute.Tensor, gY: cute.Tensor, gScale: cute.Tensor, tv_layout: cute.Layout):
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
def _deepseek_128x128_jit(mX, mY, mScale):
    # 32 threads x 512 vals cover the 128x128 = 16384-element block; thread t owns rows where
    # (row*128 + col) falls in its stripe. Layout the block row-major (128,128) and tile with a
    # (32, 512) TV mapping over the 16384 linear positions.
    tv_layout = cute.make_layout((32, 512), stride=(512, 1))
    gX = cute.zipped_divide(mX, (128, 128))
    gY = cute.zipped_divide(mY, (128, 128))
    gScale = cute.zipped_divide(mScale, (1, 1))
    _deepseek_128x128_kernel(gX, gY, gScale, tv_layout).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1], block=[32, 1, 1])


def fp8_deepseek_128x128_cute(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(M // 128, N // 128, dtype=torch.float32, device=x.device)
    mX = from_dlpack(x).mark_layout_dynamic()
    mY = from_dlpack(y).mark_layout_dynamic()
    mS = from_dlpack(s).mark_layout_dynamic()
    fn = _compiled(("deepseek_128x128", M, N), _deepseek_128x128_jit, mX, mY, mS)
    fn(mX, mY, mS)
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
