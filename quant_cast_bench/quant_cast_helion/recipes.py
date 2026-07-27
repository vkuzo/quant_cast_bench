"""Helion implementations of the quant_cast_gold recipes.

Each recipe is a `QuantCastHelionRecipe` -- it inherits the gold reference
(`pt_ref_fn`/`correctness_fn`/`example_input_fn`/`perf_description`) from a
`QuantCastSingleKernelGold` and adds `helion_fn`, a Helion-backed implementation of the same
cast. Mirrors quant_cast_triton's `QuantCastTritonRecipe` (inherit-from-gold + `from_gold`).
test/test_quant_cast_helion.py grades each `helion_fn` against its gold `pt_ref_fn`.

Recipes are ported over time; `fp8_deepseek_1x128_dim_m` and `mxfp8_floor_dim_m` are done.
"""

from dataclasses import dataclass
from typing import Callable

import helion
import helion.language as hl
import torch

from quant_cast_bench.quant_cast_gold.recipes import (
    Deepseek1x128DimMGold,
    Mxfp832x32FloorGold,
    Mxfp8FloorDimMGold,
    Nvfp4GsGold,
    QuantCastSingleKernelGold,
)

# fp8_e4m3fn representable max; the deepseek scale is amax / fp8_max (see the gold recipe).
_FP8_MAX = 448.0

# e8m0 (float8_e8m0fnu) power-of-two scale bit-math constants (see gold _amax_to_e8m0_floor /
# _e8m0_to_fp32). The mxfp8-floor recipes derive the scale by extracting amax's fp32 exponent via
# integer bit-ops (FLOOR, no log2) rather than an fp32 divide.
_E8M0_BIAS = 127
_F32_EXP_BIAS = 127
_F32_MBITS = 23
_F8E4M3_MAX_POW2 = 8


def _amax_to_e8m0_floor_biased(amax):
    """fp32 amax tile -> biased e8m0 exponent (int32, 0..255) via FLOOR exponent extraction. Ports
    the integer path of the gold `_amax_to_e8m0_floor`; the NaN branch is omitted since the
    benchmark/test inputs are finite (matches bit-for-bit on finite amax)."""
    max_abs_int32 = amax.view(torch.int32)
    extracted_pow2 = ((max_abs_int32 >> _F32_MBITS) & 0xFF) - _F32_EXP_BIAS
    scale_unbiased = extracted_pow2 - _F8E4M3_MAX_POW2
    scale_unbiased = torch.clamp(scale_unbiased, -_E8M0_BIAS, _E8M0_BIAS + 1)
    return scale_unbiased + _E8M0_BIAS  # biased exponent, int32


def _e8m0_biased_to_fp32(biased):
    """biased e8m0 exponent (int32 tile) -> fp32 pow2 factor. Ports the gold `_e8m0_to_fp32`
    inverse cast: shift the biased exponent back into the fp32 exponent field."""
    scale_fp32 = (biased << _F32_MBITS).view(torch.float32)
    return torch.clamp(scale_fp32, min=2.0**-126)


@dataclass(frozen=True)
class QuantCastHelionRecipe(QuantCastSingleKernelGold):
    """A gold recipe plus a Helion implementation of its `pt_ref_fn`. Mirrors quant_cast_triton's
    QuantCastTritonRecipe: inherits pt_ref_fn/correctness_fn/example_input_fn/perf_description from
    the gold, and adds `helion_fn` (same `(inputs) -> outputs` signature as `pt_ref_fn`)."""

    helion_fn: Callable | None = None

    @classmethod
    def from_gold(
        cls, gold: QuantCastSingleKernelGold, helion_fn: Callable
    ) -> "QuantCastHelionRecipe":
        """Build a QuantCastHelionRecipe from a gold recipe, attaching its Helion implementation."""
        return cls(
            pt_ref_fn=gold.pt_ref_fn,
            correctness_fn=gold.correctness_fn,
            example_input_fn=gold.example_input_fn,
            perf_description=gold.perf_description,
            helion_fn=helion_fn,
        )


# ---------------------------------------------------------------------------
# deepseek fp8 1x128, reduced across M (128x1 blocks along rows), transposed output.
# ---------------------------------------------------------------------------
@helion.kernel(config=helion.Config(
    atomic_indexing=[], 
    block_sizes=[1, 128], 
    indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 
    l2_groupings=[2], 
    load_eviction_policies=['last', 'last', 'first', 'last'], 
    loop_orders=[[0, 1]], 
    num_stages=3, 
    num_warps=4, 
    pid_type='flat', 
    range_flattens=[None], 
    range_multi_buffers=[None], 
    range_num_stages=[], 
    range_unroll_factors=[0], 
    range_warp_specializes=[None], 
    reduction_loops=[None]
), static_shapes=True)
def _deepseek_1x128_dim_m_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    qdata: torch.Tensor,  # (N, M) fp8_e4m3fn, mutated in place (t-contig output frame)
    scale: torch.Tensor,  # (N, M // 128) f32, mutated in place (t-contig output frame)
) -> None:
    M, N = x.shape
    rb = M // 128  # number of 128-row blocks
    # View the input as (rb, 128, N) so the middle axis is the 128-row reduction group and the
    # outer axis directly indexes the scale block. The output qdata is (N, M) = (N, rb, 128), so we
    # store the transpose of each computed block in-kernel (like the Triton kernel's tl.trans) --
    # this avoids a separate transpose pass in the wrapper.
    xv = x.view(rb, 128, N)
    qv = qdata.view(N, rb, 128)
    for tile_rb, tile_n in hl.tile([rb, N]):
        x_blk = xv[tile_rb, :, tile_n].to(torch.float32)  # (t_rb, 128, t_n)
        amax = torch.clamp(torch.amax(torch.abs(x_blk), dim=1), min=1e-12)  # (t_rb, t_n)
        s = amax / _FP8_MAX
        y = (x_blk * (1.0 / s)[:, None, :]).to(torch.float8_e4m3fn)  # (t_rb, 128, t_n)
        scale[tile_n, tile_rb] = s.permute(1, 0)  # (t_n, t_rb)
        qv[tile_n, tile_rb, :] = y.permute(2, 0, 1)  # (t_n, t_rb, 128)


def fp8_deepseek_1x128_dim_m_helion(x, **kwargs):
    """dim-M deepseek in Helion: reduce abs-max over each 128-row block (one fp32 scale per
    (128-row-block, column)), quantize to fp8, and write both outputs transposed to (N, M) /
    (N, M//128) to match the gold `deepseek_1x128_dim_m_f`. `**kwargs` (flex_tile_map tile origin
    / num_col) are accepted and ignored -- the kernel owns its own tiling."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0, f"deepseek dim_m requires M divisible by 128, got M={M}"
    qdata = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((N, M // 128), dtype=torch.float32, device=x.device)
    _deepseek_1x128_dim_m_kernel(x, qdata, scale)
    return qdata, scale


FP8_DEEPSEEK_1X128_DIM_M = QuantCastHelionRecipe.from_gold(
    Deepseek1x128DimMGold, helion_fn=fp8_deepseek_1x128_dim_m_helion
)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR dim-M: 32-row blocks down M, one e8m0 scale per (32-row-block, col); transposed
# outputs (N, M) / (N, M//32). Mirrors mxfp8_floor_dim_m_f -- same shape as the deepseek dim-M
# kernel above (32-row block instead of 128) but with an e8m0 power-of-two scale (bit-math) rather
# than an fp32 scale. autotune_effort="none" -> default config, no search (fast to iterate).
# ---------------------------------------------------------------------------
@helion.kernel(config=helion.Config(
    atomic_indexing=[], 
    block_sizes=[4, 32], 
    indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 
    l2_groupings=[4], 
    load_eviction_policies=['', 'first', '', 'first'], 
    loop_orders=[[1, 0]], 
    num_stages=4, 
    num_warps=1, 
    pid_type='flat', 
    range_flattens=[None], 
    range_multi_buffers=[None], 
    range_num_stages=[], 
    range_unroll_factors=[0], 
    range_warp_specializes=[None], 
    reduction_loops=[None]
), static_shapes=True, ignore_warnings=[helion.exc.TensorOperationInWrapper])
def _mxfp8_floor_dim_m_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    qdata: torch.Tensor,  # (N, M) fp8_e4m3fn, mutated in place (t-contig output frame)
    scale_u8: torch.Tensor,  # (N, M // 32) uint8 e8m0 bits, mutated in place (t-contig frame)
) -> None:
    M, N = x.shape
    rb = M // 32  # number of 32-row blocks
    xv = x.view(rb, 32, N)
    qv = qdata.view(N, rb, 32)
    for tile_rb, tile_n in hl.tile([rb, N]):
        x_blk = xv[tile_rb, :, tile_n].to(torch.float32)  # (t_rb, 32, t_n)
        amax = torch.amax(torch.abs(x_blk), dim=1)  # (t_rb, t_n); reduce down the 32 block rows
        biased = _amax_to_e8m0_floor_biased(amax)  # (t_rb, t_n) int32 e8m0 exponent
        sfp = _e8m0_biased_to_fp32(biased)  # (t_rb, t_n) fp32 pow2 factor
        y = (x_blk / sfp[:, None, :]).to(torch.float8_e4m3fn)  # (t_rb, 32, t_n)
        scale_u8[tile_n, tile_rb] = biased.to(torch.uint8).permute(1, 0)  # (t_n, t_rb)
        qv[tile_n, tile_rb, :] = y.permute(2, 0, 1)  # (t_n, t_rb, 32)


def mxfp8_floor_dim_m_helion(x, **kwargs):
    """dim-M mxfp8 floor in Helion: abs-max over each 32-row block, e8m0 FLOOR power-of-two scale
    per (32-row-block, column), quantize to fp8, and write both outputs transposed to (N, M) /
    (N, M//32) to match the gold `mxfp8_floor_dim_m_f`. `**kwargs` are accepted and ignored."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0, f"mxfp8_floor dim_m requires M divisible by 32, got M={M}"
    qdata = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_u8 = torch.empty((N, M // 32), dtype=torch.uint8, device=x.device)
    _mxfp8_floor_dim_m_kernel(x, qdata, scale_u8)
    return qdata, scale_u8.view(torch.float8_e8m0fnu)


MXFP8_FLOOR_DIM_M = QuantCastHelionRecipe.from_gold(
    Mxfp8FloorDimMGold, helion_fn=mxfp8_floor_dim_m_helion
)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR 32x32: one e8m0 scale per square 32x32 block; qdata keeps the input's (M, N) layout
# (no transpose), scale is (M//32, N//32). Mirrors mxfp8_32x32_floor_f. Viewing (M, N) as
# (M//32, 32, N//32, 32) makes each 32x32 block a (dim1, dim3) pair, so the block amax is a reduce
# over those two axes and the store lands back in place. autotune_effort="none" -> default config.
# ---------------------------------------------------------------------------
# block_sizes=[1, 1] pins one 32x32 block per program. This kernel views x as 4D
# (rb, 32, cb, 32) and tiles only [rb, cb] -- the two size-32 axes are untiled register dims. So
# block_sizes multiply the register tile by 32*32: the default heuristic's [16, 16] materializes a
# 16*32*16*32 = 1MB fp32 tile per program, which ptxas takes ~18s to compile cold. [1, 1] keeps it
# at one 32x32 block (4KB) -> ~0.5s cold compile. We don't autotune here (perf isn't the point for
# this recipe yet); this just makes the debug loop fast and deterministic. Raise block_sizes (or
# switch to autotune_effort="full") if/when we care about this kernel's bandwidth.
@helion.kernel(
    config=helion.Config(block_sizes=[1, 1], num_warps=4, num_stages=1),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _mxfp8_32x32_floor_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    qdata: torch.Tensor,  # (M, N) fp8_e4m3fn, mutated in place
    scale_u8: torch.Tensor,  # (M // 32, N // 32) uint8 e8m0 bits, mutated in place
) -> None:
    M, N = x.shape
    rb, cb = M // 32, N // 32  # 32x32 block grid
    xv = x.view(rb, 32, cb, 32)
    qv = qdata.view(rb, 32, cb, 32)
    for tile_rb, tile_cb in hl.tile([rb, cb]):
        x_blk = xv[tile_rb, :, tile_cb, :].to(torch.float32)  # (t_rb, 32, t_cb, 32)
        # block amax over both within-block axes (the two 32s): reduce the trailing 32, then the
        # leading 32.
        amax = torch.amax(torch.amax(torch.abs(x_blk), dim=3), dim=1)  # (t_rb, t_cb)
        biased = _amax_to_e8m0_floor_biased(amax)  # (t_rb, t_cb) int32 e8m0 exponent
        sfp = _e8m0_biased_to_fp32(biased)  # (t_rb, t_cb) fp32 pow2 factor
        y = (x_blk / sfp[:, None, :, None]).to(torch.float8_e4m3fn)  # (t_rb, 32, t_cb, 32)
        qv[tile_rb, :, tile_cb, :] = y
        scale_u8[tile_rb, tile_cb] = biased.to(torch.uint8)


def mxfp8_32x32_floor_helion(x, **kwargs):
    """mxfp8 floor with square 32x32 blocks in Helion: one e8m0 FLOOR power-of-two scale per 32x32
    block, quantize to fp8 in the input's (M, N) layout (no transpose). Matches the gold
    `mxfp8_32x32_floor_f`. `**kwargs` are accepted and ignored."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0 and N % 32 == 0, f"mxfp8_32x32 requires M,N divisible by 32, got {(M, N)}"
    qdata = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale_u8 = torch.empty((M // 32, N // 32), dtype=torch.uint8, device=x.device)
    _mxfp8_32x32_floor_kernel(x, qdata, scale_u8)
    return qdata, scale_u8.view(torch.float8_e8m0fnu)


MXFP8_32X32_FLOOR = QuantCastHelionRecipe.from_gold(
    Mxfp832x32FloorGold, helion_fn=mxfp8_32x32_floor_helion
)


# ---------------------------------------------------------------------------
# nvfp4 (two-level, no swizzle): a per-tensor fp32 OUTER scale (global amax, computed on the host
# and passed in as an aux) + a per-16-element e4m3 INNER scale, with fp4-packed qdata. Mirrors the
# gold `nvfp4_gs_f`. The fp4 (e2m1) cast+pack uses the HARDWARE path -- the same
# `cvt.rn.satfinite.e2m1x2.f32` PTX the gold takes under torch.compile, emitted here via
# `hl.inline_asm_elementwise` (Helion's own inline-asm HOP -- the gold's Inductor `inline_asm_...`
# is unreachable from a Helion kernel). One cvt turns two fp32 inputs into one packed byte (RNE,
# saturating): `$1` -> high nibble, `$2` -> low nibble (odd col -> high, even col -> low, matching
# `pack_uint4`). B200/sm100 only (that's what the benchmark asserts).
#   - The two inputs must be the even/odd columns as SEPARATE tiles, but strided slices
#     (`x[..., 0::2]`) hit Helion's InvalidIndexingType. So view each 16-block as (8, 2) and pull
#     the pair apart with masked sums over the size-2 axis: even = sum(blk*[1,0]), odd = sum(blk*[0,1]).
#   - The outer scale is a scalar (1,)-tensor arg, read with `hl.load(outer_scale, [0])` (the
#     scalar-scale idiom from helion's silu_mul_fp8).
# (A pure-bit-math encode + weighted-sum pack -- ported from quant_cast_gold.utils.f32_to_f4_unpacked
# -- also works and is bit-exact, if a non-sm100 fallback is ever needed; see git history.)
# ---------------------------------------------------------------------------
_F4_E2M1_MAX = 6.0
_F8E4M3_MAX = 448.0
_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny

# Hardware fp4 cast+pack: cvt two fp32 ($1=high nibble, $2=low nibble) to one packed e2m1x2 byte,
# widen to u16 for the =h output. Requires sm100+ (Blackwell). Matches the gold's asm exactly.
_NVFP4_CVT_ASM = "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $1, $2; cvt.u16.u8 $0, t; }"


@helion.kernel(
    # note: setting autotune_effort to full crashes with
    # https://gist.github.com/vkuzo/6f8a4beaebef60aa4e6e7059efbcfad8
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _nvfp4_gs_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    outer_scale: torch.Tensor,  # (1,) f32 per-tensor outer scale (global amax, host-computed)
    qdata: torch.Tensor,  # (M, N // 2) uint8, fp4-packed (two e2m1 per byte), mutated in place
    inner_scale: torch.Tensor,  # (M, N // 16) e4m3 inner block scale, mutated in place
) -> None:
    M, N = x.shape
    c16 = N // 16  # number of 16-element inner blocks per row (== inner-scale columns)
    # View each 16-block as (8, 2): the trailing 2 is the pack pair (even col -> low nibble, odd ->
    # high). qdata (M, N//2) == (M, c16*8) so the packed bytes land back in row-major order.
    xv = x.view(M, c16, 8, 2)
    qv = qdata.view(M, c16, 8)
    for tile_m, tile_c in hl.tile([M, c16]):
        outer = hl.load(outer_scale, [0])
        x_blk = xv[tile_m, tile_c, :, :].to(torch.float32)  # (bm, bc, 8, 2)
        amax = torch.amax(torch.amax(torch.abs(x_blk), dim=3), dim=2)  # (bm, bc) over the 16
        # inner e4m3 block scale relative to the outer scale; round-trip through e4m3 BEFORE using it
        # in the reciprocal (the gold does too -- the rounding is load-bearing for bit-exactness).
        inner_e4m3 = torch.clamp(
            (amax / _F4_E2M1_MAX) / outer, _E4M3_EPS, _F8E4M3_MAX
        ).to(torch.float8_e4m3fn)
        recip = (1.0 / outer) / inner_e4m3.to(torch.float32)  # (bm, bc)
        data_scaled = torch.clamp(
            x_blk * recip[:, :, None, None], -_F4_E2M1_MAX, _F4_E2M1_MAX
        )  # (bm, bc, 8, 2)
        # split the (8, 2) pair into even/odd tiles WITHOUT indexing (strided slices raise
        # InvalidIndexingType): mask-and-reduce the size-2 axis. w_odd = [0, 1] -> odd; 1-w_odd ->
        # even. The hardware cvt then packs (odd -> high nibble, even -> low) into one byte.
        w_odd = hl.arange(2).to(torch.float32)  # [0.0, 1.0]
        even = torch.sum(data_scaled * (1.0 - w_odd), dim=3)  # (bm, bc, 8) -> low nibble ($2)
        odd = torch.sum(data_scaled * w_odd, dim=3)  # (bm, bc, 8) -> high nibble ($1)
        packed_u16 = hl.inline_asm_elementwise(
            _NVFP4_CVT_ASM, "=h,r,r", [odd, even],
            dtype=torch.uint16, is_pure=True, pack=1,
        )
        qv[tile_m, tile_c, :] = packed_u16.to(torch.uint8)
        inner_scale[tile_m, tile_c] = inner_e4m3


def nvfp4_gs_helion(x, outer_scale, **kwargs):
    """nvfp4 two-level cast in Helion: per-16 e4m3 inner scale (relative to the host-computed
    per-tensor `outer_scale` aux), fp4-packed qdata (M, N//2) + inner scale (M, N//16) e4m3 in plain
    row-major layout (no swizzle), matching the gold `nvfp4_gs_f`. `**kwargs` are accepted and
    ignored (the kernel owns its tiling)."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert N % 16 == 0, f"nvfp4 requires N divisible by 16, got N={N}"
    qdata = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    inner_scale = torch.empty((M, N // 16), dtype=torch.float8_e4m3fn, device=x.device)
    _nvfp4_gs_kernel(x, outer_scale.reshape(1).to(torch.float32), qdata, inner_scale)
    return qdata.view(torch.float4_e2m1fn_x2), inner_scale


NVFP4 = QuantCastHelionRecipe.from_gold(Nvfp4GsGold, helion_fn=nvfp4_gs_helion)


# Order mirrors quant_cast_gold.ALL_RECIPES / quant_cast_triton.ALL_RECIPES (only the recipes with
# a Helion impl are listed; more will be added as they're ported).
ALL_RECIPES = [
    ("fp8_deepseek_1x128_dim_m", FP8_DEEPSEEK_1X128_DIM_M),
    ("mxfp8_floor_dim_m", MXFP8_FLOOR_DIM_M),
    ("mxfp8_32x32_floor", MXFP8_32X32_FLOOR),
    ("nvfp4", NVFP4),
]
