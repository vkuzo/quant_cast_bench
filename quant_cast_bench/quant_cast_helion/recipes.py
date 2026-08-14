"""Helion implementations of the quant_cast_gold recipes.

Each recipe is a `QuantCastHelionRecipe` -- it inherits the gold reference
(`pt_ref_fn`/`correctness_fn`/`example_input_fn`/`perf_description`) from a
`QuantCastSingleKernelGold` and adds `helion_fn`, a Helion-backed implementation of the same
cast. Mirrors quant_cast_triton's `QuantCastTritonRecipe` (inherit-from-gold + `from_gold`).
test/test_quant_cast_helion.py grades each `helion_fn` against its gold `pt_ref_fn`.

Recipes are ported over time; `fp8_deepseek_1x128_dim_m` and `mxfp8_dim_m` are done.
"""

from dataclasses import dataclass
from typing import Callable

import helion
import helion.language as hl
import torch

from quant_cast_bench.quant_cast_gold.recipes import (
    Deepseek1x128DimKmGold,
    Deepseek1x128DimMGold,
    Deepseek1x128Gold,
    Deepseek128x128Gold,
    Float8TensorwiseGold,
    HadamardRht,
    Mxfp832x32Gold,
    Mxfp8DimKmGold,
    Mxfp8DimKmSwizzleGold,
    Mxfp8DimMGold,
    Mxfp8DimMSwizzleGold,
    Mxfp8SwizzleGold,
    Nvfp4GsGold,
    Nvfp4GsSwizzleGold,
    QuantCastSingleKernelGold,
    SrF32ToBf16,
)

# fp8_e4m3fn representable max; the deepseek scale is amax / fp8_max (see the gold recipe).
_FP8_MAX = 448.0

# e8m0 (float8_e8m0fnu) power-of-two scale bit-math constant (see gold _amax_to_e8m0_rceil /
# _e8m0_to_fp32): the fp32 mantissa width, used to shift the exponent field for the RCEIL round-up
# and the inverse reconstruction.
_F32_MBITS = 23


def _amax_to_e8m0_biased(amax):
    """fp32 amax tile -> biased e8m0 exponent (int32, 0..255) via RCEIL round-up. Ports the gold
    `_amax_to_e8m0_rceil` bit-math: descale = amax / 448, then round the descale exponent UP
    whenever its fp32 mantissa is nonzero. The non-finite->255 branch is omitted since the
    benchmark/test inputs are finite (matches bit-for-bit on finite amax)."""
    bits = (amax * (1.0 / _FP8_MAX)).view(torch.int32)  # descale bits
    biased_exponent = (bits >> _F32_MBITS) & 0xFF
    mantissa = bits & 0x7FFFFF
    # normal fp32 rounds up on any set mantissa bit; fp32 subnormals (biased_exp == 0) only above 2^-127.
    needs_round_up = torch.where(biased_exponent == 0, mantissa > 0x400000, mantissa != 0)
    return biased_exponent + needs_round_up.to(torch.int32)  # biased exponent, int32


def _e8m0_biased_to_reciprocal_fp32(biased):
    """biased e8m0 exponent (int32 tile) -> fp32 reciprocal pow2 factor 2^(127-e). Ports the gold
    `_e8m0_scale_to_reciprocal_fp32` (reciprocal biased exponent = 254 - e): shift that into the
    fp32 exponent field. The cast multiplies data by this (torchao `_to_mx_rceil`)."""
    return ((254 - biased) << _F32_MBITS).view(torch.float32)


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
@helion.kernel(configs=[
    # for b200
    helion.Config(
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
    ),
    # for h100: same as b200 but drops range_warp_specializes, which the sm_90a config-spec
    # rejects for this kernel (expected 0 values); omitting it lets Helion fill the per-arch
    # default. Not perf-tuned for h100 -- just a config that compiles and runs there.
    helion.Config(
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
        reduction_loops=[None]
    ),
], static_shapes=True)
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
# mxfp8 dim-M: 32-row blocks down M, one e8m0 scale per (32-row-block, col); transposed
# outputs (N, M) / (N, M//32). Mirrors mxfp8_dim_m_f -- same shape as the deepseek dim-M
# kernel above (32-row block instead of 128) but with an e8m0 power-of-two scale (bit-math) rather
# than an fp32 scale. autotune_effort="none" -> default config, no search (fast to iterate).
# ---------------------------------------------------------------------------
@helion.kernel(configs=[
    # for b200
    helion.Config(
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
    ),
    # for h100: same as b200 but drops range_warp_specializes, which the sm_90a config-spec
    # rejects for this kernel (expected 0 values); omitting it lets Helion fill the per-arch
    # default. Not perf-tuned for h100 -- just a config that compiles and runs there.
    helion.Config(
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
        reduction_loops=[None]
    ),
], static_shapes=True, ignore_warnings=[helion.exc.TensorOperationInWrapper])
def _mxfp8_dim_m_kernel(
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
        biased = _amax_to_e8m0_biased(amax)  # (t_rb, t_n) int32 e8m0 exponent
        rcp = _e8m0_biased_to_reciprocal_fp32(biased)  # (t_rb, t_n) fp32 reciprocal pow2 factor
        y = (x_blk * rcp[:, None, :]).to(torch.float8_e4m3fn)  # (t_rb, 32, t_n)
        scale_u8[tile_n, tile_rb] = biased.to(torch.uint8).permute(1, 0)  # (t_n, t_rb)
        qv[tile_n, tile_rb, :] = y.permute(2, 0, 1)  # (t_n, t_rb, 32)


def mxfp8_dim_m_helion(x, **kwargs):
    """dim-M mxfp8 in Helion: abs-max over each 32-row block, e8m0 power-of-two scale
    per (32-row-block, column), quantize to fp8, and write both outputs transposed to (N, M) /
    (N, M//32) to match the gold `mxfp8_dim_m_f`. `**kwargs` are accepted and ignored."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0, f"mxfp8 dim_m requires M divisible by 32, got M={M}"
    qdata = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_u8 = torch.empty((N, M // 32), dtype=torch.uint8, device=x.device)
    _mxfp8_dim_m_kernel(x, qdata, scale_u8)
    return qdata, scale_u8.view(torch.float8_e8m0fnu)


MXFP8_DIM_M = QuantCastHelionRecipe.from_gold(
    Mxfp8DimMGold, helion_fn=mxfp8_dim_m_helion
)


# ---------------------------------------------------------------------------
# mxfp8 dim-M, with the e8m0 scale in the NVIDIA 32x4x4 SWIZZLED block grid. Same 32-row-down-M
# reduction + transposed (N, M) qdata as `_mxfp8_dim_m_kernel`, but the scale is scattered
# straight into the swizzled block layout in-kernel (not stored plain then swizzled in a wrapper
# post-pass). Mirrors the gold `mxfp8_dim_m_swizzle_f` (= mxfp8_dim_m_f then
# _to_blocked_4d on the (N, M//32) scale).
#
# Swizzle math (from the gold `_to_blocked_4d` applied to the transposed (N, M//32) scale, index
# [n, rb]): the block grid is (nrb=N//128, ncb=M//128, 32, 16) and
#     blocked[n//128, rb//4, n%32, ((n//128-local)//32)*4 + rb%4] = scale[n, rb].
# So we tile over the BLOCK-COUNT dims [ncb, nrb] -- then the two tile indices ARE the block ordinals
# (Cb, Nb), and the swizzled store is a plain 5D index. This pushes the within-block 32/128 axes into
# register dims (5D tiles), so a program's register tile is block_ncb * block_nrb * (4*32*128) fp32.
# The final (a=4, c4=4) -> 16 nibble merge is a free contiguous reshape in the wrapper (kept out of
# the kernel so the store is a pure permute, no in-kernel reshape-across-a-tiled-axis).
#   x M-axis is viewed as (ncb, c4=4, w32=32): m = 128*Cb + 32*c4 + w32; rb = m//32 = 4*Cb + c4.
#   x N-axis is viewed as (nrb, n128=128): n = 128*Nb + n128; the reduction is over w32 (32 M-rows).
#   n128 splits into (a = n128//32, b = n128%32); blocked atom = [Nb, Cb, b, a, c4].
#
# Config: we can't use autotune_effort="none" here -- the default block-size heuristic scales the
# tiled [ncb, nrb] block sizes up (~1024 total), and with the 4*32*128 = 16384-element untiled
# register dims that makes the x load's tile 16384*1024 = 16.7M elements, past triton's hard
# per-tensor cap of 1,048,576. Autotune-none runs at the test shape (512x512, tiny block counts) but
# OVERFLOWS at the 16384x16384 benchmark shape. So pin block_sizes=[1,1]: one 128x128 x-block per
# program (a 1*4*32*1*128 = 16384-element / 64KB fp32 tile, same magnitude as the other dim-M
# kernels), well under the cap and fast to compile. Raise the block sizes (or autotune) later if this
# kernel's bandwidth becomes the priority.
# ---------------------------------------------------------------------------
@helion.kernel(
    config=helion.Config(block_sizes=[1, 1], num_warps=4, num_stages=1),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _mxfp8_dim_m_swizzle_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    qdata: torch.Tensor,  # (N, M) fp8_e4m3fn, mutated in place (t-contig output frame)
    scale5: torch.Tensor,  # (nrb, ncb, 32, 4, 4) uint8 e8m0 bits, mutated (swizzled, pre (4,4)->16)
) -> None:
    M, N = x.shape
    ncb = M // 128  # M col-blocks: each = 4 rb-of-32 rows = 128 M-rows
    nrb = N // 128  # N row-blocks: each = 128 N-cols
    xv = x.view(ncb, 4, 32, nrb, 128)      # M=(Cb, c4, w32), N=(Nb, n128)
    qv = qdata.view(nrb, 128, ncb, 4, 32)  # (N, M) transposed frame = (Nb, n128, Cb, c4, w32)
    for tile_cb, tile_nb in hl.tile([ncb, nrb]):
        x_blk = xv[tile_cb, :, :, tile_nb, :].to(torch.float32)  # (tcb,4,32,tnb,128)
        amax = torch.amax(torch.abs(x_blk), dim=2)  # (tcb,4,tnb,128); reduce the 32 M-rows (w32)
        biased = _amax_to_e8m0_biased(amax)  # (tcb,4,tnb,128) int32 e8m0 exponent
        rcp = _e8m0_biased_to_reciprocal_fp32(biased)  # (tcb,4,tnb,128) fp32 reciprocal pow2 factor
        y = (x_blk * rcp[:, :, None, :, :]).to(torch.float8_e4m3fn)  # (tcb,4,32,tnb,128)
        qv[tile_nb, :, tile_cb, :, :] = y.permute(3, 4, 0, 1, 2)  # -> (tnb,128,tcb,4,32)
        # swizzle: split n128 -> (a, b); reorder [Cb,c4,Nb,a,b] -> blocked atom [Nb,Cb,b,a,c4].
        biased_r = biased.reshape(biased.shape[0], biased.shape[1], biased.shape[2], 4, 32)
        atom = biased_r.permute(2, 0, 4, 3, 1)  # (tnb,tcb,32,4,4) = [Nb,Cb,b,a,c4]
        scale5[tile_nb, tile_cb, :, :, :] = atom.to(torch.uint8)


def mxfp8_dim_m_swizzle_helion(x, **kwargs):
    """dim-M mxfp8 in Helion with the e8m0 scale in the NVIDIA 32x4x4 swizzled block grid:
    abs-max over each 32-row block down M, e8m0 power-of-two scale, quantize to fp8, and write
    the transposed (N, M) qdata plus the scale scattered directly into the swizzled block grid
    (nrb, ncb, 32, 16). Matches the gold `mxfp8_dim_m_swizzle_f`. The (4,4)->16 nibble merge is
    a free contiguous reshape here (the kernel emits (nrb, ncb, 32, 4, 4)). `**kwargs` are accepted
    and ignored."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0, (
        f"mxfp8 dim_m swizzle requires M,N divisible by 128, got {(M, N)}"
    )
    nrb, ncb = N // 128, M // 128
    qdata = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale5 = torch.empty((nrb, ncb, 32, 4, 4), dtype=torch.uint8, device=x.device)
    _mxfp8_dim_m_swizzle_kernel(x, qdata, scale5)
    scale = scale5.reshape(nrb, ncb, 32, 16).view(torch.float8_e8m0fnu)
    return qdata, scale


MXFP8_DIM_M_SWIZZLE = QuantCastHelionRecipe.from_gold(
    Mxfp8DimMSwizzleGold, helion_fn=mxfp8_dim_m_swizzle_helion
)


# ---------------------------------------------------------------------------
# mxfp8 32x32: one e8m0 scale per square 32x32 block; qdata keeps the input's (M, N) layout
# (no transpose), scale is (M//32, N//32). Mirrors mxfp8_32x32_f. Viewing (M, N) as
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
def _mxfp8_32x32_kernel(
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
        biased = _amax_to_e8m0_biased(amax)  # (t_rb, t_cb) int32 e8m0 exponent
        rcp = _e8m0_biased_to_reciprocal_fp32(biased)  # (t_rb, t_cb) fp32 reciprocal pow2 factor
        y = (x_blk * rcp[:, None, :, None]).to(torch.float8_e4m3fn)  # (t_rb, 32, t_cb, 32)
        qv[tile_rb, :, tile_cb, :] = y
        scale_u8[tile_rb, tile_cb] = biased.to(torch.uint8)


def mxfp8_32x32_helion(x, **kwargs):
    """mxfp8 with square 32x32 blocks in Helion: one e8m0 power-of-two scale per 32x32
    block, quantize to fp8 in the input's (M, N) layout (no transpose). Matches the gold
    `mxfp8_32x32_f`. `**kwargs` are accepted and ignored."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0 and N % 32 == 0, f"mxfp8_32x32 requires M,N divisible by 32, got {(M, N)}"
    qdata = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale_u8 = torch.empty((M // 32, N // 32), dtype=torch.uint8, device=x.device)
    _mxfp8_32x32_kernel(x, qdata, scale_u8)
    return qdata, scale_u8.view(torch.float8_e8m0fnu)


MXFP8_32X32 = QuantCastHelionRecipe.from_gold(
    Mxfp832x32Gold, helion_fn=mxfp8_32x32_helion
)


# ---------------------------------------------------------------------------
# deepseek fp8 128x128: one fp32 scale per square 128x128 block; qdata keeps the input's (M, N)
# layout (no transpose), scale is (M//128, N//128). The deepseek analog of `_mxfp8_32x32_kernel`
# -- same square-block view (rb, 128, cb, 128) + in-place store, but a 128x128 block with an fp32
# amax/448 (clamped to 1e-12) reciprocal scale instead of a 32x32 block with e8m0 bit-math. Mirrors
# deepseek_128x128_f. block_sizes=[1, 1] pins one 128x128 block per program (untiled 128*128 = 16384
# / 64KB fp32 register tile); a larger tiled block multiplies that by block^2 and overflows triton's
# 1,048,576 per-tensor cap, same as the other square/dim-km kernels.
# ---------------------------------------------------------------------------
@helion.kernel(
    config=helion.Config(block_sizes=[1, 1], num_warps=4, num_stages=1),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _deepseek_128x128_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    qdata: torch.Tensor,  # (M, N) fp8_e4m3fn, mutated in place
    scale: torch.Tensor,  # (M // 128, N // 128) f32, mutated in place
) -> None:
    M, N = x.shape
    rb, cb = M // 128, N // 128  # 128x128 block grid
    xv = x.view(rb, 128, cb, 128)
    qv = qdata.view(rb, 128, cb, 128)
    for tile_rb, tile_cb in hl.tile([rb, cb]):
        x_blk = xv[tile_rb, :, tile_cb, :].to(torch.float32)  # (t_rb, 128, t_cb, 128)
        # block amax over both within-block axes (the two 128s): reduce the trailing 128, then the
        # leading 128.
        amax = torch.amax(torch.amax(torch.abs(x_blk), dim=3), dim=1)  # (t_rb, t_cb)
        s = torch.clamp(amax, min=1e-12) / _FP8_MAX
        y = (x_blk * (1.0 / s)[:, None, :, None]).to(torch.float8_e4m3fn)  # (t_rb, 128, t_cb, 128)
        qv[tile_rb, :, tile_cb, :] = y
        scale[tile_rb, tile_cb] = s


def fp8_deepseek_128x128_helion(x, **kwargs):
    """deepseek fp8 with square 128x128 blocks in Helion: one fp32 amax/448 scale per 128x128 block,
    quantize to fp8 in the input's (M, N) layout (no transpose). Matches the gold
    `deepseek_128x128_f`. `**kwargs` are accepted and ignored."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0, (
        f"deepseek_128x128 requires M,N divisible by 128, got {(M, N)}"
    )
    qdata = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((M // 128, N // 128), dtype=torch.float32, device=x.device)
    _deepseek_128x128_kernel(x, qdata, scale)
    return qdata, scale


FP8_DEEPSEEK_128X128 = QuantCastHelionRecipe.from_gold(
    Deepseek128x128Gold, helion_fn=fp8_deepseek_128x128_helion
)


# ---------------------------------------------------------------------------
# mxfp8 in BOTH directions in ONE pass (dim-km): read x once and emit FOUR outputs --
# dim-K (1x32 blocks along columns, non-transposed) and dim-M (32x1 blocks down rows, transposed).
# Mirrors the gold `mxfp8_dim_km_f`: dim-K matches mxfp8_f, dim-M matches
# mxfp8_dim_m_f. Both reductions come off ONE loaded 32x32 block, so we reuse the 32x32
# block-grid view (rb, 32, cb, 32) and tile [rb, cb] like `_mxfp8_32x32_kernel`:
#   x[m, n] with m = 32*rb + r32, n = 32*cb + c32; a tile block is [rb, r32, cb, c32].
#   dim-K: reduce c32 (axis 3) -> one e8m0 scale per (m, col-block cb); qk stays in (M, N).
#   dim-M: reduce r32 (axis 1) -> one e8m0 scale per (row-block rb, n); qm/sm stored transposed.
# block_sizes=[1, 1] pins one 32x32 block per program (same rationale as the 32x32 kernel: the two
# size-32 axes are untiled register dims, so a larger tiled block size multiplies the register tile
# by 32*32 and slows compilation). We do two reductions + two quantizations off the one block.
#
# autotune_effort="none" is NOT usable here (tested empirically): its default block-size heuristic
# scales [rb, cb] up, so the register tile becomes block_rb*32*block_cb*32 -- the same *1024 blowup
# that already makes the 32x32 kernel's default [16,16] a 1MB / ~18s-to-compile tile, but WORSE here
# because we emit two reductions + four stores off the one block. At 512x512 the autotune-none
# compile did not finish in 10 minutes; at the 16384x16384 benchmark shape it would additionally
# overflow triton's hard 1,048,576 per-tensor numel cap (block^2 * 32 * 32 exceeds it once the block
# dim passes 32). Pinning [1, 1] keeps the tile at one 32x32 block -> fast compile, no overflow.
# ---------------------------------------------------------------------------
@helion.kernel(
    config=helion.Config(block_sizes=[1, 1], num_warps=4, num_stages=1),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _mxfp8_dim_km_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    qk: torch.Tensor,  # (M, N) fp8_e4m3fn dim-K qdata, mutated in place (natural frame)
    sk_u8: torch.Tensor,  # (M, N // 32) uint8 e8m0 dim-K scale bits, mutated in place
    qm: torch.Tensor,  # (N, M) fp8_e4m3fn dim-M qdata, mutated in place (t-contig frame)
    sm_u8: torch.Tensor,  # (N, M // 32) uint8 e8m0 dim-M scale bits, mutated in place (t-contig)
) -> None:
    M, N = x.shape
    rb, cb = M // 32, N // 32  # 32x32 block grid
    xv = x.view(rb, 32, cb, 32)      # [rb, r32, cb, c32]
    qkv = qk.view(rb, 32, cb, 32)    # natural (M, N) = [rb, r32, cb, c32]
    skv = sk_u8.view(rb, 32, cb)     # (M, N//32) = [rb, r32, cb]
    qmv = qm.view(cb, 32, rb, 32)    # transposed (N, M) = [cb, c32, rb, r32]
    smv = sm_u8.view(cb, 32, rb)     # transposed (N, M//32) = [cb, c32, rb]
    for tile_rb, tile_cb in hl.tile([rb, cb]):
        x_blk = xv[tile_rb, :, tile_cb, :].to(torch.float32)  # (t_rb, 32, t_cb, 32)
        abs_blk = torch.abs(x_blk)
        # dim-K: reduce the trailing 32 (the block's columns) -> scale per (row m, col-block cb).
        amax_k = torch.amax(abs_blk, dim=3)  # (t_rb, 32, t_cb) = [rb, r32, cb]
        biased_k = _amax_to_e8m0_biased(amax_k)  # int32 e8m0 exponent
        rcp_k = _e8m0_biased_to_reciprocal_fp32(biased_k)  # fp32 reciprocal pow2 factor
        y_k = (x_blk * rcp_k[:, :, :, None]).to(torch.float8_e4m3fn)  # (t_rb, 32, t_cb, 32)
        qkv[tile_rb, :, tile_cb, :] = y_k
        skv[tile_rb, :, tile_cb] = biased_k.to(torch.uint8)
        # dim-M: reduce the leading 32 (the block's rows) -> scale per (row-block rb, col n).
        amax_m = torch.amax(abs_blk, dim=1)  # (t_rb, t_cb, 32) = [rb, cb, c32]
        biased_m = _amax_to_e8m0_biased(amax_m)  # int32 e8m0 exponent
        rcp_m = _e8m0_biased_to_reciprocal_fp32(biased_m)  # fp32 reciprocal pow2 factor
        y_m = (x_blk * rcp_m[:, None, :, :]).to(torch.float8_e4m3fn)  # (t_rb, 32, t_cb, 32)
        # store transposed: (N, M) frame is [cb, c32, rb, r32]; permute the block accordingly.
        qmv[tile_cb, :, tile_rb, :] = y_m.permute(2, 3, 0, 1)  # -> (t_cb, 32, t_rb, 32)
        smv[tile_cb, :, tile_rb] = biased_m.to(torch.uint8).permute(1, 2, 0)  # -> (t_cb, 32, t_rb)


def mxfp8_dim_km_helion(x, **kwargs):
    """One-pass dim-km mxfp8 in Helion: read x once and emit both the dim-K quantization
    (1x32 blocks along columns, non-transposed (M, N)/(M, N//32)) and the dim-M quantization (32x1
    blocks down rows, transposed (N, M)/(N, M//32)) -- four outputs matching the gold
    `mxfp8_dim_km_f`. `**kwargs` are accepted and ignored (the kernel owns its tiling)."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0 and N % 32 == 0, (
        f"mxfp8 dim_km requires M,N divisible by 32, got {(M, N)}"
    )
    qk = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    sk_u8 = torch.empty((M, N // 32), dtype=torch.uint8, device=x.device)
    qm = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    sm_u8 = torch.empty((N, M // 32), dtype=torch.uint8, device=x.device)
    _mxfp8_dim_km_kernel(x, qk, sk_u8, qm, sm_u8)
    return (
        qk,
        sk_u8.view(torch.float8_e8m0fnu),
        qm,
        sm_u8.view(torch.float8_e8m0fnu),
    )


MXFP8_DIM_KM = QuantCastHelionRecipe.from_gold(
    Mxfp8DimKmGold, helion_fn=mxfp8_dim_km_helion
)


# ---------------------------------------------------------------------------
# mxfp8 both directions in ONE pass, with BOTH e8m0 scales in the NVIDIA 32x4x4 SWIZZLED block
# grid (dim-km + swizzle). Same four-quantity contract as `_mxfp8_dim_km_kernel` -- qk/qm
# qdata are byte-identical -- but each scale is scattered straight into its swizzled block grid
# in-kernel (not stored plain then swizzled in a wrapper). Mirrors `mxfp8_dim_km_swizzle_f`
# (= mxfp8_dim_km_f then _to_blocked_4d on each of sk (M,N//32) and sm (N,M//32)).
#
# To make BOTH swizzled stores plain 5D indices we tile over the 128x128 block grid [ncb, nrb] (so
# the tile indices are block ordinals) and view x with BOTH within-128 axes fully split:
#   M-rows: m = 128*Cb + 32*c4 + w32   (Cb in [0,ncb), c4 in [0,4), w32 in [0,32));  rb = m//32 = 4*Cb+c4
#   N-cols: n = 128*Nb + 32*a  + b     (Nb in [0,nrb), a  in [0,4), b   in [0,32));  kb = n//32 = 4*Nb+a
# so x_blk is 6D [Cb,c4,w32,Nb,a,b]. dim-K reduces b (the within-32-col index); dim-M reduces w32
# (the within-32-row index). The swizzle of a (H,W) scale is
#   blocked[H//128, W//4, h%32, ((h%128)//32)*4 + w%4] = scale[h, w],
# which with these already-split axes is a pure permute of the reduced scale (no in-kernel reshape
# across a tiled axis, unlike the dim-M-only swizzle kernel):
#   sk (M, N//32) [m, kb]: blocked_k[Cb, Nb, w32, c4, a]  (H=m -> RB=Cb, a'=c4, b'=w32; W=kb -> CB=Nb, c'=a)
#   sm (N, M//32) [n, rb]: blocked_m[Nb, Cb, b,   a,  c4] (H=n -> RB=Nb, a'=a,  b'=b;   W=rb -> CB=Cb, c'=c4)
# The (4,4)->16 nibble merge on each grid is a free contiguous reshape in the wrapper.
# block_sizes=[1, 1] pins one 128x128 x-block per program (untiled register dims 4*32*4*32 = 16384,
# same magnitude as the other swizzle/dim-km kernels). autotune_effort="none" is NOT usable here
# (checked empirically): its default block-size heuristic scales the tiled [ncb, nrb] block sizes up,
# so the register tile (block_ncb*block_nrb*16384) blows up and the 512x512 compile did not finish in
# 60s -- same failure as `_mxfp8_dim_km_kernel` / `_mxfp8_dim_m_swizzle_kernel` (and at the
# 16384x16384 shape it would also overflow triton's 1,048,576 per-tensor numel cap).
# ---------------------------------------------------------------------------
@helion.kernel(
    config=helion.Config(block_sizes=[1, 1], num_warps=4, num_stages=1),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _mxfp8_dim_km_swizzle_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    qk: torch.Tensor,  # (M, N) fp8_e4m3fn dim-K qdata, mutated in place (natural frame)
    scale5k: torch.Tensor,  # (ncb, nrb, 32, 4, 4) uint8 e8m0 dim-K scale, swizzled, pre (4,4)->16
    qm: torch.Tensor,  # (N, M) fp8_e4m3fn dim-M qdata, mutated in place (t-contig frame)
    scale5m: torch.Tensor,  # (nrb, ncb, 32, 4, 4) uint8 e8m0 dim-M scale, swizzled, pre (4,4)->16
) -> None:
    M, N = x.shape
    ncb = M // 128  # M 128-row blocks (each = 4 rb-of-32 rows)
    nrb = N // 128  # N 128-col blocks
    xv = x.view(ncb, 4, 32, nrb, 4, 32)    # [Cb, c4, w32, Nb, a, b]
    qkv = qk.view(ncb, 4, 32, nrb, 4, 32)  # natural (M, N), same axes as xv
    qmv = qm.view(nrb, 4, 32, ncb, 4, 32)  # transposed (N, M) = [Nb, a, b, Cb, c4, w32]
    for tile_cb, tile_nb in hl.tile([ncb, nrb]):
        x_blk = xv[tile_cb, :, :, tile_nb, :, :].to(torch.float32)  # (tcb,4,32,tnb,4,32)
        abs_blk = torch.abs(x_blk)
        # dim-K: reduce b (the within-32-col index) -> scale per (row m, col-block kb).
        amax_k = torch.amax(abs_blk, dim=5)  # (tcb,4,32,tnb,4) = [Cb,c4,w32,Nb,a]
        biased_k = _amax_to_e8m0_biased(amax_k)
        rcp_k = _e8m0_biased_to_reciprocal_fp32(biased_k)
        y_k = (x_blk * rcp_k[:, :, :, :, :, None]).to(torch.float8_e4m3fn)  # broadcast over b
        qkv[tile_cb, :, :, tile_nb, :, :] = y_k
        # swizzled dim-K scale: [Cb,c4,w32,Nb,a] -> blocked_k [Cb,Nb,w32,c4,a]
        scale5k[tile_cb, tile_nb, :, :, :] = biased_k.permute(0, 3, 2, 1, 4).to(torch.uint8)
        # dim-M: reduce w32 (the within-32-row index) -> scale per (row-block rb, col n).
        amax_m = torch.amax(abs_blk, dim=2)  # (tcb,4,tnb,4,32) = [Cb,c4,Nb,a,b]
        biased_m = _amax_to_e8m0_biased(amax_m)
        rcp_m = _e8m0_biased_to_reciprocal_fp32(biased_m)
        y_m = (x_blk * rcp_m[:, :, None, :, :, :]).to(torch.float8_e4m3fn)  # broadcast over w32
        qmv[tile_nb, :, :, tile_cb, :, :] = y_m.permute(3, 4, 5, 0, 1, 2)  # -> [Nb,a,b,Cb,c4,w32]
        # swizzled dim-M scale: [Cb,c4,Nb,a,b] -> blocked_m [Nb,Cb,b,a,c4]
        scale5m[tile_nb, tile_cb, :, :, :] = biased_m.permute(2, 0, 4, 3, 1).to(torch.uint8)


def mxfp8_dim_km_swizzle_helion(x, **kwargs):
    """One-pass dim-km mxfp8 in Helion with BOTH e8m0 scales in the NVIDIA 32x4x4 swizzled
    block grid: read x once and emit the dim-K quantization (1x32 blocks along columns, non-transposed
    (M, N) qdata + swizzled scale) and the dim-M quantization (32x1 blocks down rows, transposed (N, M)
    qdata + swizzled scale) -- four outputs matching the gold `mxfp8_dim_km_swizzle_f`. Each
    (4,4)->16 nibble merge is a free contiguous reshape here. `**kwargs` are accepted and ignored."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0, (
        f"mxfp8 dim_km swizzle requires M,N divisible by 128, got {(M, N)}"
    )
    ncb, nrb = M // 128, N // 128
    qk = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale5k = torch.empty((ncb, nrb, 32, 4, 4), dtype=torch.uint8, device=x.device)
    qm = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale5m = torch.empty((nrb, ncb, 32, 4, 4), dtype=torch.uint8, device=x.device)
    _mxfp8_dim_km_swizzle_kernel(x, qk, scale5k, qm, scale5m)
    sk = scale5k.reshape(ncb, nrb, 32, 16).view(torch.float8_e8m0fnu)
    sm = scale5m.reshape(nrb, ncb, 32, 16).view(torch.float8_e8m0fnu)
    return qk, sk, qm, sm


MXFP8_DIM_KM_SWIZZLE = QuantCastHelionRecipe.from_gold(
    Mxfp8DimKmSwizzleGold, helion_fn=mxfp8_dim_km_swizzle_helion
)


# ---------------------------------------------------------------------------
# deepseek fp8 1x128 in BOTH directions in ONE pass (dim-km): read x once and emit FOUR outputs --
# dim-K (1x128 blocks along columns, non-transposed) and dim-M (128x1 blocks down rows, transposed).
# The deepseek analog of `_mxfp8_dim_km_kernel`: same 128x128 block-grid structure and both
# reductions off one loaded block, but the scale is an fp32 amax/448 (clamped to 1e-12) and the
# quantize is a reciprocal multiply (matches deepseek_1x128_dim_km_f / the dim-K/dim-M deepseek
# kernels above), not e8m0 bit-math.
#   x[m, n] with m = 128*rb + r128, n = 128*cb + c128; a tile block is [rb, r128, cb, c128].
#   dim-K: reduce c128 (axis 3) -> one fp32 scale per (m, col-block cb); qk stays in (M, N).
#   dim-M: reduce r128 (axis 1) -> one fp32 scale per (row-block rb, n); qm/sm stored transposed.
# block_sizes=[1, 1] pins one 128x128 block per program: the two size-128 axes are untiled register
# dims (16384-element / 64KB fp32 tile), and autotune-none's default heuristic would scale [rb, cb]
# up and blow that past triton's 1,048,576 per-tensor cap -- same rationale as the other dim-km /
# swizzle kernels.
# ---------------------------------------------------------------------------
@helion.kernel(
    config=helion.Config(block_sizes=[1, 1], num_warps=4, num_stages=1),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _deepseek_1x128_dim_km_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    qk: torch.Tensor,  # (M, N) fp8_e4m3fn dim-K qdata, mutated in place (natural frame)
    sk: torch.Tensor,  # (M, N // 128) f32 dim-K scale, mutated in place
    qm: torch.Tensor,  # (N, M) fp8_e4m3fn dim-M qdata, mutated in place (t-contig frame)
    sm: torch.Tensor,  # (N, M // 128) f32 dim-M scale, mutated in place (t-contig frame)
) -> None:
    M, N = x.shape
    rb, cb = M // 128, N // 128  # 128x128 block grid
    xv = x.view(rb, 128, cb, 128)    # [rb, r128, cb, c128]
    qkv = qk.view(rb, 128, cb, 128)  # natural (M, N) = [rb, r128, cb, c128]
    skv = sk.view(rb, 128, cb)       # (M, N//128) = [rb, r128, cb]
    qmv = qm.view(cb, 128, rb, 128)  # transposed (N, M) = [cb, c128, rb, r128]
    smv = sm.view(cb, 128, rb)       # transposed (N, M//128) = [cb, c128, rb]
    for tile_rb, tile_cb in hl.tile([rb, cb]):
        x_blk = xv[tile_rb, :, tile_cb, :].to(torch.float32)  # (t_rb, 128, t_cb, 128)
        abs_blk = torch.abs(x_blk)
        # dim-K: reduce the trailing 128 (the block's columns) -> scale per (row m, col-block cb).
        amax_k = torch.amax(abs_blk, dim=3)  # (t_rb, 128, t_cb) = [rb, r128, cb]
        s_k = torch.clamp(amax_k, min=1e-12) / _FP8_MAX
        y_k = (x_blk * (1.0 / s_k)[:, :, :, None]).to(torch.float8_e4m3fn)  # (t_rb,128,t_cb,128)
        qkv[tile_rb, :, tile_cb, :] = y_k
        skv[tile_rb, :, tile_cb] = s_k
        # dim-M: reduce the leading 128 (the block's rows) -> scale per (row-block rb, col n).
        amax_m = torch.amax(abs_blk, dim=1)  # (t_rb, t_cb, 128) = [rb, cb, c128]
        s_m = torch.clamp(amax_m, min=1e-12) / _FP8_MAX
        y_m = (x_blk * (1.0 / s_m)[:, None, :, :]).to(torch.float8_e4m3fn)  # (t_rb,128,t_cb,128)
        # store transposed: (N, M) frame is [cb, c128, rb, r128]; permute the block accordingly.
        qmv[tile_cb, :, tile_rb, :] = y_m.permute(2, 3, 0, 1)  # -> (t_cb, 128, t_rb, 128)
        smv[tile_cb, :, tile_rb] = s_m.permute(1, 2, 0)  # -> (t_cb, t_rb) ... [cb, c128, rb]


def fp8_deepseek_1x128_dim_km_helion(x, **kwargs):
    """One-pass dim-km deepseek fp8 in Helion: read x once and emit both the dim-K quantization
    (1x128 blocks along columns, non-transposed (M, N)/(M, N//128)) and the dim-M quantization (128x1
    blocks down rows, transposed (N, M)/(N, M//128)) -- four outputs matching the gold
    `deepseek_1x128_dim_km_f`. `**kwargs` are accepted and ignored (the kernel owns its tiling)."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0, (
        f"deepseek dim_km requires M,N divisible by 128, got {(M, N)}"
    )
    qk = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    sk = torch.empty((M, N // 128), dtype=torch.float32, device=x.device)
    qm = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    sm = torch.empty((N, M // 128), dtype=torch.float32, device=x.device)
    _deepseek_1x128_dim_km_kernel(x, qk, sk, qm, sm)
    return qk, sk, qm, sm


FP8_DEEPSEEK_1X128_DIM_KM = QuantCastHelionRecipe.from_gold(
    Deepseek1x128DimKmGold, helion_fn=fp8_deepseek_1x128_dim_km_helion
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


# ---------------------------------------------------------------------------
# nvfp4 (two-level) with the e4m3 INNER scale in the NVIDIA 32x4x4 SWIZZLED block grid. Same two-level
# cast + fp4-packed qdata as `_nvfp4_gs_kernel` (qdata is byte-identical) -- only the inner scale
# layout differs: it's scattered straight into the swizzled block grid in-kernel instead of stored
# plain (M, N//16). Mirrors `nvfp4_gs_swizzle_f` (= nvfp4_gs_f then _to_blocked_4d on the inner scale).
#
# Like the mxfp8 swizzle kernels, to make the swizzled store a plain index we tile over the
# block-count dims [nrb, ncb] and view x with both the M-rows and the 16-block index fully split:
#   M-rows:   m = 128*RB + 32*a + b       (RB in [0,nrb), a in [0,4), b in [0,32))
#   N-cols:   n = 64*CB + 16*c + 2*j + k  (CB in [0,ncb), c in [0,4)); each 16-block is (j=8, k=2)
# so the 16-block index is w = n//16 = 4*CB + c and x_blk is 7D [RB,a,b,CB,c,j,k]. The nvfp4 cast runs
# per 16-block (reduce (j,k)); the fp4 pack is the same hardware cvt (even k=0 -> low nibble, odd k=1
# -> high). The swizzle of the inner scale (M, N//16) [m, w] is
#   blocked[m//128, w//4, m%32, ((m%128)//32)*4 + w%4] = inner[m, w],
# a pure permute of the reduced inner scale [RB,a,b,CB,c] -> blocked [RB,CB,b,a,c] (the (4,4)->16 merge
# is a free reshape in the wrapper). block_sizes=[1, 1] pins one 128x64 x-block per program (untiled
# register dims 4*32*4*8*2 = 8192).
#
# Autotuning does NOT work here (inferred, not tried), for two independent reasons:
#   1. inline-asm autotuner crash: like `_nvfp4_gs_kernel`, this uses `hl.inline_asm_elementwise`
#      (the _NVFP4_CVT_ASM cvt), and autotune_effort="full" crashes on that HOP -- see the note on the
#      non-swizzle nvfp4 kernel above (gist linked there). That's why nvfp4 is pinned to "none".
#   2. swizzle tile-shape overflow: even without the asm crash, the tile-over-block-count structure
#      means any non-tiny block size makes the register tile block_nrb*block_ncb*8192, which overflows
#      triton's 1,048,576 per-tensor cap at 16384x16384 (same failure confirmed empirically for
#      `_mxfp8_dim_km_swizzle_kernel`). The viable search space collapses to ~[1, 1] anyway.
# ---------------------------------------------------------------------------
@helion.kernel(
    config=helion.Config(block_sizes=[1, 1], num_warps=4, num_stages=1),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _nvfp4_gs_swizzle_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    outer_scale: torch.Tensor,  # (1,) f32 per-tensor outer scale (global amax, host-computed)
    qdata: torch.Tensor,  # (M, N // 2) uint8, fp4-packed (two e2m1 per byte), mutated in place
    scale5: torch.Tensor,  # (nrb, ncb, 32, 4, 4) e4m3 inner scale, swizzled, pre (4,4)->16
) -> None:
    M, N = x.shape
    nrb = M // 128  # M 128-row blocks
    ncb = N // 64   # N 64-col blocks = 4 inner-16-blocks
    xv = x.view(nrb, 4, 32, ncb, 4, 8, 2)  # [RB, a, b, CB, c, j, k]
    qv = qdata.view(nrb, 4, 32, ncb, 4, 8)  # (M, N//2) = [RB, a, b, CB, c, j] (8 packed bytes / 16-blk)
    for tile_rb, tile_cb in hl.tile([nrb, ncb]):
        outer = hl.load(outer_scale, [0])
        x_blk = xv[tile_rb, :, :, tile_cb, :, :, :].to(torch.float32)  # (trb,4,32,tcb,4,8,2)
        amax = torch.amax(torch.amax(torch.abs(x_blk), dim=6), dim=5)  # [RB,a,b,CB,c] over the 16
        # inner e4m3 block scale relative to the outer scale; round-trip through e4m3 BEFORE using it
        # in the reciprocal (matches the gold / the non-swizzle kernel -- load-bearing for bit-exact).
        inner_e4m3 = torch.clamp(
            (amax / _F4_E2M1_MAX) / outer, _E4M3_EPS, _F8E4M3_MAX
        ).to(torch.float8_e4m3fn)
        recip = (1.0 / outer) / inner_e4m3.to(torch.float32)  # [RB,a,b,CB,c]
        data_scaled = torch.clamp(
            x_blk * recip[:, :, :, :, :, None, None], -_F4_E2M1_MAX, _F4_E2M1_MAX
        )  # [RB,a,b,CB,c,j,k]
        # split the pack pair (k) without strided indexing: mask-and-reduce the size-2 axis.
        w_odd = hl.arange(2).to(torch.float32)  # [0.0, 1.0]
        even = torch.sum(data_scaled * (1.0 - w_odd), dim=6)  # [RB,a,b,CB,c,j] -> low nibble ($2)
        odd = torch.sum(data_scaled * w_odd, dim=6)  # [RB,a,b,CB,c,j] -> high nibble ($1)
        packed_u16 = hl.inline_asm_elementwise(
            _NVFP4_CVT_ASM, "=h,r,r", [odd, even],
            dtype=torch.uint16, is_pure=True, pack=1,
        )
        qv[tile_rb, :, :, tile_cb, :, :] = packed_u16.to(torch.uint8)
        # swizzled inner scale: [RB,a,b,CB,c] -> blocked [RB,CB,b,a,c]
        scale5[tile_rb, tile_cb, :, :, :] = inner_e4m3.permute(0, 3, 2, 1, 4)


def nvfp4_gs_swizzle_helion(x, outer_scale, **kwargs):
    """nvfp4 two-level cast in Helion with the e4m3 inner scale in the NVIDIA 32x4x4 swizzled block
    grid: per-16 e4m3 inner scale (relative to the host-computed per-tensor `outer_scale` aux),
    fp4-packed qdata (M, N//2) + the inner scale scattered directly into the swizzled block grid
    (nrb, ncb, 32, 16). Matches the gold `nvfp4_gs_swizzle_f`. The (4,4)->16 merge is a free contiguous
    reshape here. `**kwargs` are accepted and ignored."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0 and N % 64 == 0, (
        f"nvfp4 swizzle requires M divisible by 128 and N by 64, got {(M, N)}"
    )
    nrb, ncb = M // 128, N // 64
    qdata = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    scale5 = torch.empty((nrb, ncb, 32, 4, 4), dtype=torch.float8_e4m3fn, device=x.device)
    _nvfp4_gs_swizzle_kernel(x, outer_scale.reshape(1).to(torch.float32), qdata, scale5)
    inner_swizzled = scale5.reshape(nrb, ncb, 32, 16)
    return qdata.view(torch.float4_e2m1fn_x2), inner_swizzled


NVFP4_SWIZZLE = QuantCastHelionRecipe.from_gold(
    Nvfp4GsSwizzleGold, helion_fn=nvfp4_gs_swizzle_helion
)


# ---------------------------------------------------------------------------
# bf16 16x16 randomized Hadamard transform (RHT): bf16 in, bf16 out, NO scale (a 1-tuple output).
# Mirrors hadamard_rht_f -- out = (x.reshape(..., 16) @ rht).reshape(...). The RHT matrix is a (16,16)
# bf16 aux input (built on the host). Like the triton kernel, flatten the whole tensor to (n_groups,
# 16) groups of 16 and give each program a (BLOCK_G, 16) tile -> a batch of (BLOCK_G, 16) @ (16, 16)
# matmuls. Compute in fp32 (upcasting the bf16 inputs is exact, so an fp32 matmul reproduces torch's
# bf16 gemm with fp32 accumulation) then cast back to bf16. It's a bandwidth-bound elementwise-shaped
# op (read N, write N; the K=16 dot is tiny), but autotune_effort="none"'s default config picks a tiny
# block (block_sizes=[32] -> 512 elems/program) and only reaches ~40% peak. A swept block matters here:
# pin block_sizes=[512] (matching the triton kernel's BLOCK_G=512), num_warps=8 -> ~64% peak. Larger
# blocks (>=1024 groups) overflow tensor memory since the (bg,16)@(16,16) matmul lowers to tl.dot/tmem.
# ---------------------------------------------------------------------------
@helion.kernel(
    configs=[
        # for b200
        helion.Config(atomic_indexing=[], block_sizes=[128], indexing=['pointer', 'pointer', 'pointer'], load_eviction_policies=['first', ''], num_stages=3, num_warps=2, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[], range_unroll_factors=[0], range_warp_specializes=[None]),
        # for h100: drops range_warp_specializes (sm_90a config-spec rejects it here, expected 0);
        # Helion fills the per-arch default. Not perf-tuned for h100 -- just valid there.
        helion.Config(atomic_indexing=[], block_sizes=[128], indexing=['pointer', 'pointer', 'pointer'], load_eviction_policies=['first', ''], num_stages=3, num_warps=2, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[], range_unroll_factors=[0]),
    ],
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _rht_kernel(
    x: torch.Tensor,  # (n_groups, 16) bf16 input (the (M, N) tensor flattened to 16-element groups)
    rht: torch.Tensor,  # (16, 16) bf16 RHT matrix
    out: torch.Tensor,  # (n_groups, 16) bf16 output, mutated in place
) -> None:
    n_groups, _ = x.shape
    for tile_g in hl.tile(n_groups):
        x_blk = x[tile_g, :].to(torch.float32)  # (bg, 16)
        r = rht[:, :].to(torch.float32)  # (16, 16)
        out[tile_g, :] = torch.matmul(x_blk, r).to(torch.bfloat16)  # (bg, 16), fp32 accum -> bf16


def rht_helion(x, rht, **kwargs):
    """16x16 randomized Hadamard transform along the last dim in Helion (mirrors `hadamard_rht_f`):
    bf16 in, bf16 out, no scale. `rht` is the (16,16) RHT matrix (host-built aux). Returns a 1-tuple
    `(out,)`. `**kwargs` are accepted and ignored (the kernel owns its tiling)."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert N % 16 == 0, f"bf16_rht requires N divisible by 16, got N={N}"
    n_groups = (M * N) // 16
    out = torch.empty_like(x)
    _rht_kernel(x.view(n_groups, 16), rht.contiguous(), out.view(n_groups, 16))
    return (out,)


BF16_RHT = QuantCastHelionRecipe.from_gold(HadamardRht, helion_fn=rht_helion)


# ---------------------------------------------------------------------------
# Stochastic-rounding fp32 -> bf16 (mirrors sr_bf16_f). SR add-then-truncate: dither the 16 mantissa
# bits fp32->bf16 drops with a uniform 16-bit value, then mask them off. bf16 shares fp32's exponent
# so there's no rebias/scale/packing -- just the dither. Randomness comes from a counter-based Philox
# draw, but instead of `hl.rand` we drop into raw Triton via `hl.inline_triton` and call
# `tl.randint4x` directly -- the same batched primitive the hand-written Triton SR kernel uses (one
# Philox round yields FOUR int32 draws). The draws don't match the torch reference bit-for-bit --
# only the SR *property* (unbiased, lands on the two bracketing bf16 grid points) is well-defined, and
# that's what the test's correctness_fn checks for the *_sr recipes. To consume ALL FOUR draws from
# the single Philox round, we view the flat input as (n // 4, 4) and dither 4 elements per round: one
# offset (`hl.tile_index`) per group of 4, the four blocks (a, b, c, d) supplying the four columns'
# dithers. The columns are filled with a one-hot weighted sum over the size-4 minor axis
# (`hl.arange(4)`) rather than strided column indexing, which Helion rejects.
# Elementwise + bandwidth-bound (read fp32, write bf16) -> 81.9% peak. inline_triton is a raw-Triton
# HOP that aborts autotune_effort="full", so the config below (block_sizes=[1024]) was found under
# autotune_effort="none" and hand-pinned.
# ---------------------------------------------------------------------------
@helion.kernel(configs=[
    # for b200
    helion.Config(
        atomic_indexing=[], block_sizes=[1024], indexing=['pointer', 'tensor_descriptor'], load_eviction_policies=[''], num_stages=7, num_warps=8, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None]
    ),
    # for h100: same as b200 but drops range_warp_specializes, which the sm_90a config-spec
    # rejects for this kernel (expected 0 values); omitting it lets Helion fill the per-arch
    # default. Not perf-tuned for h100 -- just a config that compiles and runs there.
    helion.Config(
        atomic_indexing=[], block_sizes=[1024], indexing=['pointer', 'tensor_descriptor'], load_eviction_policies=[''], num_stages=7, num_warps=8, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0]
    ),
], static_shapes=True, ignore_warnings=[helion.exc.TensorOperationInWrapper])
def _sr_bf16_kernel(
    x: torch.Tensor,  # (n,) fp32 input, flattened (n divisible by 4)
    out: torch.Tensor,  # (n,) bf16 output, mutated in place
    seed: int,  # Philox seed (first 32 bits of the key)
) -> None:
    n, = x.shape
    nq = n // 4  # groups of 4 elements: one Philox round (4 draws) per group
    x4 = x.view(nq, 4)
    out4 = out.view(nq, 4)
    for tile in hl.tile(nq):
        xrow = x4[tile, :]  # (t, 4) fp32
        xi32 = xrow.view(torch.int32)  # bitcast fp32 -> int32
        col = hl.arange(4)  # size-4 minor index, for the one-hot draw -> column scatter
        offs = hl.tile_index(tile).to(torch.int64)  # (t,) per-group Philox offset
        # Raw Triton: one Philox round -> four random blocks (tl.randint4x). It returns uint32, so
        # bitcast each to int32 to match output_like / the downstream int32 math.
        r0, r1, r2, r3 = hl.inline_triton(
            """
            a, b, c, d = tl.randint4x({seed}, {offs})
            (a.to(tl.int32, bitcast=True), b.to(tl.int32, bitcast=True),
             c.to(tl.int32, bitcast=True), d.to(tl.int32, bitcast=True))
            """,
            {"seed": seed, "offs": offs},
            output_like=(offs.to(torch.int32),) * 4,  # four (t,) int32 blocks
        )
        # scatter draw j into column j across the size-4 minor axis (no strided column indexing)
        dither = (
            r0[:, None] * (col == 0).to(torch.int32)
            + r1[:, None] * (col == 1).to(torch.int32)
            + r2[:, None] * (col == 2).to(torch.int32)
            + r3[:, None] * (col == 3).to(torch.int32)
        )  # (t, 4) int32
        rand16 = dither & 0xFFFF  # low 16 bits: uniform dither in [0, 2**16)
        xi = xi32 + rand16  # add the dither...
        xi = xi & -65536  # ...then truncate the low 16 mantissa bits (-65536 == 0xFFFF0000)
        out4[tile, :] = xi.view(torch.float32).to(torch.bfloat16)  # exact: low 16 bits are zero


def sr_bf16_helion(x, key, **kwargs):
    """fp32 -> bf16 stochastic rounding in Helion (mirrors `sr_bf16_f`). `key` is a torch Philox key
    tensor; its first 32-bit word seeds `hl.rand`. SR is unbiased -- a value between two bf16 grid
    points rounds up with probability (x-lo)/(hi-lo). Returns a 1-tuple `(out,)`. `**kwargs` accepted
    and ignored (the kernel owns its tiling)."""
    assert x.dtype == torch.float32, f"SR bf16 expects fp32 input, got {x.dtype}"
    assert x.is_contiguous()
    n = x.numel()
    seed = int(key.reshape(-1)[0].item()) & 0x7FFFFFFF  # first word of the key as a Philox seed
    out = torch.empty_like(x, dtype=torch.bfloat16)
    _sr_bf16_kernel(x.view(n), out.view(n), seed)
    return (out,)


FP32_TO_BF16_SR = QuantCastHelionRecipe.from_gold(SrF32ToBf16, helion_fn=sr_bf16_helion)


# ---------------------------------------------------------------------------
# fp8 per-tensor ("tensorwise") cast with a PRECOMPUTED scale. The per-tensor amax/448 scale is a
# global reduction that is NOT tile-invariant, so (like the gold `float8_tensorwise_f`) it is
# computed outside the kernel and passed in as a scalar aux -- the kernel just divides every element
# by that one fixed scalar, a pure bandwidth-bound elementwise op (read bf16, write fp8). Flatten x
# to 1D so the tiling is a plain 1D sweep. The scalar scale is read with `hl.load(scale, [0])` (the
# scalar-scale idiom from the nvfp4 kernel). Config pinned from an autotune_effort="full" search at
# the 16384x16384 benchmark shape (a wide 8192-element 1D tile, 16 warps, tensor-descriptor load).
# ---------------------------------------------------------------------------
@helion.kernel(
    configs=[
        # for b200
        helion.Config(atomic_indexing=[], block_sizes=[8192], indexing=['tensor_descriptor', 'pointer', 'pointer'], load_eviction_policies=['first', 'first'], num_stages=2, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[], range_unroll_factors=[0], range_warp_specializes=[None]),
        # for h100: drops range_warp_specializes (sm_90a config-spec rejects it here, expected 0);
        # Helion fills the per-arch default. Not perf-tuned for h100 -- just valid there.
        helion.Config(atomic_indexing=[], block_sizes=[8192], indexing=['tensor_descriptor', 'pointer', 'pointer'], load_eviction_policies=['first', 'first'], num_stages=2, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[], range_unroll_factors=[0]),
    ],
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _float8_tensorwise_kernel(
    x: torch.Tensor,  # (n,) bf16 input, flattened
    scale: torch.Tensor,  # (1,) f32 precomputed per-tensor scale (global amax/448, host-computed)
    qdata: torch.Tensor,  # (n,) fp8_e4m3fn, mutated in place
) -> None:
    n, = x.shape
    for tile in hl.tile(n):
        s = hl.load(scale, [0])
        qdata[tile] = (x[tile].to(torch.float32) * (1.0 / s)).to(torch.float8_e4m3fn)


def float8_tensorwise_helion(x, scale, **kwargs):
    """fp8 per-tensor cast in Helion (mirrors the gold `float8_tensorwise_f`): divide every element
    by the precomputed per-tensor `scale` aux (a scalar fp32 tensor) and cast to fp8. `scale` is an
    input, not a returned output, so this returns a 1-tuple `(qdata,)`. `**kwargs` are accepted and
    ignored (the kernel owns its tiling)."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    n = x.numel()
    qdata = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    _float8_tensorwise_kernel(x.view(n), scale.reshape(1).to(torch.float32), qdata.view(n))
    return (qdata,)


FP8_TENSORWISE = QuantCastHelionRecipe.from_gold(
    Float8TensorwiseGold, helion_fn=float8_tensorwise_helion
)


# ---------------------------------------------------------------------------
# deepseek fp8 1x128 (dim-K): one fp32 scale per (row, 128-column block); qdata keeps the input's
# (M, N) layout (no transpose), scale is (M, N//128). Mirrors the gold `deepseek_1x128_f`. Viewing x
# as (M, nb=N//128, 128) makes each 128-column group the trailing reduction axis, so the block amax
# is a reduce over that axis and the store lands back in place. This is the natural
# (non-transposed) direction of the deepseek quant, so unlike the dim-M / dim-km variants there is no
# transpose-store and no 32/128-block-count tiling gymnastics -- a clean 2D tile over (M, nb).
# Config pinned from an autotune_effort="full" search at the 16384x16384 benchmark shape (a (64, 2)
# tile over (M, nb), persistent 128-column reduction, 16 warps). NOTE: that search needs
# HELION_AUTOTUNE_IGNORE_ERRORS=1 -- some flatten_loops+reduction_loops config combos miscompile the
# dim-2 reduction (`tl.max` over a flattened 2D acc) and would otherwise abort the whole search.
# ---------------------------------------------------------------------------
@helion.kernel(
    configs=[
        # for b200
        helion.Config(atomic_indexing=[], block_sizes=[64, 2], flatten_loops=[False], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], l2_groupings=[1], load_eviction_policies=['last', '', 'last'], loop_orders=[[1, 0]], num_stages=6, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[], range_unroll_factors=[0], range_warp_specializes=[None], reduction_loops=[None]),
        # for h100: drops range_warp_specializes (sm_90a config-spec rejects it here, expected 0);
        # Helion fills the per-arch default. Not perf-tuned for h100 -- just valid there.
        helion.Config(atomic_indexing=[], block_sizes=[64, 2], flatten_loops=[False], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], l2_groupings=[1], load_eviction_policies=['last', '', 'last'], loop_orders=[[1, 0]], num_stages=6, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[], range_unroll_factors=[0], reduction_loops=[None]),
    ],
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _deepseek_1x128_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    qdata: torch.Tensor,  # (M, N) fp8_e4m3fn, mutated in place
    scale: torch.Tensor,  # (M, N // 128) f32, mutated in place
) -> None:
    M, N = x.shape
    nb = N // 128  # number of 128-column blocks per row (== scale columns)
    xv = x.view(M, nb, 128)
    qv = qdata.view(M, nb, 128)
    for tile_m, tile_nb in hl.tile([M, nb]):
        x_blk = xv[tile_m, tile_nb, :].to(torch.float32)  # (t_m, t_nb, 128)
        amax = torch.clamp(torch.amax(torch.abs(x_blk), dim=2), min=1e-12)  # (t_m, t_nb)
        s = amax / _FP8_MAX
        y = (x_blk * (1.0 / s)[:, :, None]).to(torch.float8_e4m3fn)  # (t_m, t_nb, 128)
        qv[tile_m, tile_nb, :] = y
        scale[tile_m, tile_nb] = s


def fp8_deepseek_1x128_helion(x, **kwargs):
    """deepseek fp8 1x128 (dim-K) in Helion: abs-max over each 128-column block (one fp32 scale per
    (row, 128-col-block)), quantize to fp8 in the input's (M, N) layout (no transpose). Matches the
    gold `deepseek_1x128_f`. `**kwargs` are accepted and ignored (the kernel owns its tiling)."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert N % 128 == 0, f"deepseek 1x128 requires N divisible by 128, got N={N}"
    qdata = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((M, N // 128), dtype=torch.float32, device=x.device)
    _deepseek_1x128_kernel(x, qdata, scale)
    return qdata, scale


FP8_DEEPSEEK_1X128 = QuantCastHelionRecipe.from_gold(
    Deepseek1x128Gold, helion_fn=fp8_deepseek_1x128_helion
)


# ---------------------------------------------------------------------------
# mxfp8 (dim-K) with the e8m0 scale in the NVIDIA 32x4x4 SWIZZLED block grid. The natural
# (non-transposed) direction of the mxfp8 quant -- 1x32 blocks along columns, qdata stays in (M, N) --
# but the e8m0 scale is scattered straight into the swizzled block layout in-kernel (not stored plain
# then swizzled in a wrapper post-pass). Mirrors the gold `mxfp8_swizzle_f` (= mxfp8_f
# then _to_blocked_4d on the (M, N//32) scale).
#
# Swizzle math (`_to_blocked_4d` on the (M, N//32) scale, index [m, w]): the block grid is
# (nrb=M//128, ncb=N//128, 32, 16) and blocked[m//128, w//4, m%32, ((m%128)//32)*4 + w%4] = scale[m, w].
# As in the dim-M swizzle kernel, we tile over the BLOCK-COUNT dims [nrb, ncb] so the tile indices ARE
# the block ordinals (Nb, Cb) and the swizzled store is a plain 5D index. We split BOTH the M-rows and
# the 32-col-block index fully:
#   M-rows: m = 128*Nb + 32*a + b   (Nb in [0,nrb), a in [0,4), b in [0,32));  m%32 = b, (m%128)//32 = a
#   N-cols: n = 128*Cb + 32*c + g32 (Cb in [0,ncb), c in [0,4), g32 in [0,32)); the 32-col block is
#           w = n//32 = 4*Cb + c, so w//4 = Cb and w%4 = c.
# so x_blk is 6D [Nb, a, b, Cb, c, g32]. The 1x32 reduction is over g32 (the within-block columns).
# The swizzle is then a pure permute of the reduced scale [Nb,a,b,Cb,c] -> blocked atom [Nb,Cb,b,a,c];
# the final (a=4, c=4) -> 16 nibble merge is a free contiguous reshape in the wrapper.
#
# Config: autotune_effort="full" genuinely FAILS here (tried at 16384x16384), so this is pinned to a
# manual block_sizes=[1, 1] -- the same fallback as the sibling swizzle / dim-km kernels. The reason:
# tiling the BLOCK-COUNT dims means a program's x-load tile is block_nrb * block_ncb * (4*32*4*32) =
# block_nrb * block_ncb * 16384 fp32 elements, so any block product > 64 overflows triton's hard
# 1,048,576 per-tensor numel cap. The autotuner doesn't know about that 16384x multiplier: its default
# baseline config ([32, 32]) overflows (so even computing the baseline needs a custom baseline_fn), and
# its random search population is dominated by overflowing block sizes -- the full search returned
# helion.exc.NoConfigFound. The tiny valid corner it can't reliably find is exactly [1, 1]: one 128x128
# x-block per program (a 16384-element / 64KB fp32 tile), which is bit-exact to the gold AND already
# fast -- ~0.13 ms / ~79% of peak at 16384x16384, on par with the manual triton/cute swizzle kernels.
# ---------------------------------------------------------------------------
@helion.kernel(
    config=helion.Config(block_sizes=[1, 1], num_warps=4, num_stages=1),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _mxfp8_swizzle_kernel(
    x: torch.Tensor,  # (M, N) bf16 input
    qdata: torch.Tensor,  # (M, N) fp8_e4m3fn, mutated in place (natural frame)
    scale5: torch.Tensor,  # (nrb, ncb, 32, 4, 4) uint8 e8m0 bits, mutated (swizzled, pre (4,4)->16)
) -> None:
    M, N = x.shape
    nrb = M // 128  # M 128-row blocks
    ncb = N // 128  # N 128-col blocks (each = 4 col-blocks-of-32)
    xv = x.view(nrb, 4, 32, ncb, 4, 32)      # M=(Nb, a, b), N=(Cb, c, g32)
    qv = qdata.view(nrb, 4, 32, ncb, 4, 32)  # natural (M, N), same axes as xv
    for tile_nb, tile_cb in hl.tile([nrb, ncb]):
        x_blk = xv[tile_nb, :, :, tile_cb, :, :].to(torch.float32)  # (tnb,4,32,tcb,4,32)
        amax = torch.amax(torch.abs(x_blk), dim=5)  # (tnb,4,32,tcb,4); reduce the 32 columns (g32)
        biased = _amax_to_e8m0_biased(amax)  # (tnb,4,32,tcb,4) int32 e8m0 exponent
        rcp = _e8m0_biased_to_reciprocal_fp32(biased)  # (tnb,4,32,tcb,4) fp32 reciprocal pow2 factor
        y = (x_blk * rcp[:, :, :, :, :, None]).to(torch.float8_e4m3fn)  # broadcast over g32
        qv[tile_nb, :, :, tile_cb, :, :] = y
        # swizzle: [Nb,a,b,Cb,c] -> blocked atom [Nb,Cb,b,a,c].
        scale5[tile_nb, tile_cb, :, :, :] = biased.permute(0, 3, 2, 1, 4).to(torch.uint8)


def mxfp8_swizzle_helion(x, **kwargs):
    """mxfp8 (dim-K) in Helion with the e8m0 scale in the NVIDIA 32x4x4 swizzled block grid:
    abs-max over each 1x32 column block, e8m0 power-of-two scale, quantize to fp8 in the input's
    (M, N) layout (no transpose), and write the scale scattered directly into the swizzled block grid
    (nrb, ncb, 32, 16). Matches the gold `mxfp8_swizzle_f`. The (4,4)->16 nibble merge is a free
    contiguous reshape here. `**kwargs` are accepted and ignored."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0, (
        f"mxfp8 swizzle requires M,N divisible by 128, got {(M, N)}"
    )
    nrb, ncb = M // 128, N // 128
    qdata = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale5 = torch.empty((nrb, ncb, 32, 4, 4), dtype=torch.uint8, device=x.device)
    _mxfp8_swizzle_kernel(x, qdata, scale5)
    scale = scale5.reshape(nrb, ncb, 32, 16).view(torch.float8_e8m0fnu)
    return qdata, scale


MXFP8_SWIZZLE = QuantCastHelionRecipe.from_gold(
    Mxfp8SwizzleGold, helion_fn=mxfp8_swizzle_helion
)


# Order mirrors quant_cast_gold.ALL_RECIPES / quant_cast_triton.ALL_RECIPES (only the recipes with
# a Helion impl are listed; more will be added as they're ported).
ALL_RECIPES = [
    ("fp8_tensorwise_precalc_scale", FP8_TENSORWISE),
    ("fp8_deepseek_1x128", FP8_DEEPSEEK_1X128),
    ("mxfp8_swizzle", MXFP8_SWIZZLE),
    ("fp8_deepseek_1x128_dim_m", FP8_DEEPSEEK_1X128_DIM_M),
    ("mxfp8_dim_m", MXFP8_DIM_M),
    ("mxfp8_dim_m_swizzle", MXFP8_DIM_M_SWIZZLE),
    ("mxfp8_dim_km", MXFP8_DIM_KM),
    ("mxfp8_dim_km_swizzle", MXFP8_DIM_KM_SWIZZLE),
    ("fp8_deepseek_1x128_dim_km", FP8_DEEPSEEK_1X128_DIM_KM),
    ("mxfp8_32x32", MXFP8_32X32),
    ("fp8_deepseek_128x128", FP8_DEEPSEEK_128X128),
    ("nvfp4", NVFP4),
    ("nvfp4_swizzle", NVFP4_SWIZZLE),
    ("bf16_rht", BF16_RHT),
    ("fp32_to_bf16_sr", FP32_TO_BF16_SR),
]
