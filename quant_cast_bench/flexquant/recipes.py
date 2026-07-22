from dataclasses import dataclass
from typing import Callable

import torch
from torch.nn.functional import SwizzleType

from .nvfp4_utils import (
    F4_E2M1_MAX,
    f32_to_f4_unpacked,
    f32_to_f4_unpacked_lut,
    pack_uint4,
)
from .swizzle import to_blocked_2d

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12

F8E4M3_MAX = FP8_MAX
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny

# mxfp8 (e8m0 power-of-two block scale) constants, from torchao mx_formats.
E8M0_EXPONENT_BIAS = 127
F32_EXP_BIAS = 127
MBITS_F32 = 23
F8E4M3_MAX_POW2 = 8
F32_MIN_NORMAL = 2.0**-126
E8M0_NAN = 255


@dataclass(frozen=True)
class Recipe:
    name: str
    block_size: int | tuple[int, int] | list
    dim: int | tuple[int, int] | list
    qdata_dtype: torch.dtype
    scale_dtype: torch.dtype | list[torch.dtype]
    amax_to_scale_fn: Callable | list[Callable]
    cast_to_dtype_fn: Callable
    # arguments below are for debugging only
    # Plain PyTorch reference implementation (including the tiling)
    _reference_fn: Callable
    scale_swizzle: SwizzleType | None = None


def _deepseek_fp8_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    amax_fp32 = amax.clamp(min=EPS).to(torch.float32)
    return amax_fp32 / FP8_MAX


def _deepseek_fp8_cast_to_dtype_fn(
    tile: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    # Recover the reciprocal from the forward scale (prod stores forward scale
    # but multiplies by the reciprocal).
    reciprocal = (1.0 / scale)
    y = tile * reciprocal
    return y.to(torch.float8_e4m3fn)


def _deepseek_fp8_1_128_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, last = x.shape
    n_blocks = last // 128
    x_b = x.reshape(*lead, n_blocks, 128)
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=EPS).to(torch.float32)
    scale = amax / FP8_MAX  # forward scale (on-disk format)
    scale = scale.to(torch.float32)
    reciprocal = 1.0 / scale
    y = x_b.to(torch.float32) * reciprocal
    qdata = y.to(torch.float8_e4m3fn).reshape(*lead, last)
    return qdata, scale.squeeze(-1)


deepseek_fp8_1_128 = Recipe(
    name="deepseek_fp8_1_128",
    block_size=128,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn,
    _reference_fn=_deepseek_fp8_1_128_reference,
)

def _deepseek_fp8_1_128_dim_m_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # dim=-2: reduce across M, output in transposed (K, M) layout.
    return _deepseek_fp8_1_128_reference(x.transpose(-2, -1).contiguous())


deepseek_fp8_1_128_dim_m = Recipe(
    name="deepseek_fp8_1_128_dim_m",
    block_size=128,
    dim=-2,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn,
    _reference_fn=_deepseek_fp8_1_128_dim_m_reference,
)

def _deepseek_fp8_128_128_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, D1, D2 = x.shape
    n1, n2 = D1 // 128, D2 // 128
    x_b = (
        x.reshape(*lead, n1, 128, n2, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, n1, n2, 128 * 128)
    )
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=EPS).to(torch.float32)
    scale = amax / FP8_MAX  # forward scale
    scale = scale.to(torch.float32)
    reciprocal = 1.0 / scale
    y = x_b.to(torch.float32) * reciprocal
    qdata_b = y.to(torch.float8_e4m3fn)
    qdata = (
        qdata_b.reshape(*lead, n1, n2, 128, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, D1, D2)
    )
    return qdata, scale.squeeze(-1)


deepseek_fp8_128_128 = Recipe(
    name="deepseek_fp8_128_128",
    block_size=(128, 128),
    dim=(-2, -1),
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn,
    _reference_fn=_deepseek_fp8_128_128_reference,
)


# nvfp4 with single-level scaling and packed fp4 (two fp4 values per byte,
# stored as torch.float4_e2m1fn_x2). Mirrors the `per_tensor_scale is None`
# branch of `nvfp4_quantize` in torchao.

def _nvfp4_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    block_scale = amax.to(torch.float32) / F4_E2M1_MAX
    return torch.clamp(block_scale, min=E4M3_EPS, max=F8E4M3_MAX).to(
        torch.float8_e4m3fn
    )


def _nvfp4_cast_to_dtype_fn(
    tile: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    scale_fp32 = scale.to(torch.float32)
    data_scaled = tile.to(torch.float32) * (1.0 / scale_fp32)
    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_unpacked = f32_to_f4_unpacked(data_scaled)
    return pack_uint4(data_unpacked).view(torch.float4_e2m1fn_x2)


def _nvfp4_no_gs_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, last = x.shape
    n_blocks = last // 16
    x_b = x.reshape(*lead, n_blocks, 16)
    amax = x_b.abs().amax(dim=-1, keepdim=True)
    scale_e4m3 = _nvfp4_amax_to_scale_fn(amax)
    qdata_b = _nvfp4_cast_to_dtype_fn(x_b, scale_e4m3)
    # qdata_b shape: (*lead, n_blocks, 8) packed -> (*lead, last // 2)
    qdata = qdata_b.reshape(*lead, last // 2)
    return qdata, scale_e4m3.squeeze(-1)


nvfp4_no_gs = Recipe(
    name="nvfp4_no_gs",
    block_size=16,
    dim=-1,
    qdata_dtype=torch.float4_e2m1fn_x2,
    scale_dtype=torch.float8_e4m3fn,
    amax_to_scale_fn=_nvfp4_amax_to_scale_fn,
    cast_to_dtype_fn=_nvfp4_cast_to_dtype_fn,
    _reference_fn=_nvfp4_no_gs_reference,
)


# Same format as `nvfp4_no_gs`, but the block scale is returned in the NVIDIA
# blocked (32x4x4 swizzle) layout that `_scaled_mm` consumes. Only the
# reference differs from `nvfp4_no_gs` (it swizzles the scale via the same
# `to_blocked_2d` helper the api path uses).
def _nvfp4_no_gs_swizzle_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, last = x.shape
    n_blocks = last // 16
    x_b = x.reshape(*lead, n_blocks, 16)
    amax = x_b.abs().amax(dim=-1, keepdim=True)
    scale_e4m3 = _nvfp4_amax_to_scale_fn(amax)
    qdata_b = _nvfp4_cast_to_dtype_fn(x_b, scale_e4m3)
    qdata = qdata_b.reshape(*lead, last // 2)
    return qdata, to_blocked_2d(scale_e4m3.squeeze(-1))


nvfp4_no_gs_swizzle = Recipe(
    name="nvfp4_no_gs_swizzle",
    block_size=16,
    dim=-1,
    qdata_dtype=torch.float4_e2m1fn_x2,
    scale_dtype=torch.float8_e4m3fn,
    amax_to_scale_fn=_nvfp4_amax_to_scale_fn,
    cast_to_dtype_fn=_nvfp4_cast_to_dtype_fn,
    _reference_fn=_nvfp4_no_gs_swizzle_reference,
    scale_swizzle=SwizzleType.SWIZZLE_32_4_4,
)


# Same format as `nvfp4_no_gs` but using the LUT-based fp32 -> fp4 cast
# (`searchsorted` + permutation gather). No round-to-nearest-even — ties
# round toward the lower-magnitude side. Output bytes can differ from
# `nvfp4_no_gs` on tied inputs but the format and storage layout are
# identical.

def _nvfp4_lut_cast_to_dtype_fn(
    tile: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    scale_fp32 = scale.to(torch.float32)
    data_scaled = tile.to(torch.float32) * (1.0 / scale_fp32)
    data_unpacked = f32_to_f4_unpacked_lut(data_scaled)
    return pack_uint4(data_unpacked).view(torch.float4_e2m1fn_x2)


def _nvfp4_no_gs_lut_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, last = x.shape
    n_blocks = last // 16
    x_b = x.reshape(*lead, n_blocks, 16)
    amax = x_b.abs().amax(dim=-1, keepdim=True)
    scale_e4m3 = _nvfp4_amax_to_scale_fn(amax)
    qdata_b = _nvfp4_lut_cast_to_dtype_fn(x_b, scale_e4m3)
    qdata = qdata_b.reshape(*lead, last // 2)
    return qdata, scale_e4m3.squeeze(-1)


nvfp4_no_gs_lut = Recipe(
    name="nvfp4_no_gs_lut",
    block_size=16,
    dim=-1,
    qdata_dtype=torch.float4_e2m1fn_x2,
    scale_dtype=torch.float8_e4m3fn,
    amax_to_scale_fn=_nvfp4_amax_to_scale_fn,
    cast_to_dtype_fn=_nvfp4_lut_cast_to_dtype_fn,
    _reference_fn=_nvfp4_no_gs_lut_reference,
)


# nvfp4 with two-level scaling: per-tensor fp32 outer scale + per-block e4m3
# inner scale. Mirrors the `per_tensor_scale is not None` branch of
# `nvfp4_quantize` in torchao, with the outer (per-tensor) scale computed by
# the framework from the input's amax.

def _nvfp4_outer_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    return amax.to(torch.float32) / (F8E4M3_MAX * F4_E2M1_MAX)


def _nvfp4_inner_amax_to_scale_fn(
    local_amax: torch.Tensor, outer_scale: torch.Tensor
) -> torch.Tensor:
    block_scale_fp32 = local_amax.to(torch.float32) / F4_E2M1_MAX
    scaled = block_scale_fp32 / outer_scale
    return torch.clamp(scaled, min=E4M3_EPS, max=F8E4M3_MAX).to(
        torch.float8_e4m3fn
    )


def _nvfp4_with_gs_cast_to_dtype_fn(
    tile: torch.Tensor,
    inner_scale: torch.Tensor,
    outer_scale: torch.Tensor,
) -> torch.Tensor:
    inner_fp32 = inner_scale.to(torch.float32)
    reciprocal = (1.0 / outer_scale) / inner_fp32
    data_scaled = tile.to(torch.float32) * reciprocal
    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_unpacked = f32_to_f4_unpacked(data_scaled)
    return pack_uint4(data_unpacked).view(torch.float4_e2m1fn_x2)


def _nvfp4_with_gs_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    outer_amax = x.abs().to(torch.float32).amax()
    outer_scale = _nvfp4_outer_amax_to_scale_fn(outer_amax)

    *lead, last = x.shape
    n_blocks = last // 16
    x_b = x.reshape(*lead, n_blocks, 16)
    local_amax = x_b.abs().amax(dim=-1, keepdim=True)
    inner_scale = _nvfp4_inner_amax_to_scale_fn(local_amax, outer_scale)
    qdata_b = _nvfp4_with_gs_cast_to_dtype_fn(x_b, inner_scale, outer_scale)
    qdata = qdata_b.reshape(*lead, last // 2)
    return qdata, [inner_scale.squeeze(-1), outer_scale]


nvfp4_with_gs = Recipe(
    name="nvfp4_with_gs",
    block_size=[16, (-1, -1)],
    dim=[-1, (-2, -1)],
    qdata_dtype=torch.float4_e2m1fn_x2,
    scale_dtype=[torch.float8_e4m3fn, torch.float32],
    amax_to_scale_fn=[
        _nvfp4_inner_amax_to_scale_fn,
        _nvfp4_outer_amax_to_scale_fn,
    ],
    cast_to_dtype_fn=_nvfp4_with_gs_cast_to_dtype_fn,
    _reference_fn=_nvfp4_with_gs_reference,
)


# mxfp8 with FLOOR scale rounding: per-block (32 elements) e8m0 power-of-two
# scale + e4m3 data. Mirrors torchao `to_mx(..., ScaleCalculationMode.FLOOR)`
# (`mx_formats/mx_tensor.py`), the OCP MX Spec 1.0 method
# `X = 2^(floor(log2(max_abs)) - max_exp)`. Single-level, no packing.

def _mxfp8_floor_amax_to_scale_fn(amax: torch.Tensor) -> torch.Tensor:
    # amax: (M, nb, 1) -> e8m0 block scale via FLOOR rounding. The fp32
    # exponent of amax is extracted by integer bit-ops (no log2), shifted by
    # the e4m3 target power-of-two, clamped, biased, and stored as e8m0.
    max_abs = amax.to(torch.float32)
    max_abs_int32 = max_abs.view(torch.int32)
    extracted_pow2 = ((max_abs_int32 >> MBITS_F32) & 0xFF) - F32_EXP_BIAS
    scale_unbiased = extracted_pow2 - F8E4M3_MAX_POW2
    scale_unbiased = torch.clamp(
        scale_unbiased, -E8M0_EXPONENT_BIAS, E8M0_EXPONENT_BIAS + 1
    )
    scale_biased = (scale_unbiased + E8M0_EXPONENT_BIAS).to(torch.uint8)
    scale_biased = torch.where(
        torch.isnan(max_abs), torch.full_like(scale_biased, E8M0_NAN), scale_biased
    )
    return scale_biased.view(torch.float8_e8m0fnu)


def _mxfp8_floor_cast_to_dtype_fn(
    tile: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    # Reconstruct the fp32 power-of-two factor by shifting the e8m0 biased
    # exponent byte back into the fp32 exponent field, then clamp away
    # denormals (matches torchao's `clamp(min=F32_MIN_NORMAL)`).
    biased_i32 = scale.view(torch.uint8).to(torch.int32)
    scale_fp32 = (biased_i32 << MBITS_F32).view(torch.float32)
    scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)
    data = tile.to(torch.float32) / scale_fp32
    return data.to(torch.float8_e4m3fn)


def _mxfp8_floor_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, last = x.shape
    n_blocks = last // 32
    x_b = x.reshape(*lead, n_blocks, 32)
    amax = x_b.abs().amax(dim=-1, keepdim=True)
    scale_e8m0 = _mxfp8_floor_amax_to_scale_fn(amax)
    qdata_b = _mxfp8_floor_cast_to_dtype_fn(x_b, scale_e8m0)
    qdata = qdata_b.reshape(*lead, last)
    return qdata, scale_e8m0.squeeze(-1)


mxfp8_floor = Recipe(
    name="mxfp8_floor",
    block_size=32,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float8_e8m0fnu,
    amax_to_scale_fn=_mxfp8_floor_amax_to_scale_fn,
    cast_to_dtype_fn=_mxfp8_floor_cast_to_dtype_fn,
    _reference_fn=_mxfp8_floor_reference,
)


# Same format as `mxfp8_floor`, but reducing across M (dim=-2): output qdata
# and scale in transposed (K, M) layout. Mirrors `deepseek_fp8_1_128_dim_m` --
# the reference reuses the dim=-1 reference on the transposed input.
def _mxfp8_floor_dim_m_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _mxfp8_floor_reference(x.transpose(-2, -1).contiguous())


mxfp8_floor_dim_m = Recipe(
    name="mxfp8_floor_dim_m",
    block_size=32,
    dim=-2,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float8_e8m0fnu,
    amax_to_scale_fn=_mxfp8_floor_amax_to_scale_fn,
    cast_to_dtype_fn=_mxfp8_floor_cast_to_dtype_fn,
    _reference_fn=_mxfp8_floor_dim_m_reference,
)


# Same format as `mxfp8_floor`, but the e8m0 block scale is returned in the
# NVIDIA blocked (32x4x4 swizzle) layout. Only the reference differs (it
# swizzles the scale via the same `to_blocked_2d` helper the api path uses).
def _mxfp8_floor_swizzle_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, last = x.shape
    n_blocks = last // 32
    x_b = x.reshape(*lead, n_blocks, 32)
    amax = x_b.abs().amax(dim=-1, keepdim=True)
    scale_e8m0 = _mxfp8_floor_amax_to_scale_fn(amax)
    qdata_b = _mxfp8_floor_cast_to_dtype_fn(x_b, scale_e8m0)
    qdata = qdata_b.reshape(*lead, last)
    return qdata, to_blocked_2d(scale_e8m0.squeeze(-1))


mxfp8_floor_swizzle = Recipe(
    name="mxfp8_floor_swizzle",
    block_size=32,
    dim=-1,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float8_e8m0fnu,
    amax_to_scale_fn=_mxfp8_floor_amax_to_scale_fn,
    cast_to_dtype_fn=_mxfp8_floor_cast_to_dtype_fn,
    _reference_fn=_mxfp8_floor_swizzle_reference,
    scale_swizzle=SwizzleType.SWIZZLE_32_4_4,
)

