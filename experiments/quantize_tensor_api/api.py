import os
import sys
from enum import IntEnum, StrEnum

import torch
from torch import Tensor
from torch.nn.functional import SwizzleType  # core enum: NO_SWIZZLE=0, SWIZZLE_32_4_4=1

# The editable install only exposes `quant_cast_bench`, not the `experiments` tree, so put the repo
# root on sys.path to make the sibling `experiments.quantize_tensor_api.moe_utils` import resolve regardless of
# how this module is reached (as a package, as a bare `api`, or under pytest).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.quantize_tensor_api.moe_utils import (  # noqa: E402
    BLOCK_SIZE,
    _to_blocked_2d_k_groups,
    _to_blocked_2d_m_groups,
    _to_blocked_per_group_3d,
    quantize_2d_act,
)
from quant_cast_bench.quant_cast_gold.recipes import mxfp8_f  # noqa: E402
from quant_cast_bench.quant_cast_triton.recipes import (  # noqa: E402
    mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_triton,
    mxfp8_32x32_triton,
    mxfp8_dim_km_swizzle_triton,
    mxfp8_dim_km_triton,
    mxfp8_dim_m_swizzle_triton,
    mxfp8_dim_m_triton,
    mxfp8_swizzle_triton,
    mxfp8_triton,
    nvfp4_swizzle_triton,
)


class ScalingType(IntEnum):
    # Mirrors torch.nn.functional.ScalingType (core, a pybind Enum), including its int values. In the
    # real/upstreamed version this should import and use core's enum directly rather than redefining
    # it here. ScalingType names only the 2D scale BLOCK SIZE, not the tensor dims it maps to; the
    # block->dim mapping is carried separately (here by QuantOrientation, in a GEMM by scaled_mm).
    TensorWise = 0
    RowWise = 1
    BlockWise1x16 = 2
    BlockWise1x32 = 3
    BlockWise1x128 = 4
    BlockWise128x128 = 5
    BlockWise32x32 = 6  # LOCAL addition, not in core; needed for first-class 32x32 mxfp8.


class QuantOrientation(StrEnum):
    # Maps the ScalingType block onto the incoming tensor's dims and sets the output layout. Output is
    # always contiguous. The fused dual-orientation cast (both pairs at once) lives in a separate
    # entry point, quantize_tensor_bidirectional, not as a value here.

    # block maps to (M, K) as given
    NATURAL = "natural"
    # block maps to the (K, M) view; qdata/scale written transposed-contig
    TRANSPOSED = "transposed"


class RoundingMode(StrEnum):
    # how qdata values are rounded when cast into float8_e4m3fn.
    RTNE = "rtne"                # round to nearest, ties to even
    STOCHASTIC = "stochastic"    # stochastic rounding


class InnerScaleCalc(StrEnum):
    # The per-block ("inner") scale strategy: fixes BOTH the scale dtype and the amax->scale
    # computation.
    #   E8M0_RCEIL -- float8_e8m0fnu (power-of-two) scale via reciprocal-multiply with RCEIL
    #     rounding (mxfp8; no outer scale).
    #   E4M3_NVFP4 -- float8_e4m3fn inner scale computed relative to a per-tensor fp32 OUTER scale
    #     (nvfp4 two-level scaling); requires the caller to pass the precomputed `outer_scale`.
    E8M0_RCEIL = "e8m0_rceil"
    E4M3_NVFP4 = "e4m3_nvfp4"


def quantize_tensor(
    input: Tensor,
    *,
    qdata_dtype: torch.dtype = torch.float8_e4m3fn,
    inner_scale_calc: InnerScaleCalc = InnerScaleCalc.E8M0_RCEIL,
    scaling_type: ScalingType = ScalingType.BlockWise1x32,
    orientation: QuantOrientation = QuantOrientation.NATURAL,
    swizzle_type: SwizzleType = SwizzleType.SWIZZLE_32_4_4,
    rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
    outer_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize `input` to a block-scaled low-precision format in one orientation: `qdata_dtype` qdata
    + one `inner_scale_calc` scale per block.

    For the fused dual-orientation cast (both the natural and transposed pairs in one pass), use
    `quantize_tensor_bidirectional`.

    Args:
      input: 2D input tensor (bf16 or fp32) of shape (M, K), or 3D of shape (E, M, K).
      qdata_dtype: qdata element format -- torch.float8_e4m3fn (mxfp8) or torch.float4_e2m1fn_x2
        (nvfp4, 2D only).
      inner_scale_calc: per-block scale strategy -- fixes the scale dtype and the amax->scale
        computation. InnerScaleCalc.E8M0_RCEIL (mxfp8) or InnerScaleCalc.E4M3_NVFP4 (nvfp4).
      scaling_type: BlockWise1x32 / BlockWise32x32 (mxfp8), or BlockWise1x16 (nvfp4)
      orientation: how scaling_type maps onto (M, K) and how the output is laid out.
        NATURAL maps the scaling_type to (M, K)
        TRANSPOSED maps it to the (K, M) view and writes the outputs transposed-contiguous
      swizzle_type: NO_SWIZZLE or SWIZZLE_32_4_4. Note that for 3d inputs, swizzle is applied per-expert.
      rounding_mode: RTNE or STOCHASTIC
      random_key: entropy source for stochastic rounding
      outer_scale: precomputed per-tensor fp32 outer scale, required for nvfp4 (float4_e2m1fn_x2)
        two-level scaling; must be None otherwise.

    Returns:
        2 tensors (qdata, scale)
    """
    assert input.dim() in (2, 3), f"only 2D or 3D input supported, got {input.dim()}D"
    if rounding_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("rounding_mode=STOCHASTIC is not implemented yet")
    if random_key is not None:
        raise NotImplementedError("random_key (stochastic rounding) is not implemented yet")

    # nvfp4: fp4 (e2m1) qdata + a float8_e4m3fn inner scale computed relative to a per-tensor fp32
    # OUTER scale (two-level scaling). The outer scale is a global amax reduction, so the caller
    # precomputes it and passes it in. Dense 2D only for now.
    if qdata_dtype == torch.float4_e2m1fn_x2:
        assert inner_scale_calc == InnerScaleCalc.E4M3_NVFP4, (
            f"float4_e2m1fn_x2 qdata requires inner_scale_calc=E4M3_NVFP4, got {inner_scale_calc!r}"
        )
        assert input.dim() == 2, "nvfp4 quantization is only supported for 2D input"
        assert outer_scale is not None, "nvfp4 quantization requires a precomputed outer_scale"
        assert input.is_contiguous(), "input must be contiguous"
        spec = (scaling_type, orientation, swizzle_type)
        if spec == (ScalingType.BlockWise1x16, QuantOrientation.NATURAL, SwizzleType.SWIZZLE_32_4_4):
            return nvfp4_swizzle_triton(input, outer_scale)
        raise ValueError(
            f"unsupported (scaling_type, orientation, swizzle_type)={spec!r} for nvfp4 "
            "(float4_e2m1fn_x2); supported: (BlockWise1x16, NATURAL, SWIZZLE_32_4_4)"
        )

    # mxfp8: float8_e4m3fn qdata + e8m0 rceil inner scale; no outer scale.
    assert qdata_dtype == torch.float8_e4m3fn, (
        f"only float8_e4m3fn or float4_e2m1fn_x2 qdata supported, got {qdata_dtype}"
    )
    assert inner_scale_calc == InnerScaleCalc.E8M0_RCEIL, (
        f"only InnerScaleCalc.E8M0_RCEIL supported for float8_e4m3fn, got {inner_scale_calc!r}"
    )
    assert outer_scale is None, "outer_scale is only used by nvfp4 (float4_e2m1fn_x2) quantization"

    if input.dim() == 3:
        # Per-expert (E, N, K) batching is a separate code path from the 2D casts below: each of the
        # E slices is quantized independently (scaling never applies to the E dimension), swizzling is
        # per-expert. The public entry point is merged but the computation is not.
        if (scaling_type, swizzle_type) != (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}); "
                "quantize_tensor with 3D (E, N, K) input supports only (BlockWise1x32, SWIZZLE_32_4_4)"
            )
        if orientation == QuantOrientation.NATURAL:
            q_nat, s_nat = mxfp8_f(input.contiguous())  # (E,N,K), 1x32 along K
            return q_nat, _to_blocked_per_group_3d(s_nat)
        if orientation == QuantOrientation.TRANSPOSED:
            q_t, s_t = mxfp8_f(input.transpose(-2, -1).contiguous())  # (E,K,N), 1x32 along N
            return q_t, _to_blocked_per_group_3d(s_t)
        raise ValueError(
            f"orientation={orientation!r} is not supported for 3D (E, N, K) quantize_tensor; "
            "supported: NATURAL, TRANSPOSED"
        )

    assert input.is_contiguous(), "input must be contiguous"
    spec = (scaling_type, orientation, swizzle_type)
    if spec == (ScalingType.BlockWise1x32, QuantOrientation.NATURAL, SwizzleType.NO_SWIZZLE):
        assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
        return mxfp8_triton(input)
    if spec == (ScalingType.BlockWise1x32, QuantOrientation.NATURAL, SwizzleType.SWIZZLE_32_4_4):
        assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
        return mxfp8_swizzle_triton(input)
    if spec == (ScalingType.BlockWise1x32, QuantOrientation.TRANSPOSED, SwizzleType.NO_SWIZZLE):
        return mxfp8_dim_m_triton(input)
    if spec == (ScalingType.BlockWise1x32, QuantOrientation.TRANSPOSED, SwizzleType.SWIZZLE_32_4_4):
        return mxfp8_dim_m_swizzle_triton(input)
    if spec == (ScalingType.BlockWise32x32, QuantOrientation.NATURAL, SwizzleType.NO_SWIZZLE):
        assert input.shape[0] % 32 == 0, f"first dim must be a multiple of 32, got {input.shape[0]}"
        assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
        return mxfp8_32x32_triton(input)

    raise ValueError(
        f"unsupported (scaling_type, orientation, swizzle_type)={spec!r}; supported: "
        "(BlockWise1x32, NATURAL, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "(BlockWise1x32, TRANSPOSED, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "(BlockWise32x32, NATURAL, NO_SWIZZLE); "
        "for the fused dual-orientation cast use quantize_tensor_bidirectional"
    )


def quantize_tensor_bidirectional(
    input: Tensor,
    *,
    qdata_dtype: torch.dtype = torch.float8_e4m3fn,
    inner_scale_calc: InnerScaleCalc = InnerScaleCalc.E8M0_RCEIL,
    scaling_type: ScalingType = ScalingType.BlockWise1x32,
    swizzle_type: SwizzleType = SwizzleType.SWIZZLE_32_4_4,
    skip_transposed_qdata: bool = False,
    rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
    outer_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Fused dual-orientation cast: quantize `input` to a block-scaled low-precision format in BOTH
    the natural and transposed orientations in one pass. The single-orientation cast is
    `quantize_tensor`.

    Args:
      input: 2D input tensor (bf16 or fp32) of shape (M, K), or 3D of shape (E, M, K).
      qdata_dtype: qdata element format (only torch.float8_e4m3fn today).
      inner_scale_calc: per-block scale strategy -- fixes the scale dtype and the amax->scale
        computation (only InnerScaleCalc.E8M0_RCEIL today).
      scaling_type: only BlockWise1x32 and BlockWise32x32
      swizzle_type: NO_SWIZZLE or SWIZZLE_32_4_4. Note that for 3d inputs, swizzle is per-expert.
      skip_transposed_qdata: emit only the natural qdata but BOTH scales (no transposed qdata).
        Square scaling types only; needed on hardware (such as Blackwell) where the second argument
        of a scaled gemm can be row-major.
      rounding_mode: RTNE or STOCHASTIC
      random_key: entropy source for stochastic rounding

    Returns:
        4 tensors (qk, sk, qm, sm) normally -- natural (dim-K) pair then transposed (dim-M) pair
        3 tensors (qk, sk, sm) when skip_transposed_qdata is set
    """
    assert input.dim() in (2, 3), f"only 2D or 3D input supported, got {input.dim()}D"
    assert qdata_dtype == torch.float8_e4m3fn, f"only float8_e4m3fn qdata supported, got {qdata_dtype}"
    assert inner_scale_calc == InnerScaleCalc.E8M0_RCEIL, (
        f"only InnerScaleCalc.E8M0_RCEIL supported, got {inner_scale_calc!r}"
    )
    assert outer_scale is None, "outer_scale is not supported by quantize_tensor_bidirectional yet"
    if rounding_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("rounding_mode=STOCHASTIC is not implemented yet")
    if random_key is not None:
        raise NotImplementedError("random_key (stochastic rounding) is not implemented yet")

    if input.dim() == 3:
        # Per-expert (E, N, K) batching, quantized independently along N and K (never E).
        if skip_transposed_qdata:
            raise NotImplementedError(
                "skip_transposed_qdata is not supported for 3D (E, N, K) input"
            )
        if (scaling_type, swizzle_type) != (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}); "
                "quantize_tensor_bidirectional with 3D (E, N, K) input supports only "
                "(BlockWise1x32, SWIZZLE_32_4_4)"
            )
        q_nat, s_nat = mxfp8_f(input.contiguous())  # (E,N,K), 1x32 along K
        sb_nat = _to_blocked_per_group_3d(s_nat)
        q_t, s_t = mxfp8_f(input.transpose(-2, -1).contiguous())  # (E,K,N), 1x32 along N
        sb_t = _to_blocked_per_group_3d(s_t)
        return q_nat, sb_nat, q_t, sb_t

    assert input.is_contiguous(), "input must be contiguous"
    spec = (scaling_type, swizzle_type)
    if skip_transposed_qdata:
        if spec == (ScalingType.BlockWise32x32, SwizzleType.SWIZZLE_32_4_4):
            assert input.shape[0] % 32 == 0, f"first dim must be a multiple of 32, got {input.shape[0]}"
            assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
            return mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_triton(input)
        raise ValueError(
            f"unsupported (scaling_type, swizzle_type)={spec!r} for skip_transposed_qdata; "
            "supported: (BlockWise32x32, SWIZZLE_32_4_4)"
        )
    if spec == (ScalingType.BlockWise1x32, SwizzleType.NO_SWIZZLE):
        return mxfp8_dim_km_triton(input)
    if spec == (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        return mxfp8_dim_km_swizzle_triton(input)
    raise ValueError(
        f"unsupported (scaling_type, swizzle_type)={spec!r}; supported: "
        "(BlockWise1x32, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "or (BlockWise32x32, SWIZZLE_32_4_4) with skip_transposed_qdata"
    )


def quantize_tensor_grouped(
    input: Tensor,  # (total_M, C)
    offs: Tensor,
    *,
    qdata_dtype: torch.dtype = torch.float8_e4m3fn,
    inner_scale_calc: InnerScaleCalc = InnerScaleCalc.E8M0_RCEIL,
    scaling_type: ScalingType = ScalingType.BlockWise1x32,
    orientation: QuantOrientation = QuantOrientation.NATURAL,
    swizzle_type: SwizzleType = SwizzleType.SWIZZLE_32_4_4,
    rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
    outer_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Single-orientation grouped cast to a block-scaled low-precision format. For the fused
    dual-orientation cast use `quantize_tensor_grouped_bidirectional`.

    Differences from quantize_tensor:
    * 2d tensors of shape (M, K) only
    * adds an `offs` argument (each group's scales are swizzled independently)
    * swizzling is per-token-group

    Args:
      qdata_dtype: qdata element format (only torch.float8_e4m3fn today).
      inner_scale_calc: per-block scale strategy -- fixes the scale dtype and the amax->scale
        computation (only InnerScaleCalc.E8M0_RCEIL today).

    Token groups must already be block-aligned (each group's row count a multiple of 32); the caller
    is responsible for any token-group padding (see `_pad_token_groups`).
    """
    assert qdata_dtype == torch.float8_e4m3fn, f"only float8_e4m3fn qdata supported, got {qdata_dtype}"
    assert inner_scale_calc == InnerScaleCalc.E8M0_RCEIL, (
        f"only InnerScaleCalc.E8M0_RCEIL supported, got {inner_scale_calc!r}"
    )
    assert outer_scale is None, "outer_scale is not supported by quantize_tensor_grouped yet"
    if rounding_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("rounding_mode=STOCHASTIC is not implemented yet")
    if random_key is not None:
        raise NotImplementedError("random_key (stochastic rounding) is not implemented yet")
    if (scaling_type, swizzle_type) != (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        raise ValueError(
            f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}); "
            "quantize_tensor_grouped supports only (BlockWise1x32, SWIZZLE_32_4_4)"
        )
    if orientation not in (QuantOrientation.NATURAL, QuantOrientation.TRANSPOSED):
        raise ValueError(
            f"orientation={orientation!r} is not supported by quantize_tensor_grouped; "
            "supported: NATURAL, TRANSPOSED"
        )

    x = input.contiguous()
    if orientation == QuantOrientation.NATURAL:
        q, s = quantize_2d_act(x)  # (M, C), 1x32 along C
        sb = _to_blocked_2d_m_groups(s, offs)
    else:
        q, s = quantize_2d_act(x.transpose(-2, -1).contiguous())  # (C, M), 1x32 along M
        sb = _to_blocked_2d_k_groups(s, offs // BLOCK_SIZE)
    return q, sb


def quantize_tensor_grouped_bidirectional(
    input: Tensor,  # (total_M, C)
    offs: Tensor,
    *,
    qdata_dtype: torch.dtype = torch.float8_e4m3fn,
    inner_scale_calc: InnerScaleCalc = InnerScaleCalc.E8M0_RCEIL,
    scaling_type: ScalingType = ScalingType.BlockWise1x32,
    swizzle_type: SwizzleType = SwizzleType.SWIZZLE_32_4_4,
    skip_transposed_qdata: bool = False,
    rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
    outer_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fused dual-orientation grouped cast: quantize `input` to a block-scaled low-precision format in
    BOTH the natural and transposed orientations in one read. The single-orientation cast is
    `quantize_tensor_grouped`.

    `qdata_dtype` (only torch.float8_e4m3fn today) and `inner_scale_calc` (only
    InnerScaleCalc.E8M0_RCEIL today) select the format, as in `quantize_tensor`.

    Token groups must already be block-aligned (see `quantize_tensor_grouped`); the caller owns any
    token-group padding. Returns the natural (dim-K) pair then the transposed (dim-M) pair
    (q_nat, sb_nat, q_t, sb_t).
    """
    assert qdata_dtype == torch.float8_e4m3fn, f"only float8_e4m3fn qdata supported, got {qdata_dtype}"
    assert inner_scale_calc == InnerScaleCalc.E8M0_RCEIL, (
        f"only InnerScaleCalc.E8M0_RCEIL supported, got {inner_scale_calc!r}"
    )
    assert outer_scale is None, (
        "outer_scale is not supported by quantize_tensor_grouped_bidirectional yet"
    )
    if rounding_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("rounding_mode=STOCHASTIC is not implemented yet")
    if random_key is not None:
        raise NotImplementedError("random_key (stochastic rounding) is not implemented yet")
    if skip_transposed_qdata:
        # No 32x32 grouped kernel exists (the natural-qdata/both-scales path is dense-only).
        raise NotImplementedError(
            "skip_transposed_qdata is not supported by quantize_tensor_grouped_bidirectional"
        )
    if (scaling_type, swizzle_type) != (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        raise ValueError(
            f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}); "
            "quantize_tensor_grouped_bidirectional supports only (BlockWise1x32, SWIZZLE_32_4_4)"
        )

    x = input.contiguous()
    q_nat, s_nat = quantize_2d_act(x)  # (M, C), 1x32 along C
    sb_nat = _to_blocked_2d_m_groups(s_nat, offs)
    q_t, s_t = quantize_2d_act(x.transpose(-2, -1).contiguous())  # (C, M), 1x32 along M
    sb_t = _to_blocked_2d_k_groups(s_t, offs // BLOCK_SIZE)
    return q_nat, sb_nat, q_t, sb_t
