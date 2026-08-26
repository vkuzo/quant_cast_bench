from enum import IntEnum, StrEnum

import torch
import torch.func._random as prng
from torch import Tensor
from torch.nn.functional import SwizzleType  # core enum: NO_SWIZZLE=0, SWIZZLE_32_4_4=1

from quant_cast_bench.quantize_tensor_api.moe_utils import (
    BLOCK_SIZE,
    _to_blocked_2d_k_groups,
    _to_blocked_2d_m_groups,
    _to_blocked_per_group_3d,
    quantize_2d_act,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    mxfp4_f,
    mxfp8_f,
    nvfp4_gs_f,
    nvfp4_gs_swizzle_dim_k_dim_m_rht_f,
    nvfp4_gs_swizzle_dim_k_dim_m_rht_sr_f,
    nvfp4_gs_swizzle_dim_km_f,
    nvfp4_gs_swizzle_dim_m_f,
    nvfp4_gs_swizzle_dim_m_rht_f,
    nvfp4_gs_swizzle_dim_m_rht_sr_f,
    nvfp4_gs_swizzle_sr_f,
)
from quant_cast_bench.quant_cast_triton.recipes import (
    mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_triton,
    mxfp8_32x32_triton,
    mxfp8_dim_km_triton,
    mxfp8_dim_m_triton,
    mxfp8_triton,
    nvfp4_triton,
)


class ScalingType(IntEnum):
    # Mirrors torch.nn.functional.ScalingType (core, a pybind Enum), including its int values. In the
    # real/upstreamed version this should import and use core's enum directly rather than redefining
    # it here.
    TensorWise = 0
    RowWise = 1
    BlockWise1x16 = 2
    BlockWise1x32 = 3
    BlockWise1x128 = 4
    BlockWise128x128 = 5
    BlockWise32x32 = 6  # LOCAL addition, not in core; needed for first-class 32x32 mxfp8.


class RoundingMode(StrEnum):
    # how qdata values are rounded when cast into float8_e4m3fn.
    RTNE = "rtne"
    STOCHASTIC = "stochastic"


class InnerScaleCalc(StrEnum):
    # Defines how to get from (block of inputs, broadcasted chunks of outer scale)
    # to (scale, rcp_scale). Intentionally general to be able to cover
    # complicated use cases like 4over6, in case they mature enough to be upstreamed
    # to core.
    RCEIL_E8M0 = "rceil_e8m0"
    NVFP4_E4M3 = "nvfp4_e4m3"


def quantize_tensor(
    input: Tensor,
    *,
    qdata_dtype: torch.dtype,
    inner_scale_calc: InnerScaleCalc,
    scaling_type: ScalingType | list[ScalingType],
    swizzle_type: SwizzleType = SwizzleType.NO_SWIZZLE,
    qdata_rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
    outer_scale: Tensor | None = None,
    # TODO(future PR): more design on RHT, specifically:
    # 1. generalize size (today hardcodes 16x16)
    # 2. think through whether the input should be the sign vector or the RHT
    rht_tensor: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize `input` to a block-scaled low-precision format in one orientation: `qdata_dtype` qdata
    + one `inner_scale_calc` scale per block.

    For the fused dual-orientation cast (both the natural and transposed pairs in one pass), use
    `quantize_tensor_dual`.

    Args:
      input: 2D input tensor (bf16 or fp32) of shape (M, K), or 3D of shape (E, M, K).
      qdata_dtype: qdata element format -- torch.float8_e4m3fn (mxfp8) or torch.float4_e2m1fn_x2
        (nvfp4, 2D only).
      inner_scale_calc: per-block scale strategy -- fixes the scale dtype and the amax->scale
        computation. InnerScaleCalc.RCEIL_E8M0 (mxfp8) or InnerScaleCalc.NVFP4_E4M3 (nvfp4).
      scaling_type: single-level formats pass a bare ScalingType -- BlockWise1x32 / BlockWise32x32
        (mxfp8, mxfp4). Two-level nvfp4 passes [inner, outer]: [BlockWise1x16, TensorWise] for
        per-tensor, [BlockWise1x16, RowWise] for per-token. The outer level names the outer_scale
        broadcast directly (no shape guessing).

        The scaling axis follows the dims of the tensor you pass: for a (M, K) input with a 1x32
        block, the 1 maps to M and the 32 maps to K, so the scale runs along K (the "dim-k" cast).
        To scale along the other dim ("dim-m"), pass a transposed view -- input.t() for 2D,
        input.transpose(-2, -1) for 3D. The API detects the transpose, un-transposes it, and routes
        to the specialized dim-m cast; the outputs are written transposed-contiguous.
      swizzle_type: NO_SWIZZLE or SWIZZLE_32_4_4. Note that for 3d inputs, swizzle is applied per-expert.
      qdata_rounding_mode: RTNE or STOCHASTIC. STOCHASTIC is supported only by the per-tensor swizzled
        nvfp4 casts -- NATURAL (dim-k) and TRANSPOSED (dim-m, which then requires an rht_tensor) --
        and requires random_key; every other path is RTNE-only.
      random_key: SR entropy, a torch.func._random Philox key. Required when (and only when)
        qdata_rounding_mode is STOCHASTIC.
      outer_scale: precomputed fp32 outer scale, required for nvfp4 (float4_e2m1fn_x2) two-level
        scaling; must be None otherwise. A per-tensor scalar selects per-tensor nvfp4 (swizzled
        kernel); a per-token (M, 1) scale selects per-token nvfp4 (mapped to the gold reference,
        no swizzle).
      rht_tensor: optional 16x16 Random Hadamard Transform matrix. Only used by the per-tensor
        dim-m swizzled nvfp4 cast (selected by passing a transposed view), where it applies the RHT
        to the un-transposed input before quantizing (the wgrad-operand cast of nvfp4 training); must
        be None for every other path.

    Returns:
        2 tensors (qdata, scale)
    """
    assert input.dim() in (2, 3), f"only 2D or 3D input supported, got {input.dim()}D"
    # scaling_type is a bare ScalingType for single-level formats, or a two-element [inner, outer]
    # list for two-level nvfp4 (the outer level names the outer_scale broadcast: TensorWise=per-tensor,
    # RowWise=per-token).
    if isinstance(scaling_type, list):
        assert len(scaling_type) == 2, (
            "multi-level scaling_type must be [inner, outer], e.g. [BlockWise1x16, TensorWise]"
        )
        inner_scaling_type, outer_scaling_type = scaling_type
    else:
        inner_scaling_type, outer_scaling_type = scaling_type, None
    # SR is implemented only for the two nvfp4 swizzle casts below; every other path asserts RTNE
    # where it dispatches. random_key IS the SR entropy (a torch.func._random Philox key), so it and
    # STOCHASTIC must come together.
    if qdata_rounding_mode == RoundingMode.STOCHASTIC:
        assert random_key is not None, "qdata_rounding_mode=STOCHASTIC requires random_key (the SR Philox key)"
    elif random_key is not None:
        raise ValueError("random_key is only used with qdata_rounding_mode=STOCHASTIC")

    if qdata_dtype == torch.float4_e2m1fn_x2:
        assert input.dim() == 2, "fp4 quantization is only supported for 2D input"
        # dim-k (contiguous input, scale along the last dim) vs dim-m (a transposed view, scale along
        # the first dim): un-transpose the view and route to the specialized dim-m cast.
        if input.is_contiguous():
            is_dim_m, x = False, input
        else:
            x = input.transpose(-2, -1)  # un-transpose -> the original contiguous (M, K)
            assert x.is_contiguous(), (
                "input must be contiguous (dim-k), or a transpose of a contiguous tensor (dim-m)"
            )
            is_dim_m = True
        if inner_scale_calc == InnerScaleCalc.RCEIL_E8M0:
            assert not is_dim_m, "mxfp4 supports only the dim-k (contiguous input) cast"
            assert outer_scaling_type is None, "mxfp4 (RCEIL_E8M0) is single-level; pass a bare ScalingType"
            assert outer_scale is None, "mxfp4 (RCEIL_E8M0) is single-level; outer_scale must be None"
            assert rht_tensor is None, "rht_tensor is only supported by the dim-m nvfp4 cast"
            assert qdata_rounding_mode == RoundingMode.RTNE, "stochastic rounding is not supported for mxfp4"
            if (inner_scaling_type, swizzle_type) == (ScalingType.BlockWise1x32, SwizzleType.NO_SWIZZLE):
                assert x.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {x.shape[1]}"
                return mxfp4_f(x)
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}) for "
                "mxfp4 (float4_e2m1fn_x2, RCEIL_E8M0); supported: (BlockWise1x32, NO_SWIZZLE)"
            )
        assert inner_scale_calc == InnerScaleCalc.NVFP4_E4M3, (
            f"float4_e2m1fn_x2 qdata requires inner_scale_calc=NVFP4_E4M3 (nvfp4) or "
            f"RCEIL_E8M0 (mxfp4), got {inner_scale_calc!r}"
        )
        assert outer_scale is not None, "nvfp4 quantization requires a precomputed outer_scale"
        assert outer_scaling_type is not None, (
            "nvfp4 is two-level; pass scaling_type=[BlockWise1x16, TensorWise] (per-tensor) or "
            "[BlockWise1x16, RowWise] (per-token)"
        )
        assert inner_scaling_type == ScalingType.BlockWise1x16, (
            f"nvfp4 inner scaling_type must be BlockWise1x16, got {inner_scaling_type!r}"
        )
        # The outer scaling level names the outer_scale broadcast directly (no shape guessing):
        # RowWise = per-token (one fp32 value per row, (M, 1)), TensorWise = per-tensor (a scalar).
        if outer_scaling_type == ScalingType.RowWise:
            # Per-token: no Triton kernel yet, so map to the gold reference (`nvfp4_gs_f`, plain
            # row-major inner scale, no swizzle). dim-k only.
            assert not is_dim_m, "per-token (RowWise) nvfp4 supports only the dim-k (contiguous input) cast"
            assert outer_scale.shape == (x.shape[0], 1), (
                f"per-token (RowWise) nvfp4 outer_scale must be (M, 1)=({x.shape[0]}, 1), got "
                f"{tuple(outer_scale.shape)}"
            )
            assert rht_tensor is None, "rht_tensor is only supported by the dim-m nvfp4 cast"
            assert qdata_rounding_mode == RoundingMode.RTNE, "stochastic rounding is not supported for per-token nvfp4"
            if (inner_scaling_type, swizzle_type) == (ScalingType.BlockWise1x16, SwizzleType.NO_SWIZZLE):
                return nvfp4_gs_f(x, outer_scale)
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}) for "
                "per-token nvfp4 (float4_e2m1fn_x2, NVFP4_E4M3, RowWise outer); supported: "
                "(BlockWise1x16, NO_SWIZZLE)"
            )
        if outer_scaling_type == ScalingType.TensorWise:
            if not is_dim_m:
                # dim-k (natural) swizzled per-tensor nvfp4.
                if (inner_scaling_type, swizzle_type) == (ScalingType.BlockWise1x16, SwizzleType.SWIZZLE_32_4_4):
                    assert rht_tensor is None, "rht_tensor is only supported by the dim-m nvfp4 cast"
                    if qdata_rounding_mode == RoundingMode.STOCHASTIC:
                        # SR nvfp4 (Nvfp4GsSRSwizzleGold's nvfp4_gs_swizzle_sr_f): gold reference, no
                        # Triton kernel; random_key is its Philox key.
                        return nvfp4_gs_swizzle_sr_f(x, outer_scale, random_key)
                    return nvfp4_triton(x, outer_scale, swizzle=True)
                raise ValueError(
                    f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}) for "
                    "dim-k per-tensor nvfp4 (float4_e2m1fn_x2); supported: (BlockWise1x16, SWIZZLE_32_4_4)"
                )
            # dim-m per-tensor nvfp4 has no Triton kernel yet, so map it to the gold reference. RTNE
            # without an RHT is Nvfp4GsDimMSwizzleGold's nvfp4_gs_swizzle_dim_m_f (quantize x in 1x16
            # blocks along M, transposed (N, M//2) frame, swizzled e4m3 scale); RTNE with an RHT is
            # Nvfp4GsSwizzleDimMRHTGold's nvfp4_gs_swizzle_dim_m_rht_f (same, but RHT x.t() first -- the
            # wgrad-operand cast of nvfp4 training); STOCHASTIC is the SR twin of the latter
            # (Nvfp4GsDimMRHTSRSwizzleGold's nvfp4_gs_swizzle_dim_m_rht_sr_f) -- the only dim-m SR gold
            # is the RHT one, so SR requires an rht_tensor. The outer_scale must match: |x.t()| for the
            # no-RHT path, |RHT(x.t())| for the RHT paths (caller-set).
            if (inner_scaling_type, swizzle_type) == (ScalingType.BlockWise1x16, SwizzleType.SWIZZLE_32_4_4):
                if qdata_rounding_mode == RoundingMode.STOCHASTIC:
                    assert rht_tensor is not None, "stochastic dim-m nvfp4 requires an rht_tensor"
                    return nvfp4_gs_swizzle_dim_m_rht_sr_f(x, outer_scale, rht_tensor, random_key)
                if rht_tensor is None:
                    return nvfp4_gs_swizzle_dim_m_f(x, outer_scale)
                return nvfp4_gs_swizzle_dim_m_rht_f(x, outer_scale, rht_tensor)
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}) for "
                "dim-m per-tensor nvfp4 (float4_e2m1fn_x2); supported: (BlockWise1x16, SWIZZLE_32_4_4)"
            )
        raise ValueError(
            f"nvfp4 outer scaling_type must be TensorWise or RowWise, got {outer_scaling_type!r}"
        )

    # mxfp8: float8_e4m3fn qdata + e8m0 rceil inner scale; no outer scale.
    assert qdata_dtype == torch.float8_e4m3fn, (
        f"only float8_e4m3fn or float4_e2m1fn_x2 qdata supported, got {qdata_dtype}"
    )
    assert inner_scale_calc == InnerScaleCalc.RCEIL_E8M0, (
        f"only InnerScaleCalc.RCEIL_E8M0 supported for float8_e4m3fn, got {inner_scale_calc!r}"
    )
    assert outer_scaling_type is None, "mxfp8 is single-level; pass a bare ScalingType"
    assert outer_scale is None, "outer_scale is only used by nvfp4 (float4_e2m1fn_x2) quantization"
    assert rht_tensor is None, "rht_tensor is only used by nvfp4 (float4_e2m1fn_x2) quantization"
    assert qdata_rounding_mode == RoundingMode.RTNE, "stochastic rounding is not supported for mxfp8"

    if input.dim() == 3:
        # Per-expert (E, N, K) batching is a separate code path from the 2D casts below: each of the
        # E slices is quantized independently (scaling never applies to the E dimension), swizzling is
        # per-expert. The scale runs along the last dim of whatever view is passed, so the caller
        # picks the axis by transposing (E,N,K) <-> (E,K,N); there is no separate dim-m branch.
        if (inner_scaling_type, swizzle_type) != (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}); "
                "quantize_tensor with 3D (E, N, K) input supports only (BlockWise1x32, SWIZZLE_32_4_4)"
            )
        q, s = mxfp8_f(input.contiguous())  # 1x32 along the last dim
        return q, _to_blocked_per_group_3d(s)

    # 2D: dim-k (contiguous input, scale along the last dim) vs dim-m (a transposed view, scale along
    # the first dim). For dim-m un-transpose to the original contiguous tensor and use the specialized
    # dim-m kernel (numerically the same as casting the passed view along its last dim, but faster).
    spec = (inner_scaling_type, swizzle_type)
    if input.is_contiguous():
        if spec == (ScalingType.BlockWise1x32, SwizzleType.NO_SWIZZLE):
            assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
            return mxfp8_triton(input, swizzle=False)
        if spec == (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
            assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
            return mxfp8_triton(input, swizzle=True)
        if spec == (ScalingType.BlockWise32x32, SwizzleType.NO_SWIZZLE):
            assert input.shape[0] % 32 == 0, f"first dim must be a multiple of 32, got {input.shape[0]}"
            assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
            return mxfp8_32x32_triton(input, swizzle=False)
        raise ValueError(
            f"unsupported (scaling_type, swizzle_type)={spec!r} for the dim-k (contiguous input) "
            "mxfp8 cast; supported: (BlockWise1x32, NO_SWIZZLE|SWIZZLE_32_4_4), "
            "(BlockWise32x32, NO_SWIZZLE); for the fused dual-orientation cast use "
            "quantize_tensor_dual"
        )

    x = input.transpose(-2, -1)  # un-transpose -> the original contiguous (M, K)
    assert x.is_contiguous(), (
        "input must be contiguous (dim-k), or a transpose of a contiguous tensor (dim-m)"
    )
    if spec == (ScalingType.BlockWise1x32, SwizzleType.NO_SWIZZLE):
        return mxfp8_dim_m_triton(x, swizzle=False)
    if spec == (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        return mxfp8_dim_m_triton(x, swizzle=True)
    raise ValueError(
        f"unsupported (scaling_type, swizzle_type)={spec!r} for the dim-m (transposed input) mxfp8 "
        "cast; supported: (BlockWise1x32, NO_SWIZZLE|SWIZZLE_32_4_4)"
    )


def quantize_tensor_dual(
    input: Tensor,
    *,
    qdata_dtype: torch.dtype,
    inner_scale_calc: InnerScaleCalc,
    scaling_type: ScalingType | list[ScalingType],
    swizzle_type: SwizzleType = SwizzleType.NO_SWIZZLE,
    skip_transposed_qdata: bool = False,
    qdata_rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
    outer_scale: tuple[Tensor | None, Tensor | None] | None = None,
    rht_tensor: tuple[Tensor | None, Tensor | None] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Fused dual-orientation cast: quantize `input` to a block-scaled low-precision format in BOTH
    the natural and transposed orientations in one pass. The single-orientation cast is
    `quantize_tensor`.

    Args:
      input: 2D input tensor (bf16 or fp32) of shape (M, K), or 3D of shape (E, M, K). nvfp4 is 2D
        only.
      qdata_dtype: qdata element format -- torch.float8_e4m3fn (mxfp8) or torch.float4_e2m1fn_x2
        (nvfp4).
      inner_scale_calc: per-block scale strategy -- fixes the scale dtype and the amax->scale
        computation. InnerScaleCalc.RCEIL_E8M0 (mxfp8) or InnerScaleCalc.NVFP4_E4M3 (nvfp4).
      scaling_type: single-level formats pass a bare ScalingType -- BlockWise1x32 / BlockWise32x32
        (mxfp8). nvfp4 (per-tensor only here) passes [BlockWise1x16, TensorWise].
      swizzle_type: NO_SWIZZLE or SWIZZLE_32_4_4. Note that for 3d inputs, swizzle is per-expert.
      skip_transposed_qdata: emit only the natural qdata but BOTH scales (no transposed qdata).
        Square scaling types only; needed on hardware (such as Blackwell) where the second argument
        of a scaled gemm can be row-major. mxfp8 only.
      qdata_rounding_mode: RTNE or STOCHASTIC. STOCHASTIC is supported only by the RHT nvfp4 cast
        (rht_tensor=(None, rht)) -- the grad_output cast of nvfp4 training; mxfp8 and the plain
        no-RHT nvfp4 cast are RTNE only.
      random_key: Philox key for stochastic rounding (required when qdata_rounding_mode=STOCHASTIC, must
        be None otherwise). Split internally into one substream per orientation.
      outer_scale: nvfp4 per-tensor outer scale as a (dim_k, dim_m) tuple (required for nvfp4; must
        be None for mxfp8). Both orientations are always explicit -- there is no single-value form.
        Without an RHT the two are the same value (|input.t()| == |input|), so pass (os, os). With an
        RHT they differ (|input| for dim-k, |RHT(input.t())| for dim-m), so pass (dim_k, dim_m).
      rht_tensor: optional (dim_k, dim_m) tuple carrying a 16x16 Random Hadamard Transform matrix
        applied to the transposed (dim-m) cast (the wgrad-operand cast of nvfp4 training); dim-k never
        applies one. None for mxfp8 and for the plain (no-RHT) nvfp4 dim-km cast. Because the RHT is
        dim-m (second-operand) only, pass it as (None, rht) -- rht in the dim-k slot is rejected.

    Returns:
        4 tensors (qk, sk, qm, sm) normally -- natural (dim-K) pair then transposed (dim-M) pair
        3 tensors (qk, sk, sm) when skip_transposed_qdata is set
    """
    assert input.dim() in (2, 3), f"only 2D or 3D input supported, got {input.dim()}D"
    # scaling_type is a bare ScalingType for single-level mxfp8, or a two-element [inner, outer] list
    # for two-level nvfp4 (per-tensor only here, so outer must be TensorWise).
    if isinstance(scaling_type, list):
        assert len(scaling_type) == 2, (
            "multi-level scaling_type must be [inner, outer], e.g. [BlockWise1x16, TensorWise]"
        )
        inner_scaling_type, outer_scaling_type = scaling_type
    else:
        inner_scaling_type, outer_scaling_type = scaling_type, None
    # stochastic rounding and its entropy source are coupled: STOCHASTIC needs a random_key, and a
    # random_key is only meaningful under STOCHASTIC.
    if qdata_rounding_mode == RoundingMode.STOCHASTIC:
        assert random_key is not None, "qdata_rounding_mode=STOCHASTIC requires a random_key"
    else:
        assert random_key is None, "random_key is only used with qdata_rounding_mode=STOCHASTIC"
    # outer_scale / rht_tensor are per-orientation: a (dim_k, dim_m) tuple sets the natural (dim-k)
    # and transposed (dim-m) casts independently (or None for neither). Both directions are always
    # explicit -- there is no single-value form that applies to both.
    assert outer_scale is None or isinstance(outer_scale, tuple), (
        "outer_scale must be a (dim_k, dim_m) tuple (or None); pass both orientations explicitly, "
        "e.g. outer_scale=(os, os) when they share a value"
    )
    assert rht_tensor is None or isinstance(rht_tensor, tuple), (
        "rht_tensor must be a (dim_k, dim_m) tuple (or None); pass both orientations explicitly, "
        "e.g. rht_tensor=(None, rht)"
    )
    outer_scale_k, outer_scale_m = outer_scale if outer_scale is not None else (None, None)
    rht_tensor_k, rht_tensor_m = rht_tensor if rht_tensor is not None else (None, None)

    if qdata_dtype == torch.float4_e2m1fn_x2:
        # Fused dual nvfp4 cast: dim-k is plain nvfp4 over |input|; dim-m nvfp4s input.t()
        # along the original M. No Triton kernel yet, so both map to gold references.
        assert input.dim() == 2, "fp4 quantization is only supported for 2D input"
        assert input.is_contiguous(), "input must be contiguous"
        assert inner_scale_calc == InnerScaleCalc.NVFP4_E4M3, (
            f"float4_e2m1fn_x2 qdata requires inner_scale_calc=NVFP4_E4M3 (nvfp4), got {inner_scale_calc!r}"
        )
        assert not skip_transposed_qdata, "skip_transposed_qdata is not supported for nvfp4"
        assert outer_scaling_type is not None, (
            "nvfp4 is two-level; pass scaling_type=[BlockWise1x16, TensorWise]"
        )
        assert inner_scaling_type == ScalingType.BlockWise1x16, (
            f"nvfp4 inner scaling_type must be BlockWise1x16, got {inner_scaling_type!r}"
        )
        assert outer_scaling_type == ScalingType.TensorWise, (
            "dual nvfp4 is per-tensor (TensorWise) only"
        )
        spec = (inner_scaling_type, swizzle_type)
        if spec != (ScalingType.BlockWise1x16, SwizzleType.SWIZZLE_32_4_4):
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)={spec!r} for nvfp4 "
                "(float4_e2m1fn_x2, NVFP4_E4M3); supported: (BlockWise1x16, SWIZZLE_32_4_4)"
            )
        if rht_tensor_k is None and rht_tensor_m is None:
            # No RHT (Nvfp4GsDimKMSwizzleGold's nvfp4_gs_swizzle_dim_km_f): both orientations are
            # plain nvfp4, each with its own per-tensor outer scale. With no RHT |input.t()| ==
            # |input|, so callers typically pass the same value for both (outer_scale=(os, os)).
            assert outer_scale_k is not None and outer_scale_m is not None, (
                "no-RHT nvfp4 quantize_tensor_dual requires an outer_scale per orientation "
                "(a (dim_k, dim_m) tuple)"
            )
            # SR only exists for the RHT (grad_output) cast of nvfp4 training; the plain no-RHT
            # dim-km cast (activation/weight) is RTNE.
            assert qdata_rounding_mode == RoundingMode.RTNE, (
                "stochastic rounding is only supported by the RHT nvfp4 cast (rht_tensor=(None, rht))"
            )
            return nvfp4_gs_swizzle_dim_km_f(input, outer_scale_k, outer_scale_m)
        # WITH RHT (Nvfp4GsSwizzle_DimK_DimMRHT_Gold's nvfp4_gs_swizzle_dim_k_dim_m_rht_f): dim-m
        # applies the RHT to input.t() before quantizing (the wgrad-operand cast of nvfp4 training).
        # The two orientations now need DIFFERENT outer scales (|input| vs |RHT(input.t())|), so pass
        # outer_scale=(dim_k, dim_m). The RHT is dim-m (second-operand) ONLY, so it must be passed as
        # the (None, rht) tuple form -- a bare rht_tensor=rht (which would apply to both operands) is
        # rejected.
        assert rht_tensor_k is None and rht_tensor_m is not None, (
            "nvfp4 quantize_tensor_dual applies the RHT to the dim-m (second) operand only; "
            "pass rht_tensor=(None, rht), not a bare rht_tensor=rht"
        )
        assert outer_scale_k is not None and outer_scale_m is not None, (
            "RHT nvfp4 quantize_tensor_dual requires an outer_scale per orientation "
            "(a (dim_k, dim_m) tuple)"
        )
        if qdata_rounding_mode == RoundingMode.STOCHASTIC:
            # grad_output cast of nvfp4 training: dim-k (no RHT) and dim-m (RHT) both round
            # stochastically, each with its own Philox substream. The API takes one random_key;
            # split it into (key_k, key_m) so the two casts get uncorrelated dither -- bit-identical
            # to a caller doing prng.split(key, 2) itself.
            key_k, key_m = prng.split(random_key, 2)
            return nvfp4_gs_swizzle_dim_k_dim_m_rht_sr_f(
                input, outer_scale_k, outer_scale_m, rht_tensor_m, key_k, key_m
            )
        return nvfp4_gs_swizzle_dim_k_dim_m_rht_f(input, outer_scale_k, outer_scale_m, rht_tensor_m)

    # mxfp8: float8_e4m3fn qdata + e8m0 rceil inner scale; outer_scale / rht_tensor unused.
    assert qdata_dtype == torch.float8_e4m3fn, (
        f"only float8_e4m3fn or float4_e2m1fn_x2 qdata supported, got {qdata_dtype}"
    )
    assert inner_scale_calc == InnerScaleCalc.RCEIL_E8M0, (
        f"only InnerScaleCalc.RCEIL_E8M0 supported for float8_e4m3fn, got {inner_scale_calc!r}"
    )
    assert outer_scaling_type is None, "mxfp8 is single-level; pass a bare ScalingType"
    assert outer_scale_k is None and outer_scale_m is None, (
        "outer_scale is only used by nvfp4 (float4_e2m1fn_x2) quantization"
    )
    assert rht_tensor_k is None and rht_tensor_m is None, (
        "rht_tensor is only used by nvfp4 (float4_e2m1fn_x2) quantization"
    )
    assert qdata_rounding_mode == RoundingMode.RTNE, "stochastic rounding is not supported for mxfp8"

    if input.dim() == 3:
        # Per-expert (E, N, K) batching, quantized independently along N and K (never E).
        if skip_transposed_qdata:
            raise NotImplementedError(
                "skip_transposed_qdata is not supported for 3D (E, N, K) input"
            )
        if (inner_scaling_type, swizzle_type) != (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}); "
                "quantize_tensor_dual with 3D (E, N, K) input supports only "
                "(BlockWise1x32, SWIZZLE_32_4_4)"
            )
        q_nat, s_nat = mxfp8_f(input.contiguous())  # (E,N,K), 1x32 along K
        sb_nat = _to_blocked_per_group_3d(s_nat)
        q_t, s_t = mxfp8_f(input.transpose(-2, -1).contiguous())  # (E,K,N), 1x32 along N
        sb_t = _to_blocked_per_group_3d(s_t)
        return q_nat, sb_nat, q_t, sb_t

    assert input.is_contiguous(), "input must be contiguous"
    spec = (inner_scaling_type, swizzle_type)
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
        return mxfp8_dim_km_triton(input, swizzle=False)
    if spec == (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        return mxfp8_dim_km_triton(input, swizzle=True)
    raise ValueError(
        f"unsupported (scaling_type, swizzle_type)={spec!r}; supported: "
        "(BlockWise1x32, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "or (BlockWise32x32, SWIZZLE_32_4_4) with skip_transposed_qdata"
    )


def quantize_tensor_grouped(
    input: Tensor,  # (total_M, C)
    offs: Tensor,
    *,
    qdata_dtype: torch.dtype,
    inner_scale_calc: InnerScaleCalc,
    scaling_type: ScalingType | list[ScalingType],
    swizzle_type: SwizzleType = SwizzleType.NO_SWIZZLE,
    qdata_rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
    outer_scale: Tensor | None = None,
    rht_tensor: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Single-orientation grouped cast to a block-scaled low-precision format. For the fused
    dual-orientation cast use `quantize_tensor_grouped_dual`.

    The scaling axis follows the dims of the passed tensor: a contiguous `(M, C)` input casts 1x32
    along the last dim (dim-k); passing a transposed view (`input.t()`) selects the dim-m cast (1x32
    along M). The transpose is detected and un-transposed internally, so both routes hit the same
    swizzle recipe as before.

    Differences from quantize_tensor:
    * 2d tensors of shape (M, K) only
    * adds an `offs` argument (each group's scales are swizzled independently)
    * swizzling is per-token-group

    Args:
      qdata_dtype: qdata element format (only torch.float8_e4m3fn today).
      inner_scale_calc: per-block scale strategy -- fixes the scale dtype and the amax->scale
        computation (only InnerScaleCalc.RCEIL_E8M0 today).

    Token groups must already be block-aligned (each group's row count a multiple of 32); the caller
    is responsible for any token-group padding (see `_pad_token_groups`).
    """
    assert qdata_dtype == torch.float8_e4m3fn, f"only float8_e4m3fn qdata supported, got {qdata_dtype}"
    assert inner_scale_calc == InnerScaleCalc.RCEIL_E8M0, (
        f"only InnerScaleCalc.RCEIL_E8M0 supported, got {inner_scale_calc!r}"
    )
    assert outer_scale is None, "outer_scale is not supported by quantize_tensor_grouped yet"
    assert rht_tensor is None, "rht_tensor is not supported by quantize_tensor_grouped yet"
    assert not isinstance(scaling_type, list), (
        "quantize_tensor_grouped is single-level (mxfp8); pass a bare ScalingType"
    )
    inner_scaling_type = scaling_type
    if qdata_rounding_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("qdata_rounding_mode=STOCHASTIC is not implemented yet")
    if random_key is not None:
        raise NotImplementedError("random_key (stochastic rounding) is not implemented yet")
    if (inner_scaling_type, swizzle_type) != (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        raise ValueError(
            f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}); "
            "quantize_tensor_grouped supports only (BlockWise1x32, SWIZZLE_32_4_4)"
        )
    if not input.is_contiguous():
        assert input.transpose(-2, -1).is_contiguous(), (
            "input must be contiguous (dim-k), or a transpose of a contiguous tensor (dim-m)"
        )

    # A contiguous input casts along C (dim-k, M-groups scale); a transposed view un-transposes to the
    # original contiguous (C, M) buffer and casts along M (dim-m, K-groups scale).
    q, s = quantize_2d_act(input.contiguous())
    sb = (
        _to_blocked_2d_m_groups(s, offs) if input.is_contiguous()
        else _to_blocked_2d_k_groups(s, offs // BLOCK_SIZE)
    )
    return q, sb


def quantize_tensor_grouped_dual(
    input: Tensor,  # (total_M, C)
    offs: Tensor,
    *,
    qdata_dtype: torch.dtype,
    inner_scale_calc: InnerScaleCalc,
    scaling_type: ScalingType | list[ScalingType],
    swizzle_type: SwizzleType = SwizzleType.NO_SWIZZLE,
    skip_transposed_qdata: bool = False,
    qdata_rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
    # outer_scale / rht_tensor: one value applied to BOTH orientations, or a (dim_k, dim_m) tuple to
    # set the natural (dim-k) and transposed (dim-m) casts independently. Not wired to a kernel yet.
    outer_scale: Tensor | None | tuple[Tensor | None, Tensor | None] = None,
    rht_tensor: Tensor | None | tuple[Tensor | None, Tensor | None] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fused dual-orientation grouped cast: quantize `input` to a block-scaled low-precision format in
    BOTH the natural and transposed orientations in one read. The single-orientation cast is
    `quantize_tensor_grouped`.

    `qdata_dtype` (only torch.float8_e4m3fn today) and `inner_scale_calc` (only
    InnerScaleCalc.RCEIL_E8M0 today) select the format, as in `quantize_tensor`.

    Token groups must already be block-aligned (see `quantize_tensor_grouped`); the caller owns any
    token-group padding. Returns the natural (dim-K) pair then the transposed (dim-M) pair
    (q_nat, sb_nat, q_t, sb_t).

    `outer_scale` and `rht_tensor` each take either one value (applied to BOTH orientations) or a
    (dim_k, dim_m) tuple to set the natural (dim-k) and transposed (dim-m) casts independently;
    neither is wired to a kernel yet.
    """
    assert qdata_dtype == torch.float8_e4m3fn, f"only float8_e4m3fn qdata supported, got {qdata_dtype}"
    assert inner_scale_calc == InnerScaleCalc.RCEIL_E8M0, (
        f"only InnerScaleCalc.RCEIL_E8M0 supported, got {inner_scale_calc!r}"
    )
    # outer_scale / rht_tensor take either one value (applied to BOTH orientations) or a
    # (dim_k, dim_m) tuple to set the natural (dim-k) and transposed (dim-m) casts independently.
    # Normalize to the (dim_k, dim_m) tuple form here; kernels are not wired to these yet.
    outer_scale_k, outer_scale_m = (
        outer_scale if isinstance(outer_scale, tuple) else (outer_scale, outer_scale)
    )
    rht_tensor_k, rht_tensor_m = (
        rht_tensor if isinstance(rht_tensor, tuple) else (rht_tensor, rht_tensor)
    )
    assert outer_scale_k is None and outer_scale_m is None, (
        "outer_scale is not supported by quantize_tensor_grouped_dual yet"
    )
    assert rht_tensor_k is None and rht_tensor_m is None, (
        "rht_tensor is not supported by quantize_tensor_grouped_dual yet"
    )
    assert not isinstance(scaling_type, list), (
        "quantize_tensor_grouped_dual is single-level (mxfp8); pass a bare ScalingType"
    )
    inner_scaling_type = scaling_type
    if qdata_rounding_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("qdata_rounding_mode=STOCHASTIC is not implemented yet")
    if random_key is not None:
        raise NotImplementedError("random_key (stochastic rounding) is not implemented yet")
    if skip_transposed_qdata:
        # No 32x32 grouped kernel exists (the natural-qdata/both-scales path is dense-only).
        raise NotImplementedError(
            "skip_transposed_qdata is not supported by quantize_tensor_grouped_dual"
        )
    if (inner_scaling_type, swizzle_type) != (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        raise ValueError(
            f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}); "
            "quantize_tensor_grouped_dual supports only (BlockWise1x32, SWIZZLE_32_4_4)"
        )

    x = input.contiguous()
    q_nat, s_nat = quantize_2d_act(x)  # (M, C), 1x32 along C
    sb_nat = _to_blocked_2d_m_groups(s_nat, offs)
    q_t, s_t = quantize_2d_act(x.transpose(-2, -1).contiguous())  # (C, M), 1x32 along M
    sb_t = _to_blocked_2d_k_groups(s_t, offs // BLOCK_SIZE)
    return q_nat, sb_nat, q_t, sb_t
