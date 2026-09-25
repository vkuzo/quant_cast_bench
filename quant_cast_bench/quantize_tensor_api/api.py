from enum import IntEnum, StrEnum

import torch
import torch.func._random as prng
from torch import Tensor
from torch.nn.functional import SwizzleType  # core enum: NO_SWIZZLE=0, SWIZZLE_32_4_4=1

from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma.blockscaled_tma_impl import (
    mxfp4,
    mxfp4_dim_km_swizzle_v2,
    mxfp4_dim_m_swizzle_v2,
    mxfp4_swizzle_v2,
    mxfp8,
    mxfp8_32x32_swizzle_v2,
    mxfp8_swizzle_v2,
    nvfp4,
    nvfp4_dim_km_swizzle_tma,
    nvfp4_dim_m_swizzle_tma,
    nvfp4_swizzle_16x16_tma,
    nvfp4_swizzle_tma,
)
from quant_cast_bench.quant_cast_cute_hand.nvfp4_pipelined.nvfp4_pipelined_impl import (
    nvfp4_dim_m_rht_swizzle_pipelined,
    nvfp4_swizzle_dim_k_dim_m_rht_pipelined,
)
from quant_cast_bench.quantize_tensor_api.moe_utils import (
    BLOCK_SIZE,
    _to_blocked_2d_k_groups,
    _to_blocked_2d_m_groups,
    quantize_2d_act,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    mxfp4_dim_km_swizzle_f,
    mxfp4_dim_m_swizzle_f,
    mxfp4_f,
    mxfp4_swizzle_f,
    mxfp8_dim_km_swizzle_sr_f,
    mxfp8_dim_m_swizzle_sr_f,
    mxfp8_swizzle_sr_f,
    nvfp4_gs_16x16_swizzle_f,
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


def _can_use_blockscaled_tma(input: Tensor, quant_orientation: str) -> bool:
    """Whether the input satisfies the common CuTe-hand blockscaled TMA launch contract."""
    if (
        input.device.type != "cuda"
        or torch.cuda.get_device_capability(input.device) < (10, 0)
        or input.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not input.is_contiguous()
        or input.data_ptr() % 16 != 0
    ):
        return False
    M, K = input.shape
    if M > 2**31 - 1 or K > 2**31 - 1:
        return False
    if quant_orientation == "dim_k":
        return K % 32 == 0
    if quant_orientation == "dim_m":
        return M % 32 == 0 and K % 16 == 0
    if quant_orientation == "dim_km":
        return M % 32 == 0 and K % 32 == 0
    raise ValueError(f"unsupported quant_orientation: {quant_orientation}")


def _can_use_nvfp4_rht_pipelined(input: Tensor, rht_sign: Tensor) -> bool:
    """Whether the input satisfies the BF16, full-tile CuTe-hand RHT launch contract."""
    M, K = input.shape
    return (
        input.device.type == "cuda"
        and torch.cuda.get_device_capability(input.device) >= (10, 0)
        and input.dtype == torch.bfloat16
        and input.is_contiguous()
        and input.data_ptr() % 16 == 0
        and M > 0
        and K > 0
        and M % 128 == 0
        and K % 128 == 0
        and isinstance(rht_sign, torch.Tensor)
        and rht_sign.shape == (16,)
        and rht_sign.dtype == torch.bfloat16
        and rht_sign.device == input.device
        and rht_sign.is_contiguous()
    )


def quantize_tensor(
    input: Tensor,
    *,
    qdata_dtype: torch.dtype,
    inner_scale_calc: InnerScaleCalc,
    scaling_type: ScalingType | list[ScalingType],
    swizzle_type: SwizzleType = SwizzleType.NO_SWIZZLE,
    qdata_rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
    outer_quant_scale: Tensor | None = None,
    # TODO(future PR): generalize the RHT size (today it hardcodes a length-16 sign vector).
    rht_tensor: Tensor | None = None,
    scaling_type_square_block_and_expand: bool = False,
) -> tuple[Tensor, Tensor]:
    """Quantize `input` to a block-scaled low-precision format in one orientation: `qdata_dtype` qdata
    + one `inner_scale_calc` scale per block.

    For the fused dual-orientation cast (both the natural and transposed pairs in one pass), use
    `quantize_tensor_dual`.

    Args:
      input: 2D input tensor (bf16 or fp32) of shape (M, K).
      qdata_dtype: qdata element format -- torch.float8_e4m3fn (mxfp8) or torch.float4_e2m1fn_x2
        (nvfp4).
      inner_scale_calc: per-block scale strategy -- fixes the scale dtype and the amax->scale
        computation. InnerScaleCalc.RCEIL_E8M0 (mxfp8) or InnerScaleCalc.NVFP4_E4M3 (nvfp4).
      scaling_type: single-level formats pass a bare ScalingType -- BlockWise1x32 (mxfp8, mxfp4).
        Two-level nvfp4 passes [inner, outer]: [BlockWise1x16, TensorWise] for per-tensor,
        [BlockWise1x16, RowWise] for per-token. The outer level names the outer_quant_scale broadcast
        directly (no shape guessing).

        The scaling axis follows the dims of the tensor you pass: for a (M, K) input with a 1x32
        block, the 1 maps to M and the 32 maps to K, so the scale runs along K (the "dim-k" cast).
        To scale along the other dim ("dim-m"), pass a transposed view -- input.t(). The API detects
        the transpose, un-transposes it, and routes to the specialized dim-m cast; the outputs are
        written transposed-contiguous.
      swizzle_type: NO_SWIZZLE or SWIZZLE_32_4_4.
      qdata_rounding_mode: RTNE or STOCHASTIC. STOCHASTIC is supported by swizzled mxfp8 and by
        per-tensor swizzled nvfp4. The nvfp4 TRANSPOSED path also requires an rht_tensor.
      random_key: SR entropy, a torch.func._random Philox key. Required when (and only when)
        qdata_rounding_mode is STOCHASTIC.
      outer_quant_scale: precomputed fp32 outer QUANT scale, i.e. low = high * outer_quant_scale.
        Its value is 1/S where S = amax/(fp8_max*fp4_max) is the dequant multiplier F.scaled_mm
        consumes (the MSLK/torchao global_scale convention), so the caller reciprocates S -> 1/S
        before passing. Required for nvfp4 (float4_e2m1fn_x2) two-level scaling; must be None
        otherwise. A per-tensor scalar selects per-tensor nvfp4 (swizzled kernel); a per-token (M, 1)
        scale selects per-token nvfp4 (mapped to the gold reference, no swizzle).
      rht_tensor: optional length-16 sign vector defining the Random Hadamard Transform. Only used by the per-tensor
        dim-m swizzled nvfp4 cast (selected by passing a transposed view), where it applies the RHT
        to the un-transposed input before quantizing (the wgrad-operand cast of nvfp4 training); must
        be None for every other path.
      scaling_type_square_block_and_expand: compute one scale per square block and expand it into
        the row-blocked layout the GEMM consumes: 32x32 -> 1x32 for mxfp8, or 16x16 -> 1x16 for
        nvfp4. Only dim-k is supported.

    Returns:
        2 tensors (qdata, scale)
    """
    assert input.dim() == 2, f"only 2D input supported, got {input.dim()}D"
    # scaling_type is a bare ScalingType for single-level formats, or a two-element [inner, outer]
    # list for two-level nvfp4 (the outer level names the outer_quant_scale broadcast: TensorWise=per-tensor,
    # RowWise=per-token).
    if isinstance(scaling_type, list):
        assert len(scaling_type) == 2, (
            "multi-level scaling_type must be [inner, outer], e.g. [BlockWise1x16, TensorWise]"
        )
        inner_scaling_type, outer_scaling_type = scaling_type
    else:
        inner_scaling_type, outer_scaling_type = scaling_type, None
    # scaling_type_square_block_and_expand computes the scale over 32x32 square blocks and expands it
    # into the row-blocked layout the GEMM consumes: 1x32 for MXFP8, or 1x16 for NVFP4.
    if scaling_type_square_block_and_expand:
        expanded_scaling_type = (
            ScalingType.BlockWise1x16
            if inner_scale_calc == InnerScaleCalc.NVFP4_E4M3
            else ScalingType.BlockWise1x32
        )
        assert inner_scaling_type == expanded_scaling_type, (
            "scaling_type_square_block_and_expand requires scaling_type="
            f"{expanded_scaling_type!r}, got "
            f"{inner_scaling_type!r}"
        )
    # random_key is the SR entropy (a torch.func._random Philox key), so it and STOCHASTIC must come
    # together. Individual recipe branches below reject SR when they do not implement it.
    if qdata_rounding_mode == RoundingMode.STOCHASTIC:
        assert random_key is not None, "qdata_rounding_mode=STOCHASTIC requires random_key (the SR Philox key)"
    elif random_key is not None:
        raise ValueError("random_key is only used with qdata_rounding_mode=STOCHASTIC")

    if qdata_dtype == torch.float4_e2m1fn_x2:
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
            assert not scaling_type_square_block_and_expand, (
                "scaling_type_square_block_and_expand is not supported for mxfp4"
            )
            assert outer_scaling_type is None, "mxfp4 (RCEIL_E8M0) is single-level; pass a bare ScalingType"
            assert outer_quant_scale is None, "mxfp4 (RCEIL_E8M0) is single-level; outer_quant_scale must be None"
            assert rht_tensor is None, "rht_tensor is only supported by the dim-m nvfp4 cast"
            assert qdata_rounding_mode == RoundingMode.RTNE, "stochastic rounding is not supported for mxfp4"
            spec = (inner_scaling_type, swizzle_type)
            if not is_dim_m and spec == (
                ScalingType.BlockWise1x32,
                SwizzleType.NO_SWIZZLE,
            ):
                assert x.shape[1] % 32 == 0, (
                    f"last dim must be a multiple of 32, got {x.shape[1]}"
                )
                return mxfp4(x) if _can_use_blockscaled_tma(x, "dim_k") else mxfp4_f(x)
            if spec == (
                ScalingType.BlockWise1x32,
                SwizzleType.SWIZZLE_32_4_4,
            ):
                quant_orientation = "dim_m" if is_dim_m else "dim_k"
                if _can_use_blockscaled_tma(x, quant_orientation):
                    if is_dim_m:
                        return mxfp4_dim_m_swizzle_v2(x)
                    return mxfp4_swizzle_v2(x)
                if is_dim_m:
                    return mxfp4_dim_m_swizzle_f(x)
                return mxfp4_swizzle_f(x)
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}) for "
                "mxfp4 (float4_e2m1fn_x2, RCEIL_E8M0); supported: dim-k "
                "(BlockWise1x32, NO_SWIZZLE|SWIZZLE_32_4_4), dim-m "
                "(BlockWise1x32, SWIZZLE_32_4_4)"
            )
        assert inner_scale_calc == InnerScaleCalc.NVFP4_E4M3, (
            f"float4_e2m1fn_x2 qdata requires inner_scale_calc=NVFP4_E4M3 (nvfp4) or "
            f"RCEIL_E8M0 (mxfp4), got {inner_scale_calc!r}"
        )
        assert outer_quant_scale is not None, "nvfp4 quantization requires a precomputed outer_quant_scale"
        assert outer_scaling_type is not None, (
            "nvfp4 is two-level; pass scaling_type=[BlockWise1x16, TensorWise] (per-tensor) or "
            "[BlockWise1x16, RowWise] (per-token)"
        )
        assert inner_scaling_type == ScalingType.BlockWise1x16, (
            f"nvfp4 inner scaling_type must be BlockWise1x16, got {inner_scaling_type!r}"
        )
        # The outer scaling level names the outer_quant_scale broadcast directly (no shape guessing):
        # RowWise = per-token (one fp32 value per row, (M, 1)), TensorWise = per-tensor (a scalar).
        if outer_scaling_type == ScalingType.RowWise:
            # Per-token: no Triton kernel yet, so map to the gold reference (`nvfp4_gs_f`, plain
            # row-major inner scale, no swizzle). dim-k only.
            assert not is_dim_m, "per-token (RowWise) nvfp4 supports only the dim-k (contiguous input) cast"
            assert not scaling_type_square_block_and_expand, (
                "square-block scaling is not supported for per-token nvfp4"
            )
            assert outer_quant_scale.shape == (x.shape[0], 1), (
                f"per-token (RowWise) nvfp4 outer_quant_scale must be (M, 1)=({x.shape[0]}, 1), got "
                f"{tuple(outer_quant_scale.shape)}"
            )
            assert rht_tensor is None, "rht_tensor is only supported by the dim-m nvfp4 cast"
            assert qdata_rounding_mode == RoundingMode.RTNE, "stochastic rounding is not supported for per-token nvfp4"
            if (inner_scaling_type, swizzle_type) == (ScalingType.BlockWise1x16, SwizzleType.NO_SWIZZLE):
                return nvfp4_gs_f(x, outer_quant_scale)
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}) for "
                "per-token nvfp4 (float4_e2m1fn_x2, NVFP4_E4M3, RowWise outer); supported: "
                "(BlockWise1x16, NO_SWIZZLE)"
            )
        if outer_scaling_type == ScalingType.TensorWise:
            if scaling_type_square_block_and_expand:
                assert not is_dim_m, "16x16 nvfp4 supports only the dim-k cast"
                assert rht_tensor is None, "16x16 nvfp4 does not support RHT"
                assert qdata_rounding_mode == RoundingMode.RTNE, (
                    "16x16 nvfp4 supports only RTNE"
                )
                if (inner_scaling_type, swizzle_type) == (
                    ScalingType.BlockWise1x16,
                    SwizzleType.SWIZZLE_32_4_4,
                ):
                    assert x.shape[0] % 16 == 0, (
                        f"first dim must be a multiple of 16, got {x.shape[0]}"
                    )
                    assert x.shape[1] % 32 == 0, (
                        f"last dim must be a multiple of 32, got {x.shape[1]}"
                    )
                    if _can_use_blockscaled_tma(x, "dim_k"):
                        return nvfp4_swizzle_16x16_tma(x, outer_quant_scale)
                    return nvfp4_gs_16x16_swizzle_f(x, outer_quant_scale)
                raise ValueError(
                    "16x16 nvfp4 requires (BlockWise1x16, SWIZZLE_32_4_4)"
                )
            if not is_dim_m:
                # dim-k (natural) per-tensor nvfp4.
                if (inner_scaling_type, swizzle_type) == (
                    ScalingType.BlockWise1x16,
                    SwizzleType.NO_SWIZZLE,
                ):
                    assert rht_tensor is None, (
                        "rht_tensor is only supported by the dim-m nvfp4 cast"
                    )
                    assert qdata_rounding_mode == RoundingMode.RTNE, (
                        "unswizzled nvfp4 supports only RTNE"
                    )
                    if _can_use_blockscaled_tma(x, "dim_k"):
                        return nvfp4(x, outer_quant_scale)
                    return nvfp4_gs_f(x, outer_quant_scale)
                if (inner_scaling_type, swizzle_type) == (ScalingType.BlockWise1x16, SwizzleType.SWIZZLE_32_4_4):
                    assert rht_tensor is None, "rht_tensor is only supported by the dim-m nvfp4 cast"
                    if qdata_rounding_mode == RoundingMode.STOCHASTIC:
                        # SR nvfp4 (Nvfp4GsSwizzlePortableSRGold's nvfp4_gs_swizzle_sr_f): gold
                        # reference, no
                        # Triton kernel; random_key is its Philox key.
                        return nvfp4_gs_swizzle_sr_f(x, outer_quant_scale, random_key)
                    if _can_use_blockscaled_tma(x, "dim_k"):
                        return nvfp4_swizzle_tma(x, outer_quant_scale)
                    return nvfp4_triton(x, outer_quant_scale, swizzle=True)
                raise ValueError(
                    f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}) for "
                    "dim-k per-tensor nvfp4 (float4_e2m1fn_x2); supported: "
                    "(BlockWise1x16, NO_SWIZZLE|SWIZZLE_32_4_4)"
                )
            # Dim-m per-tensor NVFP4 uses CuTe-hand TMA for plain RTNE and the pipelined kernel for
            # eligible full-tile BF16 RHT RTNE inputs. Other architectures/shapes retain the gold
            # fallback. STOCHASTIC names the portable-SR gold, which is numerically distinct from
            # the pipelined kernel's NVIDIA cvt.rs recipe, so it deliberately remains a reference.
            # The outer_quant_scale must match |x.t()| without RHT or |RHT(x.t())| with RHT.
            if (inner_scaling_type, swizzle_type) == (ScalingType.BlockWise1x16, SwizzleType.SWIZZLE_32_4_4):
                if qdata_rounding_mode == RoundingMode.STOCHASTIC:
                    assert rht_tensor is not None, "stochastic dim-m nvfp4 requires an rht_tensor"
                    return nvfp4_gs_swizzle_dim_m_rht_sr_f(x, outer_quant_scale, rht_tensor, random_key)
                if rht_tensor is None:
                    if _can_use_blockscaled_tma(x, "dim_m"):
                        return nvfp4_dim_m_swizzle_tma(x, outer_quant_scale)
                    return nvfp4_gs_swizzle_dim_m_f(x, outer_quant_scale)
                if _can_use_nvfp4_rht_pipelined(x, rht_tensor):
                    return nvfp4_dim_m_rht_swizzle_pipelined(
                        x, outer_quant_scale, rht_tensor
                    )
                return nvfp4_gs_swizzle_dim_m_rht_f(x, outer_quant_scale, rht_tensor)
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
    assert outer_quant_scale is None, "outer_quant_scale is only used by nvfp4 (float4_e2m1fn_x2) quantization"
    assert rht_tensor is None, "rht_tensor is only used by nvfp4 (float4_e2m1fn_x2) quantization"

    if scaling_type_square_block_and_expand:
        # 32x32 square-block mxfp8: one scale per 32x32 tile, expanded into the 1x32 layout the gemm
        # consumes. Only the dim-k (contiguous input) cast is supported.
        assert qdata_rounding_mode == RoundingMode.RTNE, "32x32 mxfp8 supports only RTNE"
        if input.is_contiguous() and swizzle_type in (
            SwizzleType.NO_SWIZZLE,
            SwizzleType.SWIZZLE_32_4_4,
        ):
            assert input.shape[0] % 32 == 0, f"first dim must be a multiple of 32, got {input.shape[0]}"
            assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
            if (
                swizzle_type == SwizzleType.SWIZZLE_32_4_4
                and _can_use_blockscaled_tma(input, "dim_k")
            ):
                return mxfp8_32x32_swizzle_v2(input)
            return mxfp8_32x32_triton(
                input, swizzle=swizzle_type == SwizzleType.SWIZZLE_32_4_4
            )
        raise ValueError(
            "scaling_type_square_block_and_expand (32x32 mxfp8) supports only the dim-k "
            "(contiguous input) cast"
        )

    # 2D: dim-k (contiguous input, scale along the last dim) vs dim-m (a transposed view, scale along
    # the first dim). For dim-m un-transpose to the original contiguous tensor and use the specialized
    # dim-m kernel (numerically the same as casting the passed view along its last dim, but faster).
    spec = (inner_scaling_type, swizzle_type)
    if input.is_contiguous():
        if spec == (ScalingType.BlockWise1x32, SwizzleType.NO_SWIZZLE):
            assert qdata_rounding_mode == RoundingMode.RTNE, (
                "unswizzled mxfp8 supports only RTNE"
            )
            assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
            if _can_use_blockscaled_tma(input, "dim_k"):
                return mxfp8(input)
            return mxfp8_triton(input, swizzle=False)
        if spec == (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
            assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
            if _can_use_blockscaled_tma(input, "dim_k"):
                return mxfp8_swizzle_v2(
                    input,
                    key=random_key,
                    rounding_mode=qdata_rounding_mode,
                )
            if qdata_rounding_mode == RoundingMode.STOCHASTIC:
                return mxfp8_swizzle_sr_f(input, random_key)
            return mxfp8_triton(input, swizzle=True)
        raise ValueError(
            f"unsupported (scaling_type, swizzle_type)={spec!r} for the dim-k (contiguous input) "
            "mxfp8 cast; supported: (BlockWise1x32, NO_SWIZZLE|SWIZZLE_32_4_4); for the 32x32 "
            "square-block cast pass scaling_type_square_block_and_expand=True; for the fused "
            "dual-orientation cast use quantize_tensor_dual"
        )

    x = input.transpose(-2, -1)  # un-transpose -> the original contiguous (M, K)
    assert x.is_contiguous(), (
        "input must be contiguous (dim-k), or a transpose of a contiguous tensor (dim-m)"
    )
    if spec == (ScalingType.BlockWise1x32, SwizzleType.NO_SWIZZLE):
        assert qdata_rounding_mode == RoundingMode.RTNE, (
            "unswizzled dim-m mxfp8 supports only RTNE"
        )
        return mxfp8_dim_m_triton(x, swizzle=False)
    if spec == (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        if _can_use_blockscaled_tma(x, "dim_m"):
            return mxfp8_swizzle_v2(
                x,
                quant_orientation="dim_m",
                key=random_key,
                rounding_mode=qdata_rounding_mode,
            )
        if qdata_rounding_mode == RoundingMode.STOCHASTIC:
            return mxfp8_dim_m_swizzle_sr_f(x, random_key)
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
    outer_quant_scale: tuple[Tensor | None, Tensor | None] | None = None,
    rht_tensor: tuple[Tensor | None, Tensor | None] | None = None,
    scaling_type_square_block_and_expand: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Fused dual-orientation cast: quantize `input` to a block-scaled low-precision format in BOTH
    the natural and transposed orientations in one pass. The single-orientation cast is
    `quantize_tensor`.

    Args:
      input: 2D input tensor (bf16 or fp32) of shape (M, K).
      qdata_dtype: qdata element format -- torch.float8_e4m3fn (mxfp8) or torch.float4_e2m1fn_x2
        (nvfp4).
      inner_scale_calc: per-block scale strategy -- fixes the scale dtype and the amax->scale
        computation. InnerScaleCalc.RCEIL_E8M0 (mxfp8) or InnerScaleCalc.NVFP4_E4M3 (nvfp4).
      scaling_type: single-level formats pass a bare ScalingType -- BlockWise1x32 (mxfp8). nvfp4
        (per-tensor only here) passes [BlockWise1x16, TensorWise].
      swizzle_type: NO_SWIZZLE or SWIZZLE_32_4_4.
      skip_transposed_qdata: emit only the natural qdata but BOTH scales (no transposed qdata).
        Needed on hardware (such as Blackwell) where the second argument of a scaled gemm can be
        row-major. mxfp8 only, and only with scaling_type_square_block_and_expand=True +
        swizzle_type=SWIZZLE_32_4_4 (the 32x32 square-block cast).
      qdata_rounding_mode: RTNE or STOCHASTIC. STOCHASTIC is supported by swizzled mxfp8 and the RHT
        nvfp4 cast (rht_tensor=(None, rht_sign)); plain no-RHT nvfp4 is RTNE-only.
      random_key: Philox key for stochastic rounding (required when qdata_rounding_mode=STOCHASTIC, must
        be None otherwise). Split internally into one substream per orientation.
      outer_quant_scale: nvfp4 per-tensor outer QUANT scale, i.e. low = high * outer_quant_scale.
        Its value is 1/S where S = amax/(fp8_max*fp4_max) is the dequant multiplier F.scaled_mm
        consumes (the MSLK/torchao global_scale convention), so the caller reciprocates S -> 1/S
        before passing. Given as a (dim_k, dim_m) tuple (required for nvfp4; must be None for mxfp8). Both orientations are always
        explicit -- there is no single-value form. Without an RHT the two are the same value
        (|input.t()| == |input|), so pass (os, os). With an RHT they differ (|input| for dim-k,
        |RHT(input.t())| for dim-m), so pass (dim_k, dim_m).
      rht_tensor: optional (dim_k, dim_m) tuple carrying a length-16 sign vector defining the RHT
        applied to the transposed (dim-m) cast (the wgrad-operand cast of nvfp4 training); dim-k never
        applies one. None for mxfp8 and for the plain (no-RHT) nvfp4 dim-km cast. Because the RHT is
        dim-m (second-operand) only, pass it as (None, rht) -- rht in the dim-k slot is rejected.
      scaling_type_square_block_and_expand: mxfp8 only. When True, compute one scale per 32x32
        square block and expand it into the 1x32 layout the gemm consumes (requires
        scaling_type=BlockWise1x32). Only wired with skip_transposed_qdata=True +
        swizzle_type=SWIZZLE_32_4_4.

    Returns:
        4 tensors (qk, sk, qm, sm) normally -- natural (dim-K) pair then transposed (dim-M) pair
        3 tensors (qk, sk, sm) when skip_transposed_qdata is set
    """
    assert input.dim() == 2, f"only 2D input supported, got {input.dim()}D"
    # scaling_type is a bare ScalingType for single-level mxfp8, or a two-element [inner, outer] list
    # for two-level nvfp4 (per-tensor only here, so outer must be TensorWise).
    if isinstance(scaling_type, list):
        assert len(scaling_type) == 2, (
            "multi-level scaling_type must be [inner, outer], e.g. [BlockWise1x16, TensorWise]"
        )
        inner_scaling_type, outer_scaling_type = scaling_type
    else:
        inner_scaling_type, outer_scaling_type = scaling_type, None
    # scaling_type_square_block_and_expand computes a square-block scale and expands it into the
    # row-blocked layout the GEMM consumes: 1x32 for MXFP8, or 1x16 for NVFP4.
    if scaling_type_square_block_and_expand:
        expanded_scaling_type = (
            ScalingType.BlockWise1x16
            if inner_scale_calc == InnerScaleCalc.NVFP4_E4M3
            else ScalingType.BlockWise1x32
        )
        assert inner_scaling_type == expanded_scaling_type, (
            "scaling_type_square_block_and_expand requires scaling_type="
            f"{expanded_scaling_type!r}, got "
            f"{inner_scaling_type!r}"
        )
    # stochastic rounding and its entropy source are coupled: STOCHASTIC needs a random_key, and a
    # random_key is only meaningful under STOCHASTIC.
    if qdata_rounding_mode == RoundingMode.STOCHASTIC:
        assert random_key is not None, "qdata_rounding_mode=STOCHASTIC requires a random_key"
    else:
        assert random_key is None, "random_key is only used with qdata_rounding_mode=STOCHASTIC"
    # outer_quant_scale / rht_tensor are per-orientation: a (dim_k, dim_m) tuple sets the natural (dim-k)
    # and transposed (dim-m) casts independently (or None for neither). Both directions are always
    # explicit -- there is no single-value form that applies to both.
    assert outer_quant_scale is None or isinstance(outer_quant_scale, tuple), (
        "outer_quant_scale must be a (dim_k, dim_m) tuple (or None); pass both orientations explicitly, "
        "e.g. outer_quant_scale=(os, os) when they share a value"
    )
    assert rht_tensor is None or isinstance(rht_tensor, tuple), (
        "rht_tensor must be a (dim_k, dim_m) tuple (or None); pass both orientations explicitly, "
        "e.g. rht_tensor=(None, rht_sign)"
    )
    outer_quant_scale_k, outer_quant_scale_m = outer_quant_scale if outer_quant_scale is not None else (None, None)
    rht_tensor_k, rht_tensor_m = rht_tensor if rht_tensor is not None else (None, None)

    if qdata_dtype == torch.float4_e2m1fn_x2:
        assert input.is_contiguous(), "input must be contiguous"
        if inner_scale_calc == InnerScaleCalc.RCEIL_E8M0:
            assert outer_scaling_type is None, (
                "mxfp4 is single-level; pass a bare ScalingType"
            )
            assert outer_quant_scale_k is None and outer_quant_scale_m is None, (
                "mxfp4 does not use outer_quant_scale"
            )
            assert rht_tensor_k is None and rht_tensor_m is None, (
                "mxfp4 does not support RHT"
            )
            assert not skip_transposed_qdata, (
                "skip_transposed_qdata is not supported for mxfp4"
            )
            assert not scaling_type_square_block_and_expand, (
                "square-block scaling is not supported for mxfp4"
            )
            assert qdata_rounding_mode == RoundingMode.RTNE, (
                "mxfp4 supports only RTNE"
            )
            spec = (inner_scaling_type, swizzle_type)
            if spec == (
                ScalingType.BlockWise1x32,
                SwizzleType.SWIZZLE_32_4_4,
            ):
                if _can_use_blockscaled_tma(input, "dim_km"):
                    return mxfp4_dim_km_swizzle_v2(input)
                return mxfp4_dim_km_swizzle_f(input)
            raise ValueError(
                f"unsupported (scaling_type, swizzle_type)={spec!r} for dual mxfp4; "
                "supported: (BlockWise1x32, SWIZZLE_32_4_4)"
            )

        # Fused dual NVFP4 cast: dim-k is plain NVFP4 over |input|; dim-m quantizes input.t()
        # along the original M. Eligible RTNE inputs use CuTe-hand; unsupported shapes/devices and
        # the numerically distinct portable-SR recipe retain their gold fallbacks.
        assert inner_scale_calc == InnerScaleCalc.NVFP4_E4M3, (
            f"float4_e2m1fn_x2 qdata requires inner_scale_calc=NVFP4_E4M3 (nvfp4), got {inner_scale_calc!r}"
        )
        assert not skip_transposed_qdata, "skip_transposed_qdata is not supported for nvfp4"
        assert not scaling_type_square_block_and_expand, (
            "scaling_type_square_block_and_expand is not supported for dual nvfp4"
        )
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
            # |input|, so callers typically pass the same value for both (outer_quant_scale=(os, os)).
            assert outer_quant_scale_k is not None and outer_quant_scale_m is not None, (
                "no-RHT nvfp4 quantize_tensor_dual requires an outer_quant_scale per orientation "
                "(a (dim_k, dim_m) tuple)"
            )
            # SR only exists for the RHT (grad_output) cast of nvfp4 training; the plain no-RHT
            # dim-km cast (activation/weight) is RTNE.
            assert qdata_rounding_mode == RoundingMode.RTNE, (
                "stochastic rounding is only supported by the RHT nvfp4 cast "
                "(rht_tensor=(None, rht_sign))"
            )
            if _can_use_blockscaled_tma(input, "dim_km"):
                return nvfp4_dim_km_swizzle_tma(
                    input, outer_quant_scale_k, outer_quant_scale_m
                )
            return nvfp4_gs_swizzle_dim_km_f(input, outer_quant_scale_k, outer_quant_scale_m)
        # WITH RHT (Nvfp4GsSwizzle_DimK_DimMRHT_Gold's nvfp4_gs_swizzle_dim_k_dim_m_rht_f): dim-m
        # applies the RHT to input.t() before quantizing (the wgrad-operand cast of nvfp4 training).
        # The two orientations now need DIFFERENT outer scales (|input| vs |RHT(input.t())|), so pass
        # outer_quant_scale=(dim_k, dim_m). The RHT is dim-m (second-operand) ONLY, so it must be passed as
        # the (None, rht_sign) tuple form -- a bare sign vector (which would apply to both operands) is
        # rejected.
        assert rht_tensor_k is None and rht_tensor_m is not None, (
            "nvfp4 quantize_tensor_dual applies the RHT to the dim-m (second) operand only; "
            "pass rht_tensor=(None, rht_sign), not a bare sign vector"
        )
        assert outer_quant_scale_k is not None and outer_quant_scale_m is not None, (
            "RHT nvfp4 quantize_tensor_dual requires an outer_quant_scale per orientation "
            "(a (dim_k, dim_m) tuple)"
        )
        if qdata_rounding_mode == RoundingMode.STOCHASTIC:
            # grad_output cast of nvfp4 training: dim-k (no RHT) and dim-m (RHT) both round
            # stochastically, each with its own Philox substream. The API takes one random_key;
            # split it into (key_k, key_m) so the two casts get uncorrelated dither -- bit-identical
            # to a caller doing prng.split(key, 2) itself.
            key_k, key_m = prng.split(random_key, 2)
            return nvfp4_gs_swizzle_dim_k_dim_m_rht_sr_f(
                input, outer_quant_scale_k, outer_quant_scale_m, rht_tensor_m, key_k, key_m
            )
        if _can_use_nvfp4_rht_pipelined(input, rht_tensor_m):
            return nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
                input,
                outer_quant_scale_k,
                outer_quant_scale_m,
                rht_tensor_m,
            )
        return nvfp4_gs_swizzle_dim_k_dim_m_rht_f(input, outer_quant_scale_k, outer_quant_scale_m, rht_tensor_m)

    # mxfp8: float8_e4m3fn qdata + e8m0 rceil inner scale; outer_quant_scale / rht_tensor unused.
    assert qdata_dtype == torch.float8_e4m3fn, (
        f"only float8_e4m3fn or float4_e2m1fn_x2 qdata supported, got {qdata_dtype}"
    )
    assert inner_scale_calc == InnerScaleCalc.RCEIL_E8M0, (
        f"only InnerScaleCalc.RCEIL_E8M0 supported for float8_e4m3fn, got {inner_scale_calc!r}"
    )
    assert outer_scaling_type is None, "mxfp8 is single-level; pass a bare ScalingType"
    assert outer_quant_scale_k is None and outer_quant_scale_m is None, (
        "outer_quant_scale is only used by nvfp4 (float4_e2m1fn_x2) quantization"
    )
    assert rht_tensor_k is None and rht_tensor_m is None, (
        "rht_tensor is only used by nvfp4 (float4_e2m1fn_x2) quantization"
    )

    assert input.is_contiguous(), "input must be contiguous"
    spec = (inner_scaling_type, swizzle_type)
    if skip_transposed_qdata:
        # skip_transposed_qdata (natural qdata, both scales) is wired only for the 32x32 square-block
        # cast (expanded into the swizzled 1x32 layout).
        if scaling_type_square_block_and_expand and swizzle_type == SwizzleType.SWIZZLE_32_4_4:
            assert qdata_rounding_mode == RoundingMode.RTNE, (
                "32x32 mxfp8 supports only RTNE"
            )
            assert input.shape[0] % 32 == 0, f"first dim must be a multiple of 32, got {input.shape[0]}"
            assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
            return mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_triton(input)
        raise ValueError(
            "skip_transposed_qdata requires scaling_type_square_block_and_expand=True with "
            "swizzle_type=SWIZZLE_32_4_4 (the 32x32 square-block mxfp8 cast)"
        )
    if scaling_type_square_block_and_expand:
        # The full both-orientation 32x32 cast (transposed qdata too) is expressible but unwired.
        raise ValueError(
            "scaling_type_square_block_and_expand (32x32 mxfp8) is only wired with "
            "skip_transposed_qdata=True"
        )
    if spec == (ScalingType.BlockWise1x32, SwizzleType.NO_SWIZZLE):
        assert qdata_rounding_mode == RoundingMode.RTNE, (
            "unswizzled dual mxfp8 supports only RTNE"
        )
        return mxfp8_dim_km_triton(input, swizzle=False)
    if spec == (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        if _can_use_blockscaled_tma(input, "dim_km"):
            return mxfp8_swizzle_v2(
                input,
                quant_orientation="dim_km",
                key=random_key,
                rounding_mode=qdata_rounding_mode,
            )
        if qdata_rounding_mode == RoundingMode.STOCHASTIC:
            return mxfp8_dim_km_swizzle_sr_f(input, random_key)
        return mxfp8_dim_km_triton(input, swizzle=True)
    raise ValueError(
        f"unsupported (scaling_type, swizzle_type)={spec!r}; supported: "
        "(BlockWise1x32, NO_SWIZZLE|SWIZZLE_32_4_4), or the 32x32 square-block cast "
        "(scaling_type_square_block_and_expand=True, SWIZZLE_32_4_4, skip_transposed_qdata=True)"
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
    outer_quant_scale: Tensor | None = None,
    rht_tensor: Tensor | None = None,
    scaling_type_square_block_and_expand: bool = False,
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
    assert outer_quant_scale is None, "outer_quant_scale is not supported by quantize_tensor_grouped yet"
    assert rht_tensor is None, "rht_tensor is not supported by quantize_tensor_grouped yet"
    assert not isinstance(scaling_type, list), (
        "quantize_tensor_grouped is single-level (mxfp8); pass a bare ScalingType"
    )
    inner_scaling_type = scaling_type
    if scaling_type_square_block_and_expand:
        # No grouped 32x32 kernel exists (the square-block cast is dense-only).
        raise NotImplementedError(
            "scaling_type_square_block_and_expand (32x32) is not supported by quantize_tensor_grouped"
        )
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
    # outer_quant_scale / rht_tensor: one value applied to BOTH orientations, or a (dim_k, dim_m) tuple to
    # set the natural (dim-k) and transposed (dim-m) casts independently. Not wired to a kernel yet.
    outer_quant_scale: Tensor | None | tuple[Tensor | None, Tensor | None] = None,
    rht_tensor: Tensor | None | tuple[Tensor | None, Tensor | None] = None,
    scaling_type_square_block_and_expand: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fused dual-orientation grouped cast: quantize `input` to a block-scaled low-precision format in
    BOTH the natural and transposed orientations in one read. The single-orientation cast is
    `quantize_tensor_grouped`.

    `qdata_dtype` (only torch.float8_e4m3fn today) and `inner_scale_calc` (only
    InnerScaleCalc.RCEIL_E8M0 today) select the format, as in `quantize_tensor`.

    Token groups must already be block-aligned (see `quantize_tensor_grouped`); the caller owns any
    token-group padding. Returns the natural (dim-K) pair then the transposed (dim-M) pair
    (q_nat, sb_nat, q_t, sb_t).

    `outer_quant_scale` and `rht_tensor` each take either one value (applied to BOTH orientations) or a
    (dim_k, dim_m) tuple to set the natural (dim-k) and transposed (dim-m) casts independently;
    neither is wired to a kernel yet.
    """
    assert qdata_dtype == torch.float8_e4m3fn, f"only float8_e4m3fn qdata supported, got {qdata_dtype}"
    assert inner_scale_calc == InnerScaleCalc.RCEIL_E8M0, (
        f"only InnerScaleCalc.RCEIL_E8M0 supported, got {inner_scale_calc!r}"
    )
    # outer_quant_scale / rht_tensor take either one value (applied to BOTH orientations) or a
    # (dim_k, dim_m) tuple to set the natural (dim-k) and transposed (dim-m) casts independently.
    # Normalize to the (dim_k, dim_m) tuple form here; kernels are not wired to these yet.
    outer_quant_scale_k, outer_quant_scale_m = (
        outer_quant_scale if isinstance(outer_quant_scale, tuple) else (outer_quant_scale, outer_quant_scale)
    )
    rht_tensor_k, rht_tensor_m = (
        rht_tensor if isinstance(rht_tensor, tuple) else (rht_tensor, rht_tensor)
    )
    assert outer_quant_scale_k is None and outer_quant_scale_m is None, (
        "outer_quant_scale is not supported by quantize_tensor_grouped_dual yet"
    )
    assert rht_tensor_k is None and rht_tensor_m is None, (
        "rht_tensor is not supported by quantize_tensor_grouped_dual yet"
    )
    assert not isinstance(scaling_type, list), (
        "quantize_tensor_grouped_dual is single-level (mxfp8); pass a bare ScalingType"
    )
    inner_scaling_type = scaling_type
    if scaling_type_square_block_and_expand:
        # No grouped 32x32 kernel exists (the square-block cast is dense-only).
        raise NotImplementedError(
            "scaling_type_square_block_and_expand (32x32) is not supported by "
            "quantize_tensor_grouped_dual"
        )
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
