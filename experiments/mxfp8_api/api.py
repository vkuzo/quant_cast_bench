from enum import IntEnum, StrEnum

from torch import Tensor
from torch.nn.functional import SwizzleType  # core enum: NO_SWIZZLE=0, SWIZZLE_32_4_4=1

from quant_cast_bench.quant_cast_triton.recipes import (
    mxfp8_32x32_triton,
    mxfp8_dim_km_swizzle_triton,
    mxfp8_dim_km_triton,
    mxfp8_dim_m_swizzle_triton,
    mxfp8_dim_m_triton,
    mxfp8_swizzle_triton,
    mxfp8_triton,
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
    # always contiguous. For asymmetric blocks (1x32), NATURAL vs TRANSPOSED are genuinely different
    # quantizations (rowwise vs colwise -- different elements share an amax); for a square block
    # (32x32) they are the same values in a transposed layout. BOTH emits both in one fused pass.
    NATURAL = "natural"          # block maps to (M, N) as given
    TRANSPOSED = "transposed"    # block maps to the (N, M) view; qdata/scale written transposed-contig
    BOTH = "both"                # fused: emit the NATURAL pair, then the TRANSPOSED pair (4 outputs)


class RoundingMode(StrEnum):
    # how qdata values are rounded when cast into float8_e4m3fn.
    RTNE = "rtne"                # round to nearest, ties to even
    STOCHASTIC = "stochastic"    # stochastic rounding


def quantize_to_mxfp8(
    input: Tensor,
    scaling_type: ScalingType = ScalingType.BlockWise1x32,
    orientation: QuantOrientation = QuantOrientation.NATURAL,
    swizzle_type: SwizzleType = SwizzleType.NO_SWIZZLE,
    pad_input_to_next_multiple_of: tuple[int, int] | None = None,
    rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """Quantize `input` to mxfp8: float8_e4m3fn qdata + one e8m0 scale per block.

    Args:
      input: 2D contiguous input tensor (bf16 or fp32) of shape (M, N).
      scaling_type: the scale BLOCK SIZE only (mirrors PyTorch core's ScalingType); it does NOT encode
        which tensor dims the block maps to. Wired today: BlockWise1x32 and BlockWise32x32.
      orientation: how the block maps onto (M, N) and how the output is laid out (always contiguous).
        NATURAL maps the block to (M, N) as given; TRANSPOSED maps it to the (N, M) view and writes the
        outputs transposed-contiguous; BOTH runs both in one fused pass. For asymmetric blocks (1x32)
        NATURAL vs TRANSPOSED are different quantizations (rowwise vs colwise).
      swizzle_type: scale storage layout applied to every scale this cast produces. NO_SWIZZLE is a
        plain 2D scale; SWIZZLE_32_4_4 is the NVIDIA 32x4x4 blocked layout.
      pad_input_to_next_multiple_of: per-input-dimension zero-padding fused into the cast, e.g.
        (128, 32) rounds each of input's two dims up to that multiple so the outputs satisfy
        torch._scaled_mm's alignment. NOT IMPLEMENTED YET -- specifying anything other than None raises.
      rounding_mode: how qdata values are rounded into float8_e4m3fn. RTNE (default) is supported;
        STOCHASTIC is NOT IMPLEMENTED YET and raises.
      random_key: entropy source for stochastic rounding. NOT IMPLEMENTED YET -- passing a Tensor
        raises; must be None.

    Returns 2 tensors (qdata, scale) for NATURAL/TRANSPOSED, or 4 tensors (qk, sk, qm, sm) for BOTH.
    Supported (scaling_type, orientation, swizzle_type) combinations:
      * BlockWise1x32,  NATURAL,    NO_SWIZZLE / SWIZZLE_32_4_4  -> (q (M,N),  s)
      * BlockWise1x32,  TRANSPOSED, NO_SWIZZLE / SWIZZLE_32_4_4  -> (q (N,M),  s)
      * BlockWise1x32,  BOTH,       NO_SWIZZLE / SWIZZLE_32_4_4  -> (qk (M,N), sk, qm (N,M), sm)
      * BlockWise32x32, NATURAL,    NO_SWIZZLE                   -> (q (M,N),  s (M//32,N//32))
    scale tensors are float8_e8m0fnu. Any other combination raises ValueError.
    """
    assert input.dim() == 2, f"only 2D input supported for now, got {input.dim()}D"
    assert input.is_contiguous(), "input must be contiguous"
    if pad_input_to_next_multiple_of is not None:
        raise NotImplementedError("pad_input_to_next_multiple_of is not implemented yet")
    if rounding_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("rounding_mode=STOCHASTIC is not implemented yet")
    if random_key is not None:
        raise NotImplementedError("random_key (stochastic rounding) is not implemented yet")

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
    if spec == (ScalingType.BlockWise1x32, QuantOrientation.BOTH, SwizzleType.NO_SWIZZLE):
        return mxfp8_dim_km_triton(input)
    if spec == (ScalingType.BlockWise1x32, QuantOrientation.BOTH, SwizzleType.SWIZZLE_32_4_4):
        return mxfp8_dim_km_swizzle_triton(input)
    if spec == (ScalingType.BlockWise32x32, QuantOrientation.NATURAL, SwizzleType.NO_SWIZZLE):
        assert input.shape[0] % 32 == 0, f"first dim must be a multiple of 32, got {input.shape[0]}"
        assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
        return mxfp8_32x32_triton(input)

    raise ValueError(
        f"unsupported (scaling_type, orientation, swizzle_type)={spec!r}; supported: "
        "(BlockWise1x32, NATURAL, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "(BlockWise1x32, TRANSPOSED, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "(BlockWise1x32, BOTH, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "(BlockWise32x32, NATURAL, NO_SWIZZLE)"
    )
