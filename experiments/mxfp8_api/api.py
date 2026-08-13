from enum import StrEnum

from torch import Tensor

from quant_cast_bench.quant_cast_triton.recipes import (
    mxfp8_32x32_triton,
    mxfp8_dim_km_swizzle_triton,
    mxfp8_dim_km_triton,
    mxfp8_dim_m_swizzle_triton,
    mxfp8_dim_m_triton,
    mxfp8_swizzle_triton,
    mxfp8_triton,
)


class OutputKindPair(StrEnum):
    # one value per block, applied to BOTH tensors of that block's (qdata, scale) pair.
    NORMAL = "normal"                  # written as-is: qdata (M, N), scale (M, N//32)
    TRANSP_CONTIG = "transp_contig"    # transposed then made contiguous: (N, M) / (N, M//32)


class SwizzleType(StrEnum):
    # scale storage layout. one value per block; applies to that block's scale tensor.
    NO_SWIZZLE = "no_swizzle"            # plain 2D row-major scale
    SWIZZLE_32_4_4 = "swizzle_32_4_4"    # NVIDIA 32x4x4 blocked scale layout


class RoundingMode(StrEnum):
    # how qdata values are rounded when cast into float8_e4m3fn.
    RTNE = "rtne"                # round to nearest, ties to even
    STOCHASTIC = "stochastic"    # stochastic rounding


def quantize_to_mxfp8(
    input: Tensor,
    block_size: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (1, 32),
    output_kind_pair: OutputKindPair | tuple[OutputKindPair, OutputKindPair] = OutputKindPair.NORMAL,
    swizzle_type: SwizzleType | tuple[SwizzleType, SwizzleType] = SwizzleType.NO_SWIZZLE,
    pad_input_to_next_multiple_of: tuple[int, int] | None = None,
    rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """Quantize `input` to mxfp8: float8_e4m3fn qdata + one e8m0 scale per block.

    Args:
      input: 2D contiguous input tensor (bf16 or fp32).
      block_size: the scale tile in absolute (M-extent, N-extent) terms, either a single block
        e.g. (1, 32) or a tuple of blocks e.g. ((1, 32), (32, 1)) for a fused multi-way cast.
      output_kind_pair: one OutputKindPair per block, applied to both tensors (qdata, scale) of that
        block. A single block takes one OutputKindPair; two blocks take a tuple of two.
      swizzle_type: one SwizzleType per block, selecting that block's scale storage layout. A single
        block takes one SwizzleType; two blocks take a tuple of two (which today must match each other).
      pad_input_to_next_multiple_of: per-input-dimension zero-padding fused into the cast, e.g.
        (128, 32) rounds each of input's two dims up to that multiple so the outputs satisfy
        torch._scaled_mm's alignment. NOT IMPLEMENTED YET -- specifying anything other than None raises.
      rounding_mode: how qdata values are rounded into float8_e4m3fn. RTNE (default) is supported;
        STOCHASTIC is NOT IMPLEMENTED YET and raises.
      random_key: entropy source for stochastic rounding. NOT IMPLEMENTED YET -- passing a Tensor
        raises; must be None.

    Supported (block_size, output_kind_pair, swizzle_type) combinations and their outputs:
      * (1, 32),        NORMAL,        NO_SWIZZLE / SWIZZLE_32_4_4   -> (qdata (M,N), scale)
      * (32, 32),       NORMAL,        NO_SWIZZLE                    -> (qdata (M,N), scale (M//32,N//32))
      * (32, 1),        TRANSP_CONTIG, NO_SWIZZLE / SWIZZLE_32_4_4   -> (qdata (N,M), scale)
      * ((1,32),(32,1)),(NORMAL, TRANSP_CONTIG), (NO_SWIZZLE, NO_SWIZZLE) / (SWIZZLE_32_4_4,)*2
            -> (qdata_k (M,N), scale_k, qdata_m (N,M), scale_m)

    With NO_SWIZZLE each scale is plain 2D ((M,N//32) or (N,M//32)); with SWIZZLE_32_4_4 it is the
    NVIDIA 32x4x4 blocked layout. scale tensors are float8_e8m0fnu. Any other combination raises
    ValueError.
    """
    assert input.dim() == 2, f"only 2D input supported for now, got {input.dim()}D"
    assert input.is_contiguous(), "input must be contiguous"
    if pad_input_to_next_multiple_of is not None:
        raise NotImplementedError("pad_input_to_next_multiple_of is not implemented yet")
    if rounding_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("rounding_mode=STOCHASTIC is not implemented yet")
    if random_key is not None:
        raise NotImplementedError("random_key (stochastic rounding) is not implemented yet")

    spec = (block_size, output_kind_pair, swizzle_type)
    if spec == ((1, 32), OutputKindPair.NORMAL, SwizzleType.NO_SWIZZLE):
        assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
        return mxfp8_triton(input)
    if spec == ((1, 32), OutputKindPair.NORMAL, SwizzleType.SWIZZLE_32_4_4):
        assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
        return mxfp8_swizzle_triton(input)
    if spec == ((32, 32), OutputKindPair.NORMAL, SwizzleType.NO_SWIZZLE):
        assert input.shape[0] % 32 == 0, f"first dim must be a multiple of 32, got {input.shape[0]}"
        assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
        return mxfp8_32x32_triton(input)
    if spec == ((32, 1), OutputKindPair.TRANSP_CONTIG, SwizzleType.NO_SWIZZLE):
        return mxfp8_dim_m_triton(input)
    if spec == ((32, 1), OutputKindPair.TRANSP_CONTIG, SwizzleType.SWIZZLE_32_4_4):
        return mxfp8_dim_m_swizzle_triton(input)
    if spec == (
        ((1, 32), (32, 1)),
        (OutputKindPair.NORMAL, OutputKindPair.TRANSP_CONTIG),
        (SwizzleType.NO_SWIZZLE, SwizzleType.NO_SWIZZLE),
    ):
        return mxfp8_dim_km_triton(input)
    if spec == (
        ((1, 32), (32, 1)),
        (OutputKindPair.NORMAL, OutputKindPair.TRANSP_CONTIG),
        (SwizzleType.SWIZZLE_32_4_4, SwizzleType.SWIZZLE_32_4_4),
    ):
        return mxfp8_dim_km_swizzle_triton(input)

    raise ValueError(
        f"unsupported (block_size, output_kind_pair, swizzle_type)={spec!r}; supported: "
        "((1,32), NORMAL, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "((32,32), NORMAL, NO_SWIZZLE), "
        "((32,1), TRANSP_CONTIG, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "(((1,32),(32,1)), (NORMAL, TRANSP_CONTIG), (NO_SWIZZLE,NO_SWIZZLE)|(SWIZZLE_32_4_4,SWIZZLE_32_4_4))"
    )
