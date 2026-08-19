import os
import sys
from enum import IntEnum, StrEnum

from torch import Tensor
from torch.nn.functional import SwizzleType  # core enum: NO_SWIZZLE=0, SWIZZLE_32_4_4=1

# The editable install only exposes `quant_cast_bench`, not the `experiments` tree, so put the repo
# root on sys.path to make the sibling `experiments.mxfp8_api.moe_utils` import resolve regardless of
# how this module is reached (as a package, as a bare `api`, or under pytest).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.mxfp8_api.moe_utils import (  # noqa: E402
    BLOCK_SIZE,
    _pad_token_groups,
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
    # always contiguous.

    # block maps to (M, N) as given
    NATURAL = "natural"          
    # block maps to the (N, M) view; qdata/scale written transposed-contig
    TRANSPOSED = "transposed"    
    # fused: emit the NATURAL pair, then the TRANSPOSED pair (4 outputs)
    BOTH = "both"                
    # only needed for square scaling types and on hardware (such as Blackwell) where the second argument of a scaled gemm can be row-major
    BOTH_SCALES_NATURAL_QDATA = "both_scales_natural_qdata" 


class RoundingMode(StrEnum):
    # how qdata values are rounded when cast into float8_e4m3fn.
    RTNE = "rtne"                # round to nearest, ties to even
    STOCHASTIC = "stochastic"    # stochastic rounding


def quantize_to_mxfp8(
    input: Tensor,
    *,
    scaling_type: ScalingType = ScalingType.BlockWise1x32,
    orientation: QuantOrientation = QuantOrientation.NATURAL,
    swizzle_type: SwizzleType = SwizzleType.SWIZZLE_32_4_4,
    pad_input_to_next_multiple_of: tuple[int, int] | None = None,
    rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """Quantize `input` to mxfp8: float8_e4m3fn qdata + one e8m0 scale per block.

    Args:
      input: 2D input tensor (bf16 or fp32) of shape (M, N).
      scaling_type: only BlockWise1x32 and BlockWise32x32
      orientation: how scaling_type maps onto (M, N) and how the output is laid out.
        NATURAL maps the scaling_type to (M, N)
        TRANSPOSED maps it to the (N, M) view and writes the outputs transposed-contiguous
        BOTH runs both in one fused pass
        BOTH_SCALES_NATURAL_QDATA square blocks only, outputs dim-k qdata and both scales
      swizzle_type: NO_SWIZZLE or SWIZZLE_32_4_4
      pad_input_to_next_multiple_of: per-input-dimension zero-padding fused into the cast
      rounding_mode: RTNE or STOCHASTIC
      random_key: entropy source for stochastic rounding

    Returns:
        2 tensors (qdata, scale) for NATURAL or TRANSPOSED
        4 tensors (qk, sk, qm, sm) for BOTH
        3 tensors (qk, sk, sm) for BOTH_SCALES_NATURAL_QDATA
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
    if spec == (
        ScalingType.BlockWise32x32,
        QuantOrientation.BOTH_SCALES_NATURAL_QDATA,
        SwizzleType.SWIZZLE_32_4_4,
    ):
        assert input.shape[0] % 32 == 0, f"first dim must be a multiple of 32, got {input.shape[0]}"
        assert input.shape[1] % 32 == 0, f"last dim must be a multiple of 32, got {input.shape[1]}"
        return mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_triton(input)

    raise ValueError(
        f"unsupported (scaling_type, orientation, swizzle_type)={spec!r}; supported: "
        "(BlockWise1x32, NATURAL, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "(BlockWise1x32, TRANSPOSED, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "(BlockWise1x32, BOTH, NO_SWIZZLE|SWIZZLE_32_4_4), "
        "(BlockWise32x32, NATURAL, NO_SWIZZLE), "
        "(BlockWise32x32, BOTH_SCALES_NATURAL_QDATA, SWIZZLE_32_4_4)"
    )


def quantize_to_mxfp8_grouped(
    input: Tensor,  # (total_M, C)
    offs: Tensor,
    *,
    scaling_type: ScalingType = ScalingType.BlockWise1x32,
    orientation: QuantOrientation = QuantOrientation.NATURAL,
    swizzle_type: SwizzleType = SwizzleType.SWIZZLE_32_4_4,
    pad_input_to_next_multiple_of: tuple[int, int] | None = None,
    rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """One-shot mxfp8 cast of a grouped token operand for the real `torch._scaled_grouped_mm`, the
    grouped analog of the dense `quantize_to_mxfp8` above. Composes the `moe_utils` helpers
    (token-group pad -> `mxfp8_f` -> blocked-scale swizzle); no new math. Pure-PyTorch reference; a
    fused triton/cute kernel can later drop in behind this seam.

    Mirrors the dense `quantize_to_mxfp8` interface; the one extra argument is `offs` (int32 group-end
    offsets in token space), and all the knobs are keyword-only. Unlike the dense cast, `swizzle_type`
    defaults to SWIZZLE_32_4_4: the grouped op requires blocked/swizzled scales, so there is no
    NO_SWIZZLE grouped variant. Only the combination actually implemented today runs; anything else
    raises (see below). No new features are wired here relative to the previous orientation-only cast.

    Supported arguments:
      * scaling_type: BlockWise1x32 only.
      * orientation: NATURAL blocks the 1x32 along the last (contraction) dim and emits the M-groups
        blocked scale; TRANSPOSED blocks along the token dim M and emits the K-groups blocked scale
        (scale offsets = padded_offs // BLOCK_SIZE, internal); BOTH emits both pairs from a single
        padded read -- the fusion-visible case a future kernel would collapse into one pass.
      * swizzle_type: SWIZZLE_32_4_4 only.
      * pad_input_to_next_multiple_of: must be None (NOT IMPLEMENTED).
      * rounding_mode: RTNE only (STOCHASTIC NOT IMPLEMENTED).
      * random_key: must be None (NOT IMPLEMENTED).

    qdata is always returned row-major/contiguous; the caller composes the mat2 `.transpose(-2, -1)`
    view at the GEMM call site. `padded_offs` (token space) is returned for the op's `offs=` and for
    unpadding; the padded start rows and original token count are recoverable from it.

    Returns:
        NATURAL:    `(q (Mp, C),  scale_blocked_m_groups, padded_offs)`
        TRANSPOSED: `(q (C,  Mp), scale_blocked_k_groups, padded_offs)`
        BOTH:       `(q_natural (Mp, C), scale_blocked_m_groups,
                      q_transposed (C, Mp), scale_blocked_k_groups, padded_offs)`
    """
    if pad_input_to_next_multiple_of is not None:
        raise NotImplementedError("pad_input_to_next_multiple_of is not implemented yet")
    if rounding_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("rounding_mode=STOCHASTIC is not implemented yet")
    if random_key is not None:
        raise NotImplementedError("random_key (stochastic rounding) is not implemented yet")
    if (scaling_type, swizzle_type) != (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        raise ValueError(
            f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}); "
            "quantize_to_mxfp8_grouped supports only (BlockWise1x32, SWIZZLE_32_4_4)"
        )
    if orientation not in (
        QuantOrientation.NATURAL, QuantOrientation.TRANSPOSED, QuantOrientation.BOTH
    ):
        raise ValueError(
            f"orientation={orientation!r} is not supported by quantize_to_mxfp8_grouped; "
            "supported: NATURAL, TRANSPOSED, BOTH"
        )

    padded, _, padded_offs = _pad_token_groups(input, offs)

    if orientation in (QuantOrientation.NATURAL, QuantOrientation.BOTH):
        q_nat, s_nat = quantize_2d_act(padded)  # (Mp, C), 1x32 along C
        sb_nat = _to_blocked_2d_m_groups(s_nat, padded_offs)
    if orientation in (QuantOrientation.TRANSPOSED, QuantOrientation.BOTH):
        q_t, s_t = quantize_2d_act(padded.transpose(-2, -1).contiguous())  # (C, Mp), 1x32 along M
        sb_t = _to_blocked_2d_k_groups(s_t, padded_offs // BLOCK_SIZE)

    if orientation == QuantOrientation.NATURAL:
        return q_nat, sb_nat, padded_offs
    if orientation == QuantOrientation.TRANSPOSED:
        return q_t, sb_t, padded_offs
    return q_nat, sb_nat, q_t, sb_t, padded_offs


def quantize_to_mxfp8_batched(
    input: Tensor,  # (E, N, K)
    *,
    scaling_type: ScalingType = ScalingType.BlockWise1x32,
    orientation: QuantOrientation = QuantOrientation.NATURAL,
    swizzle_type: SwizzleType = SwizzleType.SWIZZLE_32_4_4,
    pad_input_to_next_multiple_of: tuple[int, int] | None = None,
    rounding_mode: RoundingMode = RoundingMode.RTNE,
    random_key: Tensor | None = None,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """Batched (per-expert) mxfp8 cast of a 3d weight stack `(E, N, K)` for the real
    `torch._scaled_grouped_mm` -- the batched analog of the dense `quantize_to_mxfp8` above. Unlike
    `quantize_to_mxfp8_grouped`, the expert axis is a plain batch dim with NO offsets (each expert is
    a full dense matrix), so this stays a separate weight-only path and takes no `offs`. Composes
    `mxfp8_f` + `_to_blocked_per_group_3d`; no new math.

    Mirrors the dense `quantize_to_mxfp8` interface exactly (same args, all knobs keyword-only), only
    over a 3d input. As with `quantize_to_mxfp8_grouped`, `swizzle_type` defaults to SWIZZLE_32_4_4:
    the grouped op requires blocked/swizzled scales, so there is no NO_SWIZZLE batched variant. Only
    the combination actually implemented today runs; anything else raises. No new features are wired
    here relative to the previous orientation-only cast.

    Supported arguments:
      * scaling_type: BlockWise1x32 only.
      * orientation: NATURAL blocks the 1x32 along the last dim K (qdata `(E,N,K)`, scale `(E,N,K//32)`);
        TRANSPOSED blocks along N (qdata `(E,K,N)`); BOTH emits both pairs from one read -- the
        fusion-visible case: for a `weight_t (E,K,N)` stack (last dim N), NATURAL is the dgrad-B cast
        (block along N) and TRANSPOSED is the fwd-B cast (block along K), so BOTH yields both weight
        casts a forward+backward step needs in a single pass.
      * swizzle_type: SWIZZLE_32_4_4 only.
      * pad_input_to_next_multiple_of: must be None (NOT IMPLEMENTED).
      * rounding_mode: RTNE only (STOCHASTIC NOT IMPLEMENTED).
      * random_key: must be None (NOT IMPLEMENTED).

    qdata is returned row-major/contiguous; the caller composes the mat2 `.transpose(-2, -1)` view at
    the GEMM call site.

    Returns:
        NATURAL:    `(q (E,N,K), scale_blocked)`
        TRANSPOSED: `(q (E,K,N), scale_blocked)`
        BOTH:       `(q_natural (E,N,K), scale_blocked_natural,
                      q_transposed (E,K,N), scale_blocked_transposed)`
    """
    assert input.dim() == 3, "input must be 3D (E, N, K)"
    if pad_input_to_next_multiple_of is not None:
        raise NotImplementedError("pad_input_to_next_multiple_of is not implemented yet")
    if rounding_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("rounding_mode=STOCHASTIC is not implemented yet")
    if random_key is not None:
        raise NotImplementedError("random_key (stochastic rounding) is not implemented yet")
    if (scaling_type, swizzle_type) != (ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4):
        raise ValueError(
            f"unsupported (scaling_type, swizzle_type)=({scaling_type!r}, {swizzle_type!r}); "
            "quantize_to_mxfp8_batched supports only (BlockWise1x32, SWIZZLE_32_4_4)"
        )
    if orientation not in (
        QuantOrientation.NATURAL, QuantOrientation.TRANSPOSED, QuantOrientation.BOTH
    ):
        raise ValueError(
            f"orientation={orientation!r} is not supported by quantize_to_mxfp8_batched; "
            "supported: NATURAL, TRANSPOSED, BOTH"
        )

    if orientation in (QuantOrientation.NATURAL, QuantOrientation.BOTH):
        q_nat, s_nat = mxfp8_f(input.contiguous())  # (E,N,K), 1x32 along K
        sb_nat = _to_blocked_per_group_3d(s_nat)
    if orientation in (QuantOrientation.TRANSPOSED, QuantOrientation.BOTH):
        q_t, s_t = mxfp8_f(input.transpose(-2, -1).contiguous())  # (E,K,N), 1x32 along N
        sb_t = _to_blocked_per_group_3d(s_t)

    if orientation == QuantOrientation.NATURAL:
        return q_nat, sb_nat
    if orientation == QuantOrientation.TRANSPOSED:
        return q_t, sb_t
    return q_nat, sb_nat, q_t, sb_t
