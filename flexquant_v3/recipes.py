"""Quant recipes for flex_tile_map, bundled as `Recipe` dataclasses.

Each recipe pairs a plain-PyTorch quant kernel `quant(x) -> (qdata, aux_out)` (v1
`_reference_fn` style) with its `dequant(qdata, scale) -> fp32` inverse. The `RECIPES`
table registers them (plus per-recipe test metadata) for the tests in test.py. Math
mirrors flexquant v1 recipes.py.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Tuple

from api import AuxKind, OutputKind

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_gold.recipes import (
    ColwiseFp8Gold,
    ColwisePrecalcGold,
    Deepseek1x128DimMGold,
    Deepseek1x128Gold,
    Deepseek128x128Gold,
    Float8TensorwiseGold,
    HadamardRht,
    Mxfp832x32FloorGold,
    Mxfp8FloorDimMGold,
    Mxfp8FloorGold,
    Mxfp8FloorSwizzleGold,
    Mxfp8BiasGold,
    Nvfp4BlockedOuterGold,
    Nvfp4GsSwizzleGold,
    QuantCastSingleKernelGold,
    RowwiseFp8Gold,
    RowwisePrecalcGold,
    SrF32ToBf16,
    SrF32ToBf16Global,
)


@dataclass(frozen=True)
class RecipeV2(QuantCastSingleKernelGold):
    """A flexquant_v3 recipe backed directly by a quant_cast_gold golden recipe.

    Inherits `pt_ref_fn`/`correctness_fn`/`example_input_fn` from `QuantCastSingleKernelGold`
    unchanged -- flexquant_v3 adds the things gold doesn't know about: the flex_tile_map
    tiling constraint (`valid_tile_size_fn`), how each aux input tiles (`aux_kinds`), and how
    each output tile is placed (`output_kinds`). The aux VALUES come from the inherited
    `example_input_fn` (which returns `(x, *aux)`); `aux_kinds` is just the per-aux tiling
    metadata. Recipes migrate here incrementally (see quant_cast_gold/recipes.py).
    """

    valid_tile_size_fn: Callable[
        [Tuple[int, int], Tuple[int, int], Tuple[int, int]], bool
    ] | None = None
    aux_kinds: Tuple[Any, ...] | None = None  # AuxKind per aux input (from example_input_fn)
    output_kinds: Tuple[Any, ...] | None = None  # OutputKind per pt_ref_fn() output

    @classmethod
    def from_gold(
        cls,
        gold: QuantCastSingleKernelGold,
        valid_tile_size_fn=None,
        aux_kinds=None,
        output_kinds=None,
    ) -> "RecipeV2":
        """Build a RecipeV2 from a gold recipe, adding the flex_tile_map tiling metadata."""
        return cls(
            pt_ref_fn=gold.pt_ref_fn,
            correctness_fn=gold.correctness_fn,
            example_input_fn=gold.example_input_fn,
            perf_description=gold.perf_description,
            valid_tile_size_fn=valid_tile_size_fn,
            aux_kinds=aux_kinds,
            output_kinds=output_kinds,
        )


# All single-kernel quant recipes have migrated to quant_cast_gold (see
# quant_cast_gold/recipes.py); their Gold objects (and precompute helpers like
# `nvfp4_blocked_outer_scale`) are imported above. Only the non-quant examples (RHT,
# stochastic rounding) remain defined locally below, alongside the RecipeV2 constructions.


# Reduction constraints check `actual` (the real, possibly-ragged tile extent) so a severed
# reduction block is rejected at the edge; swizzle-atom constraints check `padded` (the nominal
# tile size, so ragged edge tiles are exempt -- recovers the old full_tile_multiple_of semantics).
# Migrated to quant_cast_gold: pt_ref_fn/correctness_fn come straight from the Gold object,
# RecipeV2 adds only the tiling constraint.
DEEPSEEK_1X128 = RecipeV2.from_gold(
    Deepseek1x128Gold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 128 == 0,
)
DEEPSEEK_128X128 = RecipeV2.from_gold(
    Deepseek128x128Gold,
    valid_tile_size_fn=lambda ts, a, p: a[0] % 128 == 0 and a[1] % 128 == 0,
)
# dim-M: `f` transposes the tile + reduces last dim; caller pairs it with
# output_kinds=SWAP_TILE_INDEX for the grid transpose. correctness_fn works in the (K, M)
# transposed frame and transposes back before comparing to `x`. After the within-tile
# transpose the reduced (last) dim is the tile's original ROWS, so rows must be a
# 128-multiple (checked on `actual`).
# TODO(future): the recipe framework needs a new field to test correctness for both
# the reference function as well as the backend specified function, right now this
# is not intuitive.
DEEPSEEK_1X128_DIM_M = RecipeV2.from_gold(
    Deepseek1x128DimMGold,
    valid_tile_size_fn=lambda ts, a, p: a[0] % 128 == 0,
    # `f` writes both outputs transposed locally; the grid swap yields the (K, M) layout.
    output_kinds=(OutputKind.SWAP_TILE_INDEX, OutputKind.SWAP_TILE_INDEX),
)
# rowwise / colwise: the tile must span the reduced dim (predicate forces it to equal the tensor
# extent), so the framework's tile-size search selects a full-span tile.
ROWWISE_FP8 = RecipeV2.from_gold(
    RowwiseFp8Gold,
    valid_tile_size_fn=lambda ts, a, p: a[1] == ts[1],  # span all columns
)
COLWISE_FP8 = RecipeV2.from_gold(
    ColwiseFp8Gold,
    valid_tile_size_fn=lambda ts, a, p: a[0] == ts[0],  # span all rows
    # `f` writes both outputs transposed locally; the grid swap yields the (N, M) layout.
    output_kinds=(OutputKind.SWAP_TILE_INDEX, OutputKind.SWAP_TILE_INDEX),
)
# rowwise with a precalculated (M, 1) scale passed as an AuxKind.ROW aux input; the divide is
# tile-invariant under plain 2D tiling (no tiling constraint needed). The scale value is built by
# RowwisePrecalcGold.example_input_fn; aux_kinds just says how it tiles.
ROWWISE_PRECALC = RecipeV2.from_gold(
    RowwisePrecalcGold,
    aux_kinds=(AuxKind.ROW,),
)
# colwise with a precalculated (1, N) scale (AuxKind.COL) + transposed-contiguous output (pair
# with output_kinds=SWAP_TILE_INDEX); tile-invariant under plain 2D tiling.
COLWISE_PRECALC = RecipeV2.from_gold(
    ColwisePrecalcGold,
    aux_kinds=(AuxKind.COL,),
    output_kinds=(OutputKind.SWAP_TILE_INDEX,),
)
# reduction (1x32) checked on `actual`.
MXFP8_FLOOR = RecipeV2.from_gold(
    Mxfp8FloorGold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 32 == 0,
)
# dim-M mxfp8: 1x32 reduction runs down M (a[0] % 32), outputs written transposed (grid swap).
MXFP8_FLOOR_DIM_M = RecipeV2.from_gold(
    Mxfp8FloorDimMGold,
    valid_tile_size_fn=lambda ts, a, p: a[0] % 32 == 0,
    output_kinds=(OutputKind.SWAP_TILE_INDEX, OutputKind.SWAP_TILE_INDEX),
)
# mxfp8 with square 32x32 blocks: each block is independent, so a tile just needs whole blocks
# on both dims (checked on `actual`). No swizzle, no transpose.
MXFP8_32X32_FLOOR = RecipeV2.from_gold(
    Mxfp832x32FloorGold,
    valid_tile_size_fn=lambda ts, a, p: a[0] % 32 == 0 and a[1] % 32 == 0,
)
# reduction (1x32) checked on `actual`; swizzle atom (128x128) checked on `padded` (edge-exempt).
MXFP8_FLOOR_SWIZZLE = RecipeV2.from_gold(
    Mxfp8FloorSwizzleGold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 32 == 0 and p[0] % 128 == 0 and p[1] % 128 == 0,
)
# Tensorwise recipe: the per-tensor scale (built by Float8TensorwiseGold.example_input_fn) is a
# REPLICATE aux input to pt_ref_fn.
FLOAT8_TENSORWISE = RecipeV2.from_gold(
    Float8TensorwiseGold,
    aux_kinds=(AuxKind.REPLICATE,),
)
# nvfp4 recipe: the per-tensor outer scale (from Nvfp4GsSwizzleGold.example_input_fn) is a
# REPLICATE aux input. reduction (1x16 inner) on `actual`; swizzle atom (128x64) on `padded`.
NVFP4_GS_SWIZZLE = RecipeV2.from_gold(
    Nvfp4GsSwizzleGold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 16 == 0 and p[0] % 128 == 0 and p[1] % 64 == 0,
    aux_kinds=(AuxKind.REPLICATE,),
)
# nvfp4 with a 128x128-blocked outer scale (from Nvfp4BlockedOuterGold.example_input_fn) passed as
# an AuxKind.TILE aux. Same swizzle-atom constraints as NVFP4_GS_SWIZZLE; the 128x128 outer block
# is coarser than the (128, 64) atom so it adds no new alignment constraint at 128-aligned tiles.
NVFP4_BLOCKED_OUTER = RecipeV2.from_gold(
    Nvfp4BlockedOuterGold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 16 == 0 and p[0] % 128 == 0 and p[1] % 64 == 0,
    aux_kinds=(AuxKind.TILE,),
)
# mxfp8 FLOOR with an elementwise bias (same shape as input) added before quant, passed as an
# AuxKind.TILE aux with divisor (1, 1). Mxfp8BiasGold.example_input_fn supplies a fixed ones-tensor
# bias (it isn't derived from `x`); Mxfp8BiasGold's correctness_fn only checks shape/dtype.
MXFP8_BIAS = RecipeV2.from_gold(
    Mxfp8BiasGold,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 32 == 0,
    aux_kinds=(AuxKind.TILE,),
)


# RHT (non-quant): apply the 16x16 orthogonal transform along the last dim. The RHT matrix
# (built by HadamardRht.example_input_fn from a fixed +/-1 sign vector) is passed as a REPLICATE
# aux; a column tile must keep 16-groups intact (a[1] % 16 == 0), else it would sever a transform
# block. Correctness is the roundtrip check (HadamardRht.correctness_fn: x recovered via rht.t()).
HADAMARD_RHT = RecipeV2.from_gold(
    HadamardRht,
    valid_tile_size_fn=lambda ts, a, p: a[1] % 16 == 0,
    aux_kinds=(AuxKind.REPLICATE,),
)
# stochastic rounding fp32 -> bf16 (tile-LOCAL, from SrF32ToBf16 gold). The DELIBERATE
# non-tile-invariant counterexample: the dither is keyed on tile-local element order, so
# MANUAL_TILE rounds differently from REFERENCE (test_flex_tile_map_backends_keep_numerics is
# skipped for it -- see the skip in test.py). Its example_input_fn supplies the fp32 constant
# input + the REPLICATE PRNG key.
SR_BF16 = RecipeV2.from_gold(
    SrF32ToBf16,
    aux_kinds=(AuxKind.REPLICATE,),
)
# tiling-INVARIANT SR (from SrF32ToBf16Global): keys the dither on each element's GLOBAL
# position, so REFERENCE == MANUAL_TILE bit-for-bit (unlike SR_BF16). Its backend check is still
# skipped in the generic suite (kept alongside SR_BF16); the invariance is asserted by
# test_sr_bf16_global_tiling_invariant.
SR_BF16_GLOBAL = RecipeV2.from_gold(
    SrF32ToBf16Global,
    aux_kinds=(AuxKind.REPLICATE,),
)
RECIPES_V2 = [
    ("fp8_deepseek_1x128", DEEPSEEK_1X128),
    ("fp8_deepseek_128x128", DEEPSEEK_128X128),
    ("fp8_deepseek_1x128_dim_m", DEEPSEEK_1X128_DIM_M),
    ("fp8_rowwise", ROWWISE_FP8),
    ("fp8_colwise", COLWISE_FP8),
    ("fp8_rowwise_precalc_scale", ROWWISE_PRECALC),
    ("fp8_colwise_precalc_scale", COLWISE_PRECALC),
    ("mxfp8_floor", MXFP8_FLOOR),
    ("mxfp8_floor_dim_m", MXFP8_FLOOR_DIM_M),
    ("mxfp8_32x32_floor", MXFP8_32X32_FLOOR),
    ("mxfp8_floor_swizzle", MXFP8_FLOOR_SWIZZLE),
    ("fp8_tensorwise_precalc_scale", FLOAT8_TENSORWISE),
    ("nvfp4_swizzle", NVFP4_GS_SWIZZLE),
    ("nvfp4_blocked_outer", NVFP4_BLOCKED_OUTER),
    ("mxfp8_bias", MXFP8_BIAS),
    ("bf16_rht", HADAMARD_RHT),
    ("fp32_to_bf16_sr", SR_BF16),
    ("fp32_to_bf16_sr_global_offsets", SR_BF16_GLOBAL),
]


