"""Correctness tests for the Helion quant-cast recipes: each `helion_fn` must reproduce its
gold `pt_ref_fn`'s outputs. Inputs come from the recipe's (inherited) `example_input_fn`.
"""

import importlib.util
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdata_utils import mismatch_fraction, qdata_equal

# Helion is an optional dependency; skip the whole module cleanly if it (or the recipes that
# import it) can't be imported, rather than erroring at collection.
HAS_HELION = importlib.util.find_spec("helion") is not None
if HAS_HELION:
    from quant_cast_bench.quant_cast_helion.recipes import ALL_RECIPES
else:
    ALL_RECIPES = []

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_HELION,
    reason="requires CUDA and the helion package",
)


# fraction of the RNE-tie divergence between Helion's and PyTorch's fp8/fp4 casts we tolerate
# before treating it as a real bug (see the fallback in the test below).
_MAX_MISMATCH_FRAC = 0.01


@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_helion_matches_reference(name, recipe):
    # the Helion kernel should reproduce the gold reference bit-for-bit (identical fp32 math + RNE
    # cast). example_input_fn builds the full positional inputs (x, *aux).
    torch.manual_seed(0)
    inputs = recipe.example_input_fn(512, 512)

    # flex_tile_map framework kwargs naming the tile's global origin + parent row stride. The test
    # runs the whole tensor as one tile, so origin = (0, 0) and num_col = full width. Every recipe
    # fn takes **kwargs, and the Helion kernels ignore them (they own their own tiling), so passing
    # them is harmless.
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    hel_outs = recipe.helion_fn(*inputs, **tile_kwargs)

    assert len(hel_outs) == len(ref_outs), f"{name}: output count {len(hel_outs)} != {len(ref_outs)}"
    for i, (t, r) in enumerate(zip(hel_outs, ref_outs)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )

    if all(qdata_equal(t, r) for t, r in zip(hel_outs, ref_outs)):
        return  # exact match to the reference (the common case)

    # Some outputs differ. The legitimate source here is Helion-vs-PyTorch RNE tie-breaking in the
    # fp8/fp4 cast (the pre-cast fp32 math is identical to the reference). Accept iff the Helion
    # outputs are still a valid quantization (gold correctness_fn) AND the byte-level divergence is
    # tiny (guards against real bugs).
    recipe.correctness_fn(inputs, hel_outs)
    if "_sr" in name:
        return
    for i, (t, r) in enumerate(zip(hel_outs, ref_outs)):
        frac = mismatch_fraction(t, r)
        assert frac < _MAX_MISMATCH_FRAC, (
            f"{name} output {i}: {frac:.3%} of elements differ from reference -- too many for "
            f"RNE ties, likely a real bug"
        )
