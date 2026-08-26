"""Correctness tests for the Helion quant-cast recipes: each `helion_fn` must reproduce its
gold `pt_ref_fn`'s outputs. Inputs come from the recipe's (inherited) `example_input_fn`.
"""

import importlib.util
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdata_utils import mismatch_fraction, qdata_and_scale_equal

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


# Recipes whose Helion kernels emit the Blackwell-only fp4 E2M1 cvt (`cvt.e2m1x2.f32`); ptxas
# rejects it below sm_100, so gate them to cuda capability 10.0. (The mxfp8 dim_m Helion kernel
# computes E8M0 in software and runs everywhere, so it is not gated.)
_REQUIRES_SM100 = frozenset({
    "nvfp4",
    "nvfp4_swizzle",
})


@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_helion_matches_reference(name, recipe):
    # the Helion kernel should reproduce the gold reference bit-for-bit (identical fp32 math + RNE
    # cast). example_input_fn builds the full positional inputs (x, *aux).
    if name in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{name} emits Blackwell-only PTX; requires cuda capability 10.0")
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

    # Every recipe's outputs must be a valid quantization (the gold correctness_fn).
    recipe.correctness_fn(inputs, hel_outs)

    # Stochastic rounding (the *_sr recipes) is the one case that can't bit-match: the Helion kernel
    # draws from its own counter-based Philox, not the reference's torch RNG, so only the SR *property*
    # (unbiased, lands on the two bracketing grid points) is well-defined -- correctness_fn above checks
    # that, and we stop (~2p(1-p) of elements differ between any two independent draws, so a per-element
    # bound is meaningless here).
    if "_sr" in name:
        return

    # Every other recipe must reproduce the gold bit-for-bit: identical fp32 math + RNE cast (the
    # kernels mirror torch's per-op rounding, incl. reciprocal-multiply vs div.rn, so even the fp8/fp4
    # RNE ties resolve the same way). Any divergence is a real bug, not tolerable RNE noise.
    for i, (t, r) in enumerate(zip(hel_outs, ref_outs)):
        assert qdata_and_scale_equal(t, r), (
            f"{name} output {i}: {mismatch_fraction(t, r):.3%} of elements differ from the gold "
            f"reference -- expected bit-for-bit equality"
        )
