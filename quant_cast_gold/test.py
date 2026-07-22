"""Standalone correctness tests for the golden quant-cast recipes.

Each `QuantCastSingleKernelGold` must be internally consistent: running its `correctness_fn`
on `pt_ref_fn`'s own outputs has to pass. That's a gold-package concern (no flex_tile_map
involved), so it lives here rather than in flexquant_v3/test.py. Kept independent of
flexquant_v3 -- inputs (and any aux args) come from each recipe's own `example_input_fn`.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_gold.recipes import ALL_RECIPES

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.mark.parametrize(
    "name, gold",
    ALL_RECIPES,
    ids=[name for name, _ in ALL_RECIPES],
)
def test_ref_correctness(name, gold):
    # each gold recipe is internally consistent: pt_ref_fn's own outputs clear its correctness_fn.
    # example_input_fn builds the full positional inputs (x, *aux). Calls pt_ref_fn directly on
    # the whole tensor (no flex_tile_map). The whole tensor is one tile, so we pass the origin
    # position kwargs a REFERENCE-style whole-tensor call would -- recipes that ignore them accept
    # **kwargs; sr_bf16_global needs them for its per-element global-position dither.
    torch.manual_seed(0)
    inputs = gold.example_input_fn(512, 512)

    outputs = gold.pt_ref_fn(*inputs, global_row=0, global_col=0, num_col=inputs[0].shape[1])
    gold.correctness_fn(inputs, outputs)  # raises AssertionError on failure
