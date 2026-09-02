"""Correctness tests for the handwritten CuTeDSL quant-cast recipes (quant_cast_cute_hand): each
`cute_fn` must reproduce its gold `pt_ref_fn` bit-for-bit. Mirrors test_quant_cast_cute.py; this is
the playground module we iterate on.
"""

import importlib.metadata
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdata_utils import mismatch_fraction, qdata_and_scale_equal

# The CuTeDSL kernels import `_maybe_recast_from_f4_f6` (the fp4/fp6 register-packing helper) from
# cutlass.cute.testing. That is the nvidia-cutlass-dsl >= 4.5.2 name; gate the whole module on the
# installed version so an older (or absent) install skips cleanly instead of erroring at collection.
_MIN_CUTEDSL = (4, 5, 2)
try:
    _cutedsl_version = tuple(
        int(x) for x in importlib.metadata.version("nvidia-cutlass-dsl").split(".")[:3]
    )
except (ImportError, importlib.metadata.PackageNotFoundError):
    _cutedsl_version = None

HAS_CUTEDSL = _cutedsl_version is not None and _cutedsl_version >= _MIN_CUTEDSL

if HAS_CUTEDSL:
    from quant_cast_bench.quant_cast_cute_hand.recipes import (
        ALL_RECIPES, add_v0, add_v1, add_v2
    )
else:
    ALL_RECIPES = []

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_CUTEDSL,
    reason=(
        f"requires CUDA and nvidia-cutlass-dsl >= {'.'.join(map(str, _MIN_CUTEDSL))} "
        f"(found {'.'.join(map(str, _cutedsl_version)) if _cutedsl_version else 'none'})"
    ),
)

def _get_recipe(recipe_name):
    _recipe_name, recipe = [x for x in ALL_RECIPES if x[0] == recipe_name][0]
    return recipe

def test_add_v0():
    M, K = 2, 64
    inputs = torch.arange(M * K, device="cuda", dtype=torch.float32).view(M, K)
    print(inputs.shape)
    print(inputs)

    num = 1.0
    outputs = add_v0(inputs, num)
    print(outputs)
    assert torch.equal(outputs, inputs + num)

def test_add_v1():
    M, K = 2, 64
    inputs = torch.arange(M * K, device="cuda", dtype=torch.float32).view(M, K)
    print(inputs.shape)
    print(inputs)

    num = 1.0
    outputs = add_v1(inputs, num)
    print(outputs)
    assert torch.equal(outputs, inputs + num)

def test_add_v2():
    M, K = 2, 64
    # M, K = 4, 1024
    inputs = torch.arange(M * K, device="cuda", dtype=torch.float32).view(M, K)
    # print(inputs.shape)
    # print(inputs)

    num = 1.0
    outputs = add_v2(inputs, num)
    # print(outputs)
    assert torch.equal(outputs, inputs + num)

def test_deepseek_1x128():
    recipe = _get_recipe("deepseek_1x128")
    # inputs = recipe.example_input_fn(512, 512)
    # print(inputs)

    inputs = torch.arange(32, device="cuda").unsqueeze(0)
    print(inputs.shape)
    print(inputs)

    outputs = recipe.cute_fn(inputs)
    print(outputs)



# TODO(re-enable this after numerics work)
@pytest.mark.skip()
@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_cute_hand_matches_reference(name, recipe):
    # the CuTeDSL kernel should reproduce the gold reference bit-for-bit (identical fp32 math + RNE
    # cast). example_input_fn builds the full positional inputs (x, *aux).
    torch.manual_seed(0)
    inputs = recipe.example_input_fn(512, 512)

    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    cute_outs = recipe.cute_fn(*inputs, **tile_kwargs)

    assert len(cute_outs) == len(ref_outs), f"{name}: output count {len(cute_outs)} != {len(ref_outs)}"
    for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )

    # Every recipe's outputs must be a valid quantization (the gold correctness_fn).
    recipe.correctness_fn(inputs, cute_outs)

    # And must reproduce the gold bit-for-bit: identical fp32 math + RNE cast.
    for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
        assert qdata_and_scale_equal(t, r), (
            f"{name} output {i}: {mismatch_fraction(t, r):.3%} of elements differ from the gold "
            f"reference -- expected bit-for-bit equality"
        )
