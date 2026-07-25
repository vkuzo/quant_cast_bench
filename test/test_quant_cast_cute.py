"""Correctness tests for the CuTeDSL quant-cast recipes: each `cute_fn` must reproduce its gold
`pt_ref_fn`'s outputs. Mirrors quant_cast_triton/test.py (same comparison + tolerance fallback).
"""

import importlib.metadata
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdata_utils import mismatch_fraction, qdata_equal

# The CuTeDSL kernels import `_maybe_recast_from_f4_f6` (the fp4/fp6 register-packing helper) from
# cutlass.cute.testing. That is the nvidia-cutlass-dsl >= 4.5.2 name -- older releases spelled it
# `_maybe_recast_from_f4` (f4-only). Gate the whole module on the installed version so an older (or
# absent) install skips cleanly instead of erroring at collection with an ImportError, and guard the
# recipes import (it's used in a parametrize decorator, which runs at collection time regardless of
# the skip mark).
_MIN_CUTEDSL = (4, 5, 2)
try:
    _cutedsl_version = tuple(
        int(x) for x in importlib.metadata.version("nvidia-cutlass-dsl").split(".")[:3]
    )
except (ImportError, importlib.metadata.PackageNotFoundError):
    _cutedsl_version = None

HAS_CUTEDSL = _cutedsl_version is not None and _cutedsl_version >= _MIN_CUTEDSL

if HAS_CUTEDSL:
    from quant_cast_bench.quant_cast_cute.recipes import ALL_RECIPES
else:
    ALL_RECIPES = []

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_CUTEDSL,
    reason=(
        f"requires CUDA and nvidia-cutlass-dsl >= {'.'.join(map(str, _MIN_CUTEDSL))} "
        f"(found {'.'.join(map(str, _cutedsl_version)) if _cutedsl_version else 'none'})"
    ),
)

_MAX_MISMATCH_FRAC = 0.01


@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_cute_matches_reference(name, recipe):
    # the CuTeDSL kernel should reproduce the gold reference bit-for-bit; where the hardware cvt
    # rounding legitimately differs (fp4/e8m0 ties), accept a valid quantization with tiny divergence.
    torch.manual_seed(0)
    inputs = recipe.example_input_fn(512, 512)

    # flex_tile_map framework kwargs naming the tile's global origin + parent row stride. The test
    # runs the whole tensor as one tile, so origin = (0, 0) and num_col = full width. These are needed
    # by the global-offsets SR *reference* (`sr_bf16_global_f`) to reconstruct each element's global
    # index from a sub-tile; every recipe fn takes **kwargs, and all the cute kernels ignore them
    # (they own their own tiling), so passing them is harmless.
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    cute_outs = recipe.cute_fn(*inputs, **tile_kwargs)

    assert len(cute_outs) == len(ref_outs), f"{name}: output count {len(cute_outs)} != {len(ref_outs)}"
    for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )

    if all(qdata_equal(t, r) for t, r in zip(cute_outs, ref_outs)):
        return  # exact match

    # Legitimate CuTeDSL-vs-PyTorch hardware-rounding differences: fp8/fp4 cast RNE ties, and f32
    # scales computed with the GPU's *approximate* division (~1 ULP). Accept iff the cute output is
    # still a valid quantization AND every output is close: narrow types (fp8/fp4/e8m0) must match
    # bit-for-bit on all but <1% of bytes; float (fp32 scale) outputs must be allclose to ~1 ULP.
    recipe.correctness_fn(inputs, cute_outs)
    if "_sr" in name:
        # stochastic rounding: the cute kernel draws its dither from a hand-written in-kernel Philox
        # keyed on the global flat index, which cannot bit-match the reference's torch Philox -- only
        # the SR property (checked above via the gold correctness_fn) is well-defined, so a per-element
        # bound is meaningless (like triton).
        return
    for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
        if t.dtype in (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, torch.float8_e4m3fn):
            frac = mismatch_fraction(t, r)
            assert frac < _MAX_MISMATCH_FRAC, (
                f"{name} output {i}: {frac:.3%} of narrow-type elements differ -- likely a real bug"
            )
        else:
            assert torch.allclose(t.float(), r.float(), rtol=2e-6, atol=1e-20), (
                f"{name} output {i}: float output not within ~1 ULP of reference (max rel "
                f"{((t.float() - r.float()).abs() / r.float().abs().clamp(min=1e-30)).max().item():.2e})"
            )
