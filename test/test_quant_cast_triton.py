"""Correctness tests for the Triton quant-cast recipes: each `triton_fn` must reproduce its
gold `pt_ref_fn`'s outputs. Inputs come from the recipe's (inherited) `example_input_fn`.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_bench.quant_cast_triton.recipes import ALL_RECIPES

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


# fraction of the RNE-tie divergence between Triton's and PyTorch's fp8/fp4 casts we tolerate
# before treating it as a real bug (see the fallback in the test below).
_MAX_MISMATCH_FRAC = 0.01


def _as_bytes_or_fp32(t):
    # packed fp4 / e8m0 have no lossless float cast, so compare their raw bytes; everything else
    # (fp8_e4m3, fp32) casts to fp32 (lossless).
    if t.dtype in (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu):
        return t.view(torch.uint8)
    return t.to(torch.float32)


def _qdata_equal(a, b):
    return torch.equal(_as_bytes_or_fp32(a), _as_bytes_or_fp32(b))


def _mismatch_fraction(a, b):
    av, bv = _as_bytes_or_fp32(a), _as_bytes_or_fp32(b)
    return (av != bv).float().mean().item()


@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_triton_matches_reference(name, recipe):
    # the Triton kernel should reproduce the gold reference bit-for-bit (identical fp32 math + RNE
    # cast). example_input_fn builds the full positional inputs (x, *aux).
    torch.manual_seed(0)
    inputs = recipe.example_input_fn(512, 512)

    # flex_tile_map framework kwargs naming the tile's global origin + parent row stride. The test
    # runs the whole tensor as one tile, so origin = (0, 0) and num_col = full width. These are needed
    # by the global-offsets SR *reference* (`sr_bf16_global_f`) to reconstruct each element's global
    # index from a sub-tile; every recipe fn takes **kwargs, and all the Triton kernels ignore them
    # (they own their own tiling), so passing them is harmless.
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    tri_outs = recipe.triton_fn(*inputs, **tile_kwargs)

    assert len(tri_outs) == len(ref_outs), f"{name}: output count {len(tri_outs)} != {len(ref_outs)}"
    for i, (t, r) in enumerate(zip(tri_outs, ref_outs)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )

    if all(_qdata_equal(t, r) for t, r in zip(tri_outs, ref_outs)):
        return  # exact match to the reference (the common case)

    # Some outputs differ. Two legitimate sources:
    #  (1) stochastic rounding (the *_sr recipes): the Triton kernel draws from its own counter-based
    #      Philox (tl.randint4x), so it cannot bit-match the reference's torch RNG -- only the SR
    #      *property* (unbiased, lands on the two bracketing bf16 grid points) is well-defined. Check
    #      that via the gold correctness_fn and stop; a per-element mismatch bound is meaningless for
    #      an inherently random cast (~2p(1-p) of elements differ between any two independent draws).
    #  (2) Triton-vs-PyTorch RNE tie-breaking in the fp8/fp4 cast (the pre-cast fp32 math is identical
    #      to the reference). Accept iff the Triton outputs are still a valid quantization (gold
    #      correctness_fn) AND the byte-level divergence is tiny (guards against real bugs).
    recipe.correctness_fn(inputs, tri_outs)
    if "_sr" in name:
        return
    for i, (t, r) in enumerate(zip(tri_outs, ref_outs)):
        frac = _mismatch_fraction(t, r)
        assert frac < _MAX_MISMATCH_FRAC, (
            f"{name} output {i}: {frac:.3%} of elements differ from reference -- too many for "
            f"RNE ties, likely a real bug"
        )
