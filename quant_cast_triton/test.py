"""Correctness tests for the Triton quant-cast recipes: each `triton_fn` must reproduce its
gold `pt_ref_fn`'s outputs. Inputs come from the recipe's (inherited) `example_input_fn`.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_triton.recipes import ALL_RECIPES

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

    ref_outs = recipe.pt_ref_fn(*inputs)
    tri_outs = recipe.triton_fn(*inputs)

    assert len(tri_outs) == len(ref_outs), f"{name}: output count {len(tri_outs)} != {len(ref_outs)}"
    for i, (t, r) in enumerate(zip(tri_outs, ref_outs)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )

    if all(_qdata_equal(t, r) for t, r in zip(tri_outs, ref_outs)):
        return  # exact match to the reference (the common case)

    # Some outputs differ. The only legitimate source is Triton-vs-PyTorch RNE tie-breaking in the
    # fp8/fp4 cast (the pre-cast fp32 math is identical to the reference). Accept iff (a) the Triton
    # outputs are still a valid quantization -- dequant to x above the recipe's SQNR threshold, via
    # the gold correctness_fn -- and (b) the byte-level divergence is tiny (guards against real bugs).
    recipe.correctness_fn(inputs, tri_outs)
    for i, (t, r) in enumerate(zip(tri_outs, ref_outs)):
        frac = _mismatch_fraction(t, r)
        assert frac < _MAX_MISMATCH_FRAC, (
            f"{name} output {i}: {frac:.3%} of elements differ from reference -- too many for "
            f"RNE ties, likely a real bug"
        )
