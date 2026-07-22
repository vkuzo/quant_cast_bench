"""Correctness tests for the CuTeDSL quant-cast recipes: each `cute_fn` must reproduce its gold
`pt_ref_fn`'s outputs. Mirrors quant_cast_triton/test.py (same comparison + tolerance fallback).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_bench.quant_cast_cute.recipes import ALL_RECIPES

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

_MAX_MISMATCH_FRAC = 0.01


def _as_bytes_or_fp32(t):
    if t.dtype in (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu):
        return t.view(torch.uint8)
    return t.to(torch.float32)


def _qdata_equal(a, b):
    return torch.equal(_as_bytes_or_fp32(a), _as_bytes_or_fp32(b))


def _mismatch_fraction(a, b):
    av, bv = _as_bytes_or_fp32(a), _as_bytes_or_fp32(b)
    return (av != bv).float().mean().item()


@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_cute_matches_reference(name, recipe):
    # the CuTeDSL kernel should reproduce the gold reference bit-for-bit; where the hardware cvt
    # rounding legitimately differs (fp4/e8m0 ties), accept a valid quantization with tiny divergence.
    torch.manual_seed(0)
    inputs = recipe.example_input_fn(512, 512)

    ref_outs = recipe.pt_ref_fn(*inputs)
    cute_outs = recipe.cute_fn(*inputs)

    assert len(cute_outs) == len(ref_outs), f"{name}: output count {len(cute_outs)} != {len(ref_outs)}"
    for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )

    if all(_qdata_equal(t, r) for t, r in zip(cute_outs, ref_outs)):
        return  # exact match

    # Legitimate CuTeDSL-vs-PyTorch hardware-rounding differences: fp8/fp4 cast RNE ties, and f32
    # scales computed with the GPU's *approximate* division (~1 ULP). Accept iff the cute output is
    # still a valid quantization AND every output is close: narrow types (fp8/fp4/e8m0) must match
    # bit-for-bit on all but <1% of bytes; float (fp32 scale) outputs must be allclose to ~1 ULP.
    recipe.correctness_fn(inputs, cute_outs)
    for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
        if t.dtype in (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, torch.float8_e4m3fn):
            frac = _mismatch_fraction(t, r)
            assert frac < _MAX_MISMATCH_FRAC, (
                f"{name} output {i}: {frac:.3%} of narrow-type elements differ -- likely a real bug"
            )
        else:
            assert torch.allclose(t.float(), r.float(), rtol=2e-6, atol=1e-20), (
                f"{name} output {i}: float output not within ~1 ULP of reference (max rel "
                f"{((t.float() - r.float()).abs() / r.float().abs().clamp(min=1e-30)).max().item():.2e})"
            )
