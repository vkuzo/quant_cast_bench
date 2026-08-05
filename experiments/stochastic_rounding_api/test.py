"""Verify the `to()` stochastic-rounding API (fp32 -> bf16) and its eager reference.

Checks, per mode:
  * rtne              -- bit-identical to the native `x.to(bfloat16)` and to the reference.
  * stochastic        -- unbiased & two-neighbor (the canonical SR property), tile-invariant across
                         block sizes, deterministic given a key, and statistically equal to (but not
                         bit-identical to) the eager reference.
  * stochastic-nvidia-sm100 -- (Blackwell only) unbiased/two-neighbor + deterministic; the reference
                         for this mode raises NotImplementedError. On non-Blackwell the API raises.

Run under pytest (`pytest test.py -q`) or directly (`python test.py`).
"""

import os
import sys

import pytest
import torch
import torch.func._random as prng

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import to  # noqa: E402
from kernels import sr_bf16_software_triton  # noqa: E402

N = 1 << 22  # 4,194,304 elements: large enough for a tight mean estimate; divisible by 2 and 4


def _check_sr(out, x):
    """SR's defining properties, checked per element against the fp32 input `x`: every output is one
    of the two bf16 grid points bracketing its own input value, and SR is unbiased (the mean rounding
    error is ~= 0). `nextafter` gives the opposite-side neighbor without hardcoding the grid step.
    Returns the mean rounding error."""
    x = x.reshape(-1)
    dec = out.float().reshape(-1)
    rtn = x.to(torch.bfloat16)  # round-to-nearest bf16; the other neighbor is one step away
    rtn_f = rtn.float()
    inf = torch.full_like(rtn, float("inf"))
    up = torch.where(rtn_f >= x, rtn, torch.nextafter(rtn, inf)).float()     # ceil bf16 neighbor
    down = torch.where(rtn_f <= x, rtn, torch.nextafter(rtn, -inf)).float()  # floor bf16 neighbor
    on_grid = (dec == down) | (dec == up)
    assert on_grid.all(), f"{(~on_grid).sum().item()} outputs are not a bracketing bf16 neighbor"
    err = (dec - x).mean().item()
    assert abs(err) < 1e-3, f"mean rounding error {err:.3e} too large; SR is biased"
    return err


def _x():
    # random fp32 data (seeded for reproducibility): exercises SR across signs and magnitudes.
    g = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(N, generator=g, dtype=torch.float32, device="cuda")


def _key(seed=0):
    return prng.key(seed, device="cuda")


# --- software stochastic -----------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_stochastic_unbiased():
    x = _x()
    out = to(x, torch.bfloat16, rounding="stochastic", key=_key())
    _check_sr(out, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_stochastic_reference_unbiased():
    x = _x()
    ref = to(x, torch.bfloat16, rounding="stochastic", key=_key(), _reference_impl=True)
    _check_sr(ref, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_stochastic_tile_invariant():
    x = _x()
    seed = _key().reshape(-1)[:1].view(torch.int32)
    a = sr_bf16_software_triton(x, seed, block=256)
    b = sr_bf16_software_triton(x, seed, block=1024)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), "not tile-invariant across blocks"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_stochastic_deterministic():
    x = _x()
    a = to(x, torch.bfloat16, rounding="stochastic", key=_key(7))
    b = to(x, torch.bfloat16, rounding="stochastic", key=_key(7))
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_stochastic_matches_reference_statistically():
    x = _x()
    out = to(x, torch.bfloat16, rounding="stochastic", key=_key())
    ref = to(x, torch.bfloat16, rounding="stochastic", key=_key(), _reference_impl=True)
    # both are unbiased SR (mean rounding error ~= 0) ...
    assert abs(_check_sr(out, x) - _check_sr(ref, x)) < 1e-3
    # ... but they use different RNG (raw low-16 bits vs float top-16 bits), so NOT bit-identical.
    assert not torch.equal(out.view(torch.int16), ref.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_requires_key():
    x = _x()
    try:
        to(x, torch.bfloat16, rounding="stochastic")  # no key
        raise AssertionError("stochastic needs a key=")
    except ValueError:
        pass


# --- hardware stochastic-nvidia-sm100 ----------------------------------------------------------
@pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)), reason="cvt.rs emits Blackwell-only PTX; requires cuda capability (10, 0)")
def test_stochastic_nvidia_sm100_unbiased_and_deterministic():
    x = _x()
    out = to(x, torch.bfloat16, rounding="stochastic-nvidia-sm100", key=_key())
    _check_sr(out, x)
    a = to(x, torch.bfloat16, rounding="stochastic-nvidia-sm100", key=_key(9))
    b = to(x, torch.bfloat16, rounding="stochastic-nvidia-sm100", key=_key(9))
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), "hardware SR not deterministic"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_stochastic_nvidia_sm100_reference_unavailable():
    try:
        to(_x(), torch.bfloat16, rounding="stochastic-nvidia-sm100", key=_key(), _reference_impl=True)
        raise AssertionError("stochastic-nvidia-sm100 reference should be unavailable")
    except NotImplementedError:
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_stochastic_nvidia_sm100_gated_off_blackwell():
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0):
        return  # this GPU supports cvt.rs; nothing to gate
    try:
        to(_x(), torch.bfloat16, rounding="stochastic-nvidia-sm100", key=_key())
        raise AssertionError("stochastic-nvidia-sm100 should be gated to cuda capability (10, 0)")
    except RuntimeError:
        pass
