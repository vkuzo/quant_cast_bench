"""Verify the `to()` stochastic-rounding API (fp32 -> bf16) and its eager reference.

Checks, per mode:
  * rtne              -- bit-identical to the native `x.to(bfloat16)` and to the reference.
  * stochastic        -- unbiased & two-neighbor (the canonical SR property), tile-invariant across
                         block sizes, deterministic given a key, and statistically equal to (but not
                         bit-identical to) the eager reference.
  * stochastic-approx -- (Blackwell only) unbiased/two-neighbor + deterministic; the reference for
                         this mode raises NotImplementedError. On non-Blackwell the API raises.
  * generator path    -- valid SR, successive calls differ (state advances), reseeding reproduces.

Run under pytest (`pytest test.py -q`) or directly (`python test.py`).
"""

import os
import sys

import torch
import torch.func._random as prng

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import to  # noqa: E402
from kernels import sr_bf16_software_triton  # noqa: E402
from reference import to_reference  # noqa: E402

CUDA_OK = torch.cuda.is_available()
SM100 = CUDA_OK and torch.cuda.get_device_capability() == (10, 0)

try:
    import pytest

    requires_cuda = pytest.mark.skipif(not CUDA_OK, reason="needs a CUDA device")
    requires_sm100 = pytest.mark.skipif(
        not SM100, reason="cvt.rs emits Blackwell-only PTX; requires cuda capability (10, 0)"
    )
except ImportError:  # allow running as a plain script without pytest

    def requires_cuda(fn):
        return fn

    def requires_sm100(fn):
        return fn


N = 1 << 22  # 4,194,304 elements: divisible by 2 and 4, large enough for a tight mean estimate
V = 1.0 + 0.003  # constant test value strictly between two bf16 grid points (spacing 2**-7 near 1)


def _bf16_neighbors(v):
    """The two bf16 grid points bracketing v (spacing 2**-7 in [1, 2))."""
    lo = torch.tensor(v, dtype=torch.bfloat16).float().item()  # RTN neighbor (round down)
    hi = torch.tensor(v + 2**-7, dtype=torch.bfloat16).float().item()
    assert lo < v < hi, f"v={v} not strictly between bf16 neighbors ({lo}, {hi})"
    return lo, hi


def _check_sr(out, v):
    """SR's defining property: every output is one of the two bracketing bf16 grid points, and the
    mean is ~= v (unbiased). Returns the sample mean."""
    lo, hi = _bf16_neighbors(v)
    dec = out.float().reshape(-1)
    uniq = set(dec.unique().tolist())
    assert uniq <= {lo, hi}, f"outputs {uniq} not within neighbors {{{lo}, {hi}}}"
    mean = dec.mean().item()
    assert abs(mean - v) < 1e-3, f"mean {mean:.6f} not ~= v {v:.6f}; biased"
    return mean


def _x():
    return torch.full((N,), V, dtype=torch.float32, device="cuda")


def _key(seed=0):
    return prng.key(seed, device="cuda")


# --- rtne --------------------------------------------------------------------------------------
@requires_cuda
def test_rtne_matches_native_cast():
    x = _x()
    out = to(x, torch.bfloat16)  # default rounding
    assert torch.equal(out.view(torch.int16), x.to(torch.bfloat16).view(torch.int16))
    ref = to_reference(x, torch.bfloat16, rounding="rtne")
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@requires_cuda
def test_rtne_rejects_randomness():
    x = _x()
    for kw in ({"key": _key()}, {"generator": torch.Generator(device="cuda")}):
        try:
            to(x, torch.bfloat16, rounding="rtne", **kw)
            raise AssertionError("rtne should reject a randomness source")
        except ValueError:
            pass


# --- software stochastic -----------------------------------------------------------------------
@requires_cuda
def test_stochastic_unbiased():
    out = to(_x(), torch.bfloat16, rounding="stochastic", key=_key())
    _check_sr(out, V)


@requires_cuda
def test_stochastic_reference_unbiased():
    ref = to_reference(_x(), torch.bfloat16, rounding="stochastic", key=_key())
    _check_sr(ref, V)


@requires_cuda
def test_stochastic_tile_invariant():
    x = _x()
    seed = _key().reshape(-1)[:1].view(torch.int32)
    a = sr_bf16_software_triton(x, seed, block=256)
    b = sr_bf16_software_triton(x, seed, block=1024)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), "not tile-invariant across blocks"


@requires_cuda
def test_stochastic_deterministic():
    x = _x()
    a = to(x, torch.bfloat16, rounding="stochastic", key=_key(7))
    b = to(x, torch.bfloat16, rounding="stochastic", key=_key(7))
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@requires_cuda
def test_stochastic_matches_reference_statistically():
    x = _x()
    out = to(x, torch.bfloat16, rounding="stochastic", key=_key())
    ref = to_reference(x, torch.bfloat16, rounding="stochastic", key=_key())
    # both are unbiased SR (means ~= V) ...
    assert abs(_check_sr(out, V) - _check_sr(ref, V)) < 1e-3
    # ... but they use different RNG (raw low-16 bits vs float top-16 bits), so NOT bit-identical.
    assert not torch.equal(out.view(torch.int16), ref.view(torch.int16))


# --- generator path ----------------------------------------------------------------------------
@requires_cuda
def test_generator_path():
    x = _x()
    g = torch.Generator(device="cuda").manual_seed(123)
    a = to(x, torch.bfloat16, rounding="stochastic", generator=g)
    b = to(x, torch.bfloat16, rounding="stochastic", generator=g)
    _check_sr(a, V)
    _check_sr(b, V)
    # drawing the seed advances the generator, so successive calls differ ...
    assert not torch.equal(a.view(torch.int16), b.view(torch.int16))
    # ... and reseeding reproduces the sequence.
    g.manual_seed(123)
    a2 = to(x, torch.bfloat16, rounding="stochastic", generator=g)
    assert torch.equal(a.view(torch.int16), a2.view(torch.int16))


@requires_cuda
def test_requires_exactly_one_source():
    x = _x()
    for kw in ({}, {"key": _key(), "generator": torch.Generator(device="cuda")}):
        try:
            to(x, torch.bfloat16, rounding="stochastic", **kw)
            raise AssertionError("stochastic needs exactly one of key=/generator=")
        except ValueError:
            pass


# --- hardware stochastic-approx ----------------------------------------------------------------
@requires_sm100
def test_stochastic_approx_unbiased_and_deterministic():
    x = _x()
    out = to(x, torch.bfloat16, rounding="stochastic-approx", key=_key())
    _check_sr(out, V)
    a = to(x, torch.bfloat16, rounding="stochastic-approx", key=_key(9))
    b = to(x, torch.bfloat16, rounding="stochastic-approx", key=_key(9))
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), "hardware SR not deterministic"


@requires_cuda
def test_stochastic_approx_reference_unavailable():
    try:
        to_reference(_x(), torch.bfloat16, rounding="stochastic-approx", key=_key())
        raise AssertionError("stochastic-approx reference should be unavailable")
    except NotImplementedError:
        pass


@requires_cuda
def test_stochastic_approx_gated_off_blackwell():
    if SM100:
        return  # this GPU supports cvt.rs; nothing to gate
    try:
        to(_x(), torch.bfloat16, rounding="stochastic-approx", key=_key())
        raise AssertionError("stochastic-approx should be gated to cuda capability (10, 0)")
    except RuntimeError:
        pass


# --- script runner -----------------------------------------------------------------------------
def main():
    if not CUDA_OK:
        print("[SKIP] no CUDA device")
        return 0

    tests = [
        ("rtne_matches_native_cast", test_rtne_matches_native_cast),
        ("rtne_rejects_randomness", test_rtne_rejects_randomness),
        ("stochastic_unbiased", test_stochastic_unbiased),
        ("stochastic_reference_unbiased", test_stochastic_reference_unbiased),
        ("stochastic_tile_invariant", test_stochastic_tile_invariant),
        ("stochastic_deterministic", test_stochastic_deterministic),
        ("stochastic_matches_reference_statistically", test_stochastic_matches_reference_statistically),
        ("generator_path", test_generator_path),
        ("requires_exactly_one_source", test_requires_exactly_one_source),
        ("stochastic_approx_reference_unavailable", test_stochastic_approx_reference_unavailable),
        ("stochastic_approx_gated_off_blackwell", test_stochastic_approx_gated_off_blackwell),
    ]
    if SM100:
        tests.append(
            ("stochastic_approx_unbiased_and_deterministic", test_stochastic_approx_unbiased_and_deterministic)
        )
    else:
        cap = torch.cuda.get_device_capability()
        print(f"[note] sm_{cap[0]}{cap[1]}: cvt.rs (stochastic-approx) kernel test skipped; gate test runs")

    failed = False
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
