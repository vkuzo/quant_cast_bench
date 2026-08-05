"""Verify the `to()` stochastic-rounding API (fp32 -> bf16 / float8_e4m3fn) and its eager reference.

Checks, per mode:
  * rtne              -- bit-identical to the native `x.to(dtype)` and to the reference.
  * stochastic        -- (both output dtypes) unbiased & two-neighbor (the canonical SR property),
                         tile-invariant across block sizes, deterministic given a key, and bit-
                         identical to the eager reference (both read the same raw Philox words via
                         prng.bits).
  * stochastic-nvidia-sm100 -- (Blackwell only, both output dtypes) unbiased/two-neighbor +
                         deterministic; the reference for this mode raises NotImplementedError. On
                         non-Blackwell the API raises.

Run under pytest (`pytest test.py -q`) or directly (`python test.py`).
"""

import os
import sys

import pytest
import torch
import torch.func._random as prng

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import to  # noqa: E402
from kernels import sr_bf16_software_triton, sr_fp8_software_triton  # noqa: E402

N = 1 << 22  # 4,194,304 elements: large enough for a tight mean estimate; divisible by 2 and 4


def _check_sr(out, x, dtype):
    """SR's defining properties, checked per element against the fp32 input `x`: every output is one
    of the two `dtype` grid points bracketing its own input value, and SR is unbiased (the mean
    rounding error is ~= 0). The two neighbors come from the dtype's own finite grid (all 256 fp8
    codes / all 2^16 bf16 codes, cast to fp32, then searchsorted) -- no hardcoded step, and it works
    for float8_e4m3fn where torch.nextafter does not. Returns the mean rounding error."""
    n_codes = 256 if dtype == torch.float8_e4m3fn else 1 << 16  # every bit pattern of the dtype
    codes = torch.arange(n_codes, device=x.device).to(torch.uint8 if n_codes == 256 else torch.int16)
    vals = codes.view(dtype).float()
    grid = torch.unique(vals[torch.isfinite(vals)]).sort().values  # sorted finite grid, as fp32
    x = x.reshape(-1)
    dec = out.float().reshape(-1)
    xc = x.clamp(grid[0].item(), grid[-1].item())
    idx = torch.searchsorted(grid, xc).clamp(max=grid.numel() - 1)  # first grid point >= xc
    up = grid[idx]                       # ceil neighbor
    down = grid[(idx - 1).clamp(min=0)]  # floor neighbor
    on_grid = (dec == down) | (dec == up)
    assert on_grid.all(), f"{(~on_grid).sum().item()} outputs are not a bracketing {dtype} neighbor"
    err = (dec - x).mean().item()
    assert abs(err) < 1e-3, f"mean rounding error {err:.3e} too large; SR is biased"
    return err


def _x():
    # random fp32 data (seeded for reproducibility): exercises SR across signs and magnitudes.
    g = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(N, generator=g, dtype=torch.float32, device="cuda")


def _key(seed=0):
    return prng.key(seed, device="cuda")


# --- software stochastic (both output dtypes: bf16 and fp8_e4m3fn) -----------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_stochastic_unbiased(dtype):
    x = _x()
    out = to(x, dtype, rounding="stochastic", key=_key())
    _check_sr(out, x, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_stochastic_reference_unbiased(dtype):
    x = _x()
    ref = to(x, dtype, rounding="stochastic", key=_key(), _reference_impl=True)
    _check_sr(ref, x, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_stochastic_tile_invariant(dtype):
    x = _x()
    seed = _key().reshape(-1)[:1].view(torch.int32)
    launch = sr_bf16_software_triton if dtype == torch.bfloat16 else sr_fp8_software_triton
    a = launch(x, seed, block=256)
    b = launch(x, seed, block=1024)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), "not tile-invariant across blocks"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_stochastic_deterministic(dtype):
    x = _x()
    a = to(x, dtype, rounding="stochastic", key=_key(7))
    b = to(x, dtype, rounding="stochastic", key=_key(7))
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_stochastic_matches_reference_bitwise(dtype):
    x = _x()
    out = to(x, dtype, rounding="stochastic", key=_key())
    ref = to(x, dtype, rounding="stochastic", key=_key(), _reference_impl=True)
    # the eager reference reads the same raw Philox words (prng.bits) the kernel's randint4x lays
    # across elements, so it is BIT-IDENTICAL to the kernel, not merely statistically equal.
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
    _check_sr(out, x, dtype)  # and is still valid unbiased SR


# --- hardware stochastic-nvidia-sm100 (both output dtypes: bf16 via bf16x2, fp8 via e4m3x4) -----
@pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)), reason="cvt.rs emits Blackwell-only PTX; requires cuda capability (10, 0)")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_stochastic_nvidia_sm100_unbiased_and_deterministic(dtype):
    x = _x()
    out = to(x, dtype, rounding="stochastic-nvidia-sm100", key=_key())
    _check_sr(out, x, dtype)
    a = to(x, dtype, rounding="stochastic-nvidia-sm100", key=_key(9))
    b = to(x, dtype, rounding="stochastic-nvidia-sm100", key=_key(9))
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), "hardware SR not deterministic"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_stochastic_nvidia_sm100_reference_unavailable(dtype):
    try:
        to(_x(), dtype, rounding="stochastic-nvidia-sm100", key=_key(), _reference_impl=True)
        raise AssertionError("stochastic-nvidia-sm100 reference should be unavailable")
    except NotImplementedError:
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_stochastic_nvidia_sm100_gated_off_blackwell(dtype):
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0):
        return  # this GPU supports cvt.rs; nothing to gate
    try:
        to(_x(), dtype, rounding="stochastic-nvidia-sm100", key=_key())
        raise AssertionError("stochastic-nvidia-sm100 should be gated to cuda capability (10, 0)")
    except RuntimeError:
        pass
