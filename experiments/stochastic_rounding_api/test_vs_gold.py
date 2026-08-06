"""Score the SR implementations against a GOLDEN reference of *correct* stochastic rounding.

The eager references in `api.py` are built to match the Triton kernels bit-for-bit, so by
construction they cannot flag an implementation that is faithfully *wrong* (e.g. the software fp8
path dithers a fixed 20-bit field, which is too narrow at fp8 subnormals -- see the note about the
double-rounding bug). This file adds a standalone golden reference that defines correct SR
independently of any kernel, and scores every implementation against it.

Correct SR: for input x bracketed by dtype grid neighbors lo <= x <= hi, round to hi with
probability frac = (x - lo)/(hi - lo), else to lo. Then E[out] = x exactly (unbiased), for normals,
subnormals, and the bottom bin alike.

This can be done entirely in floating point, EXACTLY, in fp64: bf16 and float8_e4m3fn have far
fewer significand/exponent bits than fp64, so every grid point and every gap is exactly
representable and `frac` is exact. But correct SR is a *probability*, not a single output, so we
score implementations STATISTICALLY -- empirical P(up) vs the exact frac -- plus hard on-grid and
bias checks. It is not a bitwise diff (each impl draws randomness through its own Philox layout).

Two estimators of P(up):
  * repeated-value probes -- pick values at known fractional positions inside each regime (normal
    binades, the three fp8 subnormal binades, the 0..2^-9 bottom bin, near-saturation), repeat each
    M times, measure the empirical round-up fraction. Sharp and regime-targeted.
  * multi-seed over distributions -- run each impl over many seeds on randn (+ subnormal-scaled
    randn), bin elements by frac, compare aggregated P(up) to the bin's mean frac. Broad coverage.

Run under pytest (`pytest test_vs_gold.py -q`) or directly (`python test_vs_gold.py`) for a table.
"""

import os
import sys

import pytest
import torch
import torch.func._random as prng

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import to  # noqa: E402

_CUDA = torch.cuda.is_available()
_SM100 = _CUDA and torch.cuda.get_device_capability() == (10, 0)

DTYPES = [torch.bfloat16, torch.float8_e4m3fn]
FRAC_POS = (0.1, 0.25, 0.5, 0.75, 0.9)  # fractional positions probed inside each (lo, hi) bin
PROBE_M = 1 << 14                       # copies per probe value; P(up) std ~ 0.5/sqrt(M) ~ 0.004
PROB_TOL = 0.02                         # ~5 sigma of the probe sampling noise
MULTISEED_SEEDS = tuple(range(8))       # seeds averaged in the multi-seed estimator


# --- golden reference: exact correct-SR in fp64 -------------------------------------------------
_GRID_CACHE = {}


def _grid(dtype, device):
    """All finite representable values of `dtype` as a sorted signed fp64 grid. Enumerates every bit
    pattern (256 for fp8, 2^16 for bf16), reinterprets as the dtype, widens to fp64 (exact -- the
    dtype has far fewer bits than fp64), drops NaN/Inf, dedups (+0/-0 collapse)."""
    ck = (dtype, device.type)
    if ck not in _GRID_CACHE:
        n = 256 if dtype == torch.float8_e4m3fn else 1 << 16
        codes = torch.arange(n, device=device).to(torch.uint8 if n == 256 else torch.int16)
        vals = codes.view(dtype).double()
        _GRID_CACHE[ck] = torch.unique(vals[torch.isfinite(vals)]).sort().values
    return _GRID_CACHE[ck]


def _neighbors_frac(x, grid):
    """For each element of `x`, its bracketing grid neighbors (lo, hi) and frac = (x-lo)/(hi-lo), all
    fp64. Out-of-range magnitudes clamp to the grid ends (saturation); exact grid values get frac 0."""
    xd = x.double().reshape(-1).clamp(grid[0], grid[-1])
    idx = (torch.searchsorted(grid, xd, right=True) - 1).clamp(0, grid.numel() - 2)
    lo, hi = grid[idx], grid[idx + 1]
    return lo, hi, (xd - lo) / (hi - lo)


def _golden_sr(x, dtype, u):
    """The correct-SR oracle: round up to hi iff uniform u < frac, else down to lo. Unbiased by
    construction. `u` is a uniform in [0, 1). Used as a validation baseline sampler."""
    grid = _grid(dtype, x.device)
    lo, hi, frac = _neighbors_frac(x, grid)
    return torch.where(u.reshape(-1) < frac, hi, lo).to(dtype).reshape(x.shape)


# --- implementations under test, as a common sampler(x, seed) -> output surface -----------------
def _make_sampler(kind, dtype):
    if kind == "golden-oracle":
        def s(x, seed):
            g = torch.Generator(device=x.device).manual_seed(int(seed))
            u = torch.rand(x.numel(), generator=g, dtype=torch.float64, device=x.device)
            return _golden_sr(x, dtype, u)
        return s
    rounding, ref = {
        "software-kernel": ("stochastic", False),
        "software-ref": ("stochastic", True),
        "hardware-kernel": ("stochastic-nvidia-sm100", False),
        "hardware-ref": ("stochastic-nvidia-sm100", True),
    }[kind]

    def s(x, seed):
        return to(x, dtype, rounding=rounding, key=prng.key(int(seed), device=x.device), _reference_impl=ref)
    return s


# --- estimator (A): repeated-value probes -------------------------------------------------------
def _regime_anchors(dtype):
    """(regime, anchor value) pairs. Each anchor's grid slot gives a real adjacent (lo, hi)."""
    if dtype == torch.float8_e4m3fn:
        return [
            ("normal", 1.0), ("normal", 64.0), ("normal", -1.0),
            ("subnormal", 2.0**-9), ("subnormal", 2.0**-8), ("subnormal", 2.0**-7),  # E = -9, -8, -7
            ("bottom", 0.0),          # bin [0, 2^-9)
            ("saturation", 448.0),    # top finite pair (416, 448)
        ]
    return [("normal", 1.0), ("normal", 64.0), ("normal", -1.0)]  # bf16 subnormals (~2^-133) skipped


def _probe_prob_error(sampler, dtype, seed=0, M=PROBE_M):
    """Score by repeated-value probes. Returns a list of (regime, abs prob error, on-grid violations),
    one entry per probe value."""
    device = torch.device("cuda")
    grid = _grid(dtype, device)
    xs, regimes = [], []
    for regime, a in _regime_anchors(dtype):
        i = int(torch.searchsorted(grid, torch.tensor(a, dtype=torch.float64, device=device)).item())
        i = min(i, grid.numel() - 2)          # top-anchor -> use the last pair
        lo, hi = grid[i].item(), grid[i + 1].item()
        for t in FRAC_POS:
            xs.append(lo + t * (hi - lo))     # fp64 target; the fp32 cast below shifts it slightly
            regimes.append(regime)
    xvals = torch.tensor(xs, dtype=torch.float32, device=device)
    lo_g, hi_g, frac_g = _neighbors_frac(xvals, grid)  # exact golden frac of the fp32-rounded probes
    x = xvals.repeat_interleave(M)                      # each probe value in a contiguous block of M
    out = sampler(x, seed).float().double().reshape(len(xs), M)
    results = []
    for j in range(len(xs)):
        row = out[j]
        p_up = (row == hi_g[j]).double().mean().item()
        viol = int(((row != lo_g[j]) & (row != hi_g[j])).sum().item())
        results.append((regimes[j], abs(p_up - frac_g[j].item()), viol))
    return results


# --- estimator (B): multi-seed over distributions -----------------------------------------------
def _multiseed_prob_error(sampler, dtype, seeds=MULTISEED_SEEDS):
    """Score over randn (+ subnormal-scaled randn) across `seeds`. Bins elements by frac and compares
    each bin's aggregated P(up) to its mean frac. Returns (max bin prob error, on-grid violations,
    mean bias in ulps)."""
    device = torch.device("cuda")
    grid = _grid(dtype, device)
    gmax = grid.abs().max().float().item()
    g = torch.Generator(device=device).manual_seed(12345)
    n = 1 << 20
    x = torch.cat([
        torch.randn(n, generator=g, dtype=torch.float32, device=device),
        torch.randn(n, generator=g, dtype=torch.float32, device=device) * (2.0**-8),  # fill subnormals
    ])
    x = x[x.abs() < gmax]                     # keep in-range so both neighbors are finite
    x = x[: (x.numel() // 4) * 4].contiguous()  # divisible by 4 (covers bf16's %2 and fp8's %4)
    lo, hi, frac = _neighbors_frac(x, grid)
    up_count = torch.zeros_like(frac)
    bias_sum, viol = 0.0, 0
    for s in seeds:
        out = sampler(x, s).float().double()
        up = out == hi
        up_count += up.double()
        viol += int(((out != lo) & (out != hi)).sum().item())
        bias_sum += (up.double() - frac).sum().item()  # (up - frac) is the per-draw ulp bias
    p_up = up_count / len(seeds)
    edges = torch.linspace(0, 1, 21, device=device, dtype=torch.float64)[1:-1]
    b = torch.bucketize(frac, edges)
    max_err = 0.0
    for k in range(20):
        m = b == k
        if int(m.sum().item()) * len(seeds) < 5000:  # too few draws for a stable estimate
            continue
        max_err = max(max_err, abs(p_up[m].mean().item() - frac[m].mean().item()))
    return max_err, viol, bias_sum / (x.numel() * len(seeds))


# --- tests --------------------------------------------------------------------------------------
@pytest.mark.skipif(not _CUDA, reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", DTYPES)
def test_golden_oracle_unbiased(dtype):
    # self-consistency: the golden oracle, sampled with its own uniforms, must score ~0 error under
    # BOTH estimators everywhere. Validates the grid/frac math and the scoring harness.
    s = _make_sampler("golden-oracle", dtype)
    for regime, err, viol in _probe_prob_error(s, dtype):
        assert viol == 0 and err < PROB_TOL, (regime, err, viol)
    max_err, viol, bias = _multiseed_prob_error(s, dtype)
    assert viol == 0 and max_err < PROB_TOL and abs(bias) < PROB_TOL, (max_err, viol, bias)


@pytest.mark.skipif(not _SM100, reason="cvt.rs is Blackwell-only; requires cuda capability (10, 0)")
@pytest.mark.parametrize("dtype", DTYPES)
def test_hardware_matches_golden(dtype):
    # the cvt.rs hardware path IS correct SR across all regimes, including fp8 subnormals + bottom bin.
    s = _make_sampler("hardware-kernel", dtype)
    for regime, err, viol in _probe_prob_error(s, dtype):
        assert viol == 0 and err < PROB_TOL, (regime, err, viol)
    max_err, viol, bias = _multiseed_prob_error(s, dtype)
    assert viol == 0 and max_err < PROB_TOL and abs(bias) < PROB_TOL, (max_err, viol, bias)


@pytest.mark.skipif(not _CUDA, reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", DTYPES)
def test_hardware_ref_matches_golden(dtype):
    # the eager cvt.rs reference (exponent-dependent drop width + bottom-bin frac) is genuinely
    # correct SR, not merely kernel-matching. Runs on any CUDA GPU (no cvt.rs needed).
    s = _make_sampler("hardware-ref", dtype)
    for regime, err, viol in _probe_prob_error(s, dtype):
        assert viol == 0 and err < PROB_TOL, (regime, err, viol)
    max_err, viol, bias = _multiseed_prob_error(s, dtype)
    assert viol == 0 and max_err < PROB_TOL and abs(bias) < PROB_TOL, (max_err, viol, bias)


@pytest.mark.skipif(not _CUDA, reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", DTYPES)
def test_software_matches_golden_normals(dtype):
    # the software fixed-width dither is correct SR in the NORMAL regime (what it is designed for).
    s = _make_sampler("software-kernel", dtype)
    for regime, err, viol in _probe_prob_error(s, dtype):
        if regime != "normal":
            continue
        assert viol == 0 and err < PROB_TOL, (regime, err, viol)


@pytest.mark.skipif(not _CUDA, reason="needs a CUDA device")
def test_golden_detects_fp8_software_subnormal_bias():
    # the payoff: the golden must FLAG the software fp8 path at subnormals / bottom bin, where its
    # fixed 20-bit dither is too narrow (near-deterministic rounding -> P(up) far from frac). This is
    # the bug the bitwise-matched reference in api.py hides. If software fp8 subnormal SR is ever
    # fixed, this test should be updated to expect a pass instead.
    s = _make_sampler("software-kernel", torch.float8_e4m3fn)
    errs = [err for regime, err, _ in _probe_prob_error(s, torch.float8_e4m3fn)
            if regime in ("subnormal", "bottom")]
    assert max(errs) > PROB_TOL, f"expected the scorer to flag fp8 software SR at subnormals, got {max(errs):.4f}"


# --- runnable report ----------------------------------------------------------------------------
def _report():
    kinds = ["golden-oracle", "software-kernel", "software-ref", "hardware-ref", "hardware-kernel"]
    print(f"{'impl':<16}{'dtype':<12}{'probe_max':>10}{'ms_max':>9}{'viol':>7}{'bias':>10}  result")
    print("-" * 72)
    for dtype in DTYPES:
        for kind in kinds:
            s = _make_sampler(kind, dtype)
            try:
                probe = _probe_prob_error(s, dtype)
                pmax = max(e for _, e, _ in probe)
                pviol = sum(v for _, _, v in probe)
                mmax, mviol, bias = _multiseed_prob_error(s, dtype)
            except RuntimeError as e:
                print(f"{kind:<16}{str(dtype).replace('torch.',''):<12}  skipped: {str(e).splitlines()[0][:36]}")
                continue
            viol = pviol + mviol
            ok = pmax < PROB_TOL and mmax < PROB_TOL and viol == 0
            print(f"{kind:<16}{str(dtype).replace('torch.',''):<12}{pmax:>10.4f}{mmax:>9.4f}"
                  f"{viol:>7}{bias:>10.4f}  {'PASS' if ok else 'FAIL'}")
    print("\n(FAIL for software-kernel/ref on fp8 is expected: fixed-width dither is wrong at subnormals.)")


if __name__ == "__main__":
    if not _CUDA:
        raise SystemExit("needs a CUDA device")
    _report()
