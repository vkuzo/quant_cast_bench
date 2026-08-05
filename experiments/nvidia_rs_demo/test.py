"""Verify each `cvt.rs` demo kernel in api.py performs correct stochastic rounding.

For every dtype we cast a large CONSTANT fp32 tensor whose value `v` lies strictly between two
representable neighbors `lo` < v < `hi` of the target format, and check SR's defining properties
(mirrors the repo's canonical `_sr_bf16_unbiased_correctness`):

  1. only-two-neighbors: every output is either `lo` or `hi` (never anything else, never a value
     from a different input) -- this also proves the kernel actually reads its input.
  2. unbiased: the mean of the outputs equals `v` (E[SR(v)] = v), so the fraction rounded up is
     ~= p = (v - lo) / (hi - lo).
  3. deterministic: the same seed reproduces the same bits.

Run under pytest (`pytest nvidia_rs_demo/test.py -q`) or directly (`python nvidia_rs_demo/test.py`).
Requires a Blackwell GPU (sm_100) -- `cvt.rs` is Blackwell-only PTX.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import rs_bf16, rs_e2m1, rs_e4m3, rs_e5m2, rs_f16  # noqa: E402

CUDA_OK = torch.cuda.is_available()
SM100 = CUDA_OK and torch.cuda.get_device_capability() == (10, 0)

try:
    import pytest

    requires_sm100 = pytest.mark.skipif(
        not SM100, reason="cvt.rs emits Blackwell-only PTX; requires cuda capability (10, 0)"
    )
except ImportError:  # allow running as a plain script without pytest
    def requires_sm100(fn):
        return fn


N = 1 << 22  # 4,194,304 elements: divisible by 2 and 4, large enough for a tight mean estimate
SEEDS = (0, 1, 2, 12345)

# e2m1 (fp4) positive finite grid: value magnitude indexed by the low 3 bits of the code.
FP4_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

_INT_VIEW = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float8_e4m3fn: torch.int8,
    torch.float8_e5m2: torch.int8,
}


def _step(t, direction):
    """Next representable value away from (+1) or toward (-1) zero, for a positive `t` (shape (1,))."""
    iv = _INT_VIEW[t.dtype]
    i = t.view(iv) + direction
    return i.to(iv).view(t.dtype).float().item()


def _bracket_torch(v, dt):
    near = torch.tensor([v], dtype=dt).float().item()
    assert near != v, f"{v} is exactly representable in {dt}; pick a value between grid points"
    if near > v:
        hi, lo = near, _step(torch.tensor([near], dtype=dt), -1)
    else:
        lo, hi = near, _step(torch.tensor([near], dtype=dt), +1)
    assert lo < v < hi, f"{dt}: bracket failed lo={lo} v={v} hi={hi}"
    return lo, hi


def _bracket_fp4(v):
    lo = max(g for g in FP4_GRID if g < v)
    hi = min(g for g in FP4_GRID if g > v)
    assert lo < v < hi
    return lo, hi


def _decode_fp4(packed):
    """Decode a packed float4_e2m1fn_x2 tensor to fp32 values (low nibble first)."""
    b = packed.view(torch.uint8).reshape(-1).to(torch.int32)
    codes = torch.stack([b & 0xF, (b >> 4) & 0xF], dim=-1).reshape(-1)
    mag = torch.tensor(FP4_GRID, device=codes.device)[codes & 0x7]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    return mag * sign


# (name, kernel_fn, dtype_kind, test value v)
CONFIGS = [
    ("f16", rs_f16, torch.float16, 1.0 + 0.0003),
    ("bf16", rs_bf16, torch.bfloat16, 1.0 + 0.003),
    ("e4m3", rs_e4m3, torch.float8_e4m3fn, 1.0 + 0.04),
    ("e5m2", rs_e5m2, torch.float8_e5m2, 1.0 + 0.08),
    ("e2m1", rs_e2m1, "fp4", 1.0 + 0.16),
]


def _neighbors(kind, v):
    return _bracket_fp4(v) if kind == "fp4" else _bracket_torch(v, kind)


def _decode(out, kind):
    return _decode_fp4(out) if kind == "fp4" else out.float()


def _seed(s):
    """Philox seed as an on-device int32 tensor -- rs_* takes the seed as a tensor, not an int."""
    return torch.tensor([s], dtype=torch.int32, device="cuda")


def _check_one(name, fn, kind, v):
    lo, hi = _neighbors(kind, v)
    x = torch.full((N,), v, dtype=torch.float32, device="cuda")

    means = []
    for seed in SEEDS:
        out = fn(x, _seed(seed))
        dec = _decode(out, kind).reshape(-1)
        assert dec.numel() == N, f"{name}: expected {N} decoded values, got {dec.numel()}"

        uniq = set(dec.unique().tolist())
        assert uniq <= {lo, hi}, f"{name}: outputs {uniq} not within neighbors {{{lo}, {hi}}}"
        means.append(dec.mean().item())

    mean = sum(means) / len(means)
    tol = 0.01 * (hi - lo)
    assert abs(mean - v) < tol, f"{name}: mean {mean:.6f} not ~= v {v:.6f} (tol {tol:.2e}); biased"

    # determinism: same seed -> identical bits
    a, b = fn(x, _seed(SEEDS[0])), fn(x, _seed(SEEDS[0]))
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"{name}: not deterministic"

    p = (v - lo) / (hi - lo)
    return lo, hi, mean, p


@requires_sm100
def test_rs_f16():
    _check_one("f16", rs_f16, torch.float16, 1.0 + 0.0003)


@requires_sm100
def test_rs_bf16():
    _check_one("bf16", rs_bf16, torch.bfloat16, 1.0 + 0.003)


@requires_sm100
def test_rs_e4m3():
    _check_one("e4m3", rs_e4m3, torch.float8_e4m3fn, 1.0 + 0.04)


@requires_sm100
def test_rs_e5m2():
    _check_one("e5m2", rs_e5m2, torch.float8_e5m2, 1.0 + 0.08)


@requires_sm100
def test_rs_e2m1():
    _check_one("e2m1", rs_e2m1, "fp4", 1.0 + 0.16)


def main():
    if not CUDA_OK:
        print("[SKIP] no CUDA device")
        return 0
    if not SM100:
        cap = torch.cuda.get_device_capability()
        print(f"[SKIP] cvt.rs is Blackwell-only (sm_100); this GPU is sm_{cap[0]}{cap[1]}")
        return 0

    print(f"{'dtype':>6} {'lo':>10} {'v':>10} {'hi':>10} {'mean':>10} {'p(up)':>8}  result")
    failed = False
    for name, fn, kind, v in CONFIGS:
        try:
            lo, hi, mean, p = _check_one(name, fn, kind, v)
            print(f"{name:>6} {lo:>10.5f} {v:>10.5f} {hi:>10.5f} {mean:>10.5f} {p:>8.3f}  [PASS]")
        except AssertionError as e:
            print(f"{name:>6}  [FAIL] {e}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
