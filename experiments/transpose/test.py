"""Correctness test for the transpose kernels vs a PyTorch reference (y = x.t()).

Runs each kernel on a few shapes (square, non-square, and a non-tile-divisible shape to exercise
masking/predication) and checks bit-exact equality against `x.t()` — a transpose only moves bytes,
so the result must match exactly.

Usage: python experiments/transpose/test.py
"""

import torch

from transpose_cute import transpose_cute
from transpose_cute_v2 import transpose_cute_v2
from transpose_triton import transpose_triton

SHAPES = [
    (16384, 16384),  # large square (benchmark shape)
    (1024, 4096),    # non-square
    (4096, 1024),    # non-square (other way)
    (777, 1279),     # not divisible by any tile size -> exercises masking
]


def _check(name, fn, x):
    ref = x.t().contiguous()
    out = fn(x)
    assert out.shape == ref.shape, f"{name}: shape {out.shape} != {ref.shape}"
    ok = torch.equal(out, ref)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:16s} shape={tuple(x.shape)}")
    if not ok:
        diff = (out.float() - ref.float()).abs()
        print(f"        max_abs_diff={diff.max().item():.3e}  n_mismatch={(diff != 0).sum().item()}")
    return ok


def main():
    all_ok = True
    for M, N in SHAPES:
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        all_ok &= _check("triton", transpose_triton, x)
        all_ok &= _check("cute", transpose_cute, x)
        all_ok &= _check("cute_v2", transpose_cute_v2, x)
    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
