"""One test case for the Four Over Six (4/6) NVFP4 reference.

Reproduces the worked example from the paper (arXiv:2512.02010): for the block [10, 20, 30, 40],
baseline NVFP4 (always scale-to-6) mis-rounds 30 to 26.67, while 4/6 picks scale-to-4 for the block
and reconstructs it exactly. Asserts 4/6 is exact and strictly better (higher SQNR) than baseline.

Usage: python experiments/four_over_six/test.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quant_cast_bench.quant_cast_gold.recipes import _compute_error  # noqa: E402

from four_over_six import four_over_six_roundtrip, nvfp4_roundtrip  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _check() -> bool:
    x = torch.tensor([10.0, 20.0, 30.0, 40.0], device=DEV)  # a single 4-element block
    x46 = four_over_six_roundtrip(x, block_size=4)          # picks scale-4 -> exact
    xnv = nvfp4_roundtrip(x, block_size=4)                  # scale-6 -> 30 mis-rounds
    sqnr46 = _compute_error(x, x46)
    sqnrnv = _compute_error(x, xnv)

    exact_46 = torch.allclose(x46, x, atol=1e-4)
    baseline_lossy = not torch.allclose(xnv, x, atol=1e-1)
    better = sqnr46 > sqnrnv
    ok = exact_46 and baseline_lossy and better

    print(f"[{'PASS' if ok else 'FAIL'}] four_over_six worked example [10,20,30,40]")
    if not ok:
        print(f"  x            = {x.tolist()}")
        print(f"  4/6 dequant  = {x46.tolist()}  (exact={exact_46}, sqnr={sqnr46.item():.2f} dB)")
        print(f"  nvfp4 dequant= {xnv.tolist()}  (lossy={baseline_lossy}, sqnr={sqnrnv.item():.2f} dB)")
        print(f"  4/6 better than baseline: {better}")
    return ok


def main() -> int:
    ok = _check()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
