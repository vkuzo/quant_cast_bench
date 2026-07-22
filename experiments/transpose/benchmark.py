"""Benchmark the transpose kernels against each other (and torch.t().contiguous()).

Reports per kernel: gpu time (ms) and achieved memory bandwidth as a percentage of an assumed
8.0 TB/s peak. A transpose reads M*N elements and writes M*N elements, so bytes moved =
2 * M * N * elem_size.

Usage: python experiments/transpose/benchmark.py
"""

import torch

from transpose_cute import transpose_cute
from transpose_cute_v2 import transpose_cute_v2
from transpose_triton import transpose_triton

PEAK_BW_TBPS = 8.0  # B200 assumed peak, TB/s
M = N = 16384
DTYPE = torch.bfloat16


def bench(fn, x, iters=50, warmup=10):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    x = torch.randn(M, N, dtype=DTYPE, device="cuda")
    elem_size = x.element_size()
    bytes_moved = 2 * M * N * elem_size  # read x + write y

    kernels = {
        "torch (t().contiguous())": lambda x: x.t().contiguous(),
        "triton": transpose_triton,
        "cute": transpose_cute,
        "cute_v2": transpose_cute_v2,
    }

    print(f"transpose y = x.t()  shape={M}x{N}  dtype={DTYPE}  peak={PEAK_BW_TBPS} TB/s\n")
    print(f"{'kernel':28s} {'time_ms':>10s} {'GB/s':>10s} {'pct_peak_bw':>12s}")
    print("-" * 64)
    for name, fn in kernels.items():
        ms = bench(fn, x)
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        pct = gbps / (PEAK_BW_TBPS * 1000) * 100
        print(f"{name:28s} {ms:10.4f} {gbps:10.1f} {pct:11.1f}%")


if __name__ == "__main__":
    main()
