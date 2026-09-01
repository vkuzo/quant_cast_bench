"""Memory-bandwidth benchmark for the handwritten CuTeDSL playground kernels.

Each kernel here is a memory-bound elementwise cast/op, so the signal we care about is achieved
memory bandwidth vs. the H100 ceiling (3.35 TB/s HBM3, SXM5). We build a bf16 (M, K) input, run
the selected kernel, time it with `do_bench_using_profiling`, and report GPU time + GB/s + % of peak.

    python -m quant_cast_bench.quant_cast_cute_hand.benchmark --kernel add_v0
    python -m quant_cast_bench.quant_cast_cute_hand.benchmark --kernel add_v0 --M 8192 --K 8192
"""

import os
import sys

import fire
import tabulate
import torch
from torch._inductor.utils import do_bench_using_profiling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quant_cast_bench.quant_cast_cute_hand.recipes import add_v0

# H100 SXM5 HBM3 peak: 3.35 TB/s. (PCIe H100 is ~2 TB/s -- adjust if benching on that part.)
H100_PEAK_BW_GBPS = 3350.0


def _bench_add_v0(M, K):
    # read input (M*K bf16) + write output (M*K bf16); a trivially memory-bound + 1 elementwise op.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float32, device="cuda")

    def run():
        return add_v0(x, 1.0)

    out = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor (e.g. a grid that
    # covers only one block): without this the timing is just launch overhead and the reported
    # bandwidth is fictional. Require the result to actually equal x + 1 across all elements.
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), x.float() + 1.0)
    bytes_per_iter = x.numel() * x.element_size() + out.numel() * out.element_size()
    return run, bytes_per_iter


# name -> builder returning (run_fn, bytes_per_iter). Add new playground kernels here.
_KERNELS = {
    "add_v0": _bench_add_v0,
}


def main(
    kernel: str = "add_v0",
    M: int = 16384,
    K: int = 16384,
):
    """Benchmark one handwritten CuTeDSL kernel and print GPU time + achieved memory bandwidth."""
    device_name = torch.cuda.get_device_name(0)
    assert "H100" in device_name, f"this benchmark assumes H100, got {device_name!r}"

    if kernel not in _KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; have {sorted(_KERNELS)}")

    run, bytes_per_iter = _KERNELS[kernel](M, K)

    # warm up so first-call costs (compile, autotune, allocator) don't leak into the timing.
    for _ in range(2):
        run()
    torch.cuda.synchronize()

    gpu_time_ms = do_bench_using_profiling(run)
    gbps = bytes_per_iter / (gpu_time_ms * 1e-3) / 1e9
    pct_peak = gbps / H100_PEAK_BW_GBPS * 100

    print(f"kernel: {kernel}  shape: ({M}, {K})  dtype: bfloat16")
    print(
        tabulate.tabulate(
            [[kernel, f"{gpu_time_ms:.4f}", f"{gbps:.1f}", f"{pct_peak:.1f}%"]],
            headers=["kernel", "gpu_time_ms", "gbps", "pct_peak"],
            colalign=("left", "right", "right", "right"),
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
