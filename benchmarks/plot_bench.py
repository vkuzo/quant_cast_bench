"""Regenerate the memory-bandwidth-by-mode bar chart embedded in benchmarks/README.md.

Two-step workflow:

  1. Produce the data by running the benchmark once per mode, appending to one CSV:
       rm -f benchmarks/bench_results.csv
       python benchmarks/benchmark.py --mode compile --csv benchmarks/bench_results.csv
       python benchmarks/benchmark.py --mode triton  --csv benchmarks/bench_results.csv
       python benchmarks/benchmark.py --mode cute    --csv benchmarks/bench_results.csv
  2. Render the chart from that CSV (no GPU needed):
       python benchmarks/plot_bench.py

The CSV is long/tidy (kernel,mode,gpu_time_ms,gbps,mem_bw_pct); this script pivots it to a scatter
plot with one marker series per mode (points not connected). Kernels benchmarked in only some modes
(e.g. the compile-only bf16_rht / fp32_to_bf16_sr) simply show fewer points.
"""

import csv
import os

import fire
import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402

_MODES = ["compile", "triton", "cute"]       # fixed series order (also the legend order)
_COLORS = {"compile": "#4C72B0", "triton": "#DD8452", "cute": "#55A868"}
_MARKERS = {"compile": "o", "triton": "s", "cute": "^"}  # distinct icon per series
# legend labels spell out what each mode is
_LABELS = {
    "compile": "compile: torch.compile + inductor",
    "triton": "triton: triton vibed with opus 4.8",
    "cute": "cute: cuteDSL vibed with opus 4.8",
}
_BASELINE = "relu (baseline)"
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(csv_path):
    """Read the long CSV -> ({kernel: {mode: pct}}, kernel_order). Dedup keeps the last row for a
    (kernel, mode) pair so re-appended sweeps are safe; kernel order = first appearance."""
    data, order = {}, []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            kernel, mode = row["kernel"], row["mode"]
            if kernel not in data:
                data[kernel] = {}
                order.append(kernel)
            data[kernel][mode] = float(row["mem_bw_pct"])
    return data, order


def main(csv="benchmarks/bench_results.csv", out="benchmarks/mem_bw.png"):
    # resolve paths relative to the repo root so the script works from any cwd
    root = os.path.dirname(_HERE)
    csv_path = csv if os.path.isabs(csv) else os.path.join(root, csv)
    out_path = out if os.path.isabs(out) else os.path.join(root, out)

    data, order = _load(csv_path)
    kernels = [k for k in order if k != _BASELINE]  # exclude the relu baseline entirely
    x = range(len(kernels))

    fig, ax = plt.subplots(figsize=(14, 6))
    # one scatter series per mode, distinct marker+color, points NOT connected
    for mode in _MODES:
        xs = [i for i, k in enumerate(kernels) if mode in data[k]]
        ys = [data[k][mode] for k in kernels if mode in data[k]]
        ax.scatter(xs, ys, marker=_MARKERS[mode], color=_COLORS[mode], label=_LABELS[mode],
                   s=70, zorder=3)

    ax.set_ylabel("achieved memory bandwidth (% of B200 peak, 8 TB/s)")
    ax.set_title("quant_cast memory bandwidth by implementation (16384×16384, B200)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(kernels, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(title="mode")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path} ({len(kernels)} kernels, modes={_MODES})")


if __name__ == "__main__":
    fire.Fire(main)
