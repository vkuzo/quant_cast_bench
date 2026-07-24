"""Regenerate the memory-bandwidth-by-mode scatter chart embedded in benchmarks/README.md.

Two-step workflow:

  1. Produce the data by running the benchmark once per mode, appending to one CSV:
       rm -f benchmarks/bench_results.csv
       python benchmarks/benchmark.py --mode compile --csv benchmarks/bench_results.csv
       python benchmarks/benchmark.py --mode triton  --csv benchmarks/bench_results.csv
       python benchmarks/benchmark.py --mode cute    --csv benchmarks/bench_results.csv
  2. Render the chart from that CSV (no GPU needed):
       python benchmarks/plot_bench.py

The CSV is long/tidy (kernel,mode,gpu_time_ms,gbps,mem_bw_pct); this script pivots it to a scatter
plot with one marker series per mode (points not connected) -- bandwidth on the x-axis, one kernel
per row on the y-axis. Kernels benchmarked in only some modes (e.g. the compile-only bf16_rht /
fp32_to_bf16_sr) simply show fewer points.
"""

import csv
import os

import fire
import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

_MODES = ["compile", "triton", "cute"]       # default series order (also the legend order)
_COLORS = {
    "compile": "#4C72B0",
    "triton": "#DD8452",
    "cute": "#55A868",
    "flex_tile_map_triton": "#C44E52",
}
_MARKERS = {"compile": "o", "triton": "s", "cute": "^", "flex_tile_map_triton": "D"}  # icon per series
# legend labels spell out what each mode is
_LABELS = {
    "compile": "compile: torch.compile + inductor",
    "triton": "triton: triton vibed with opus 4.8",
    "cute": "cute: cuteDSL vibed with opus 4.8",
    "flex_tile_map_triton": "flex_tile_map_triton: plain-pytorch f on the hop/ template",
}
_BASELINE = "relu (baseline)"
# row groupings (by position, top-to-bottom): (label, number of rows). The last group takes all
# remaining rows. Rendered as full-width dashed separators with a label at the top of each band.
_GROUP_SIZES = [
    ("8-bit elementwise", 1),
    ("8-bit dim-k", 2),
    ("8-bit dim-m", 3),
    ("8-bit dim-km", 3),
    ("8-bit square", 2),
    ("row|col-wise", 2),
    ("other", None),  # None = all remaining rows
]
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


_BW_LABEL = "achieved memory bandwidth (% of B200 peak, 8 TB/s)"
_TITLE = "quant_cast memory bandwidth by implementation (16384×16384, B200)"


def _plot(data, kernels, out_path, modes=_MODES, group_sizes=_GROUP_SIZES, title=_TITLE,
          fig_height=6.0):
    """Scatter the pivoted data to `out_path`: bandwidth on the x-axis, one kernel per row on the
    y-axis (first kernel at the top). `modes` selects (and orders) the series; `group_sizes` draws
    the row-band separators (pass None to skip them, e.g. for a short single-topic chart);
    `fig_height` sizes the figure in inches (shrink it for few-row charts so rows aren't sparse)."""
    fig, ax = plt.subplots(figsize=(9, fig_height))
    # one scatter series per mode, distinct marker+color, points NOT connected
    for mode in modes:
        idx = [i for i, k in enumerate(kernels) if mode in data[k]]
        pct = [data[k][mode] for k in kernels if mode in data[k]]
        ax.scatter(pct, idx, marker=_MARKERS[mode], color=_COLORS[mode], label=_LABELS[mode],
                   s=70, zorder=3)

    ax.set_xlabel(_BW_LABEL)
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels(kernels)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))  # tick labels read "20%", "40%", ...
    ax.invert_yaxis()  # first kernel at the top
    ax.grid(axis="x", ls=":", alpha=0.4)
    # reference marker for the "good enough" bandwidth target; label overlaid vertically on the line
    # (kept out of the legend) via a blended transform: x in data coords, y in axes fraction.
    ax.axvline(80, color="purple", ls="--", lw=1.5, zorder=2)
    ax.text(80, 0.99, "80% speed of light", transform=ax.get_xaxis_transform(), rotation=90,
            va="top", ha="right", color="purple", fontsize=9, zorder=4,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0))

    # row groupings: a full-width dashed separator between bands + a label at the top of each band
    # (y-axis is inverted, so the smallest row index in a band is its visual top).
    if group_sizes is not None:
        start = 0
        for label, size in group_sizes:
            ax.text(0.7, start - 0.44, label, ha="left", va="top", fontsize=9, style="italic",
                    color="#444444", zorder=4,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0))
            if size is None:
                break  # final "other" group runs to the last row -- no separator after it
            start += size
            ax.axhline(start - 0.5, color="gray", ls="--", lw=1.0, alpha=0.7, zorder=1)

    # pad the title up to leave room for the horizontal legend sitting just above the axes
    ax.set_title(title, pad=34)
    # legend outside the data area (above the axes), series laid out horizontally
    ax.legend(title="mode", ncol=len(modes), loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")  # bbox_inches captures the outside legend


def main(
    csv="benchmarks/bench_results.csv",
    out="benchmarks/mem_bw.png",
    modes=None,
    kernel_filter=None,
    groups=True,
    title=_TITLE,
    fig_height=6.0,
):
    """Render the mem-bandwidth scatter from the CSV.

    Defaults reproduce the benchmarks/README.md chart (all kernels, compile/triton/cute, row bands).
    Override for a focused chart, e.g. the flex_tile_map deepseek view:
        python benchmarks/plot_bench.py \\
            --modes compile,triton,flex_tile_map_triton --kernel_filter deepseek \\
            --groups False --fig_height 3.0 \\
            --out quant_cast_bench/flex_tile_map/deepseek_mem_bw.png \\
            --title "flex_tile_map deepseek casts: memory bandwidth (16384x16384, B200)"
    """
    modes = modes.split(",") if isinstance(modes, str) else (modes or _MODES)
    # resolve paths relative to the repo root so the script works from any cwd
    root = os.path.dirname(_HERE)

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(root, p)

    data, order = _load(_resolve(csv))
    kernels = [k for k in order if k != _BASELINE]  # exclude the relu baseline entirely
    if kernel_filter is not None:
        kernels = [k for k in kernels if kernel_filter in k]

    _plot(data, kernels, _resolve(out), modes=modes,
          group_sizes=_GROUP_SIZES if groups else None, title=title, fig_height=fig_height)
    print(f"wrote {_resolve(out)} ({len(kernels)} kernels, modes={modes})")


if __name__ == "__main__":
    fire.Fire(main)
