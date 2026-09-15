"""Render CuTe-hand versus TransformerEngine benchmark results."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OURS_COLOR = "#1f77b4"
TE_COLOR = "#d62728"
CHART_HEIGHT = 3.3 * 1.25

DIM_M_RHT_TMA_KERNELS = {
    "nvfp4_dim_m_rht_swizzle_tma",
    "nvfp4_dim_m_swizzle_rht_sr_tma",
}
DEPRECATED_KERNEL = "nvfp4_swizzle_dim_k_dim_m_rht_tma"


def _is_true(value: str) -> bool:
    return value.lower() == "true"


def _implementation(kernel: str, family: str) -> str:
    if family == "mxfp8":
        implementation = kernel.rsplit("_", 1)[-1]
        if "32x32" in kernel:
            return f"32x32_{implementation}"
        return implementation
    return "pipelined" if kernel.endswith("_pipelined") else "tma"


def _read_results(
    path: Path,
) -> dict[tuple[str, str, bool, str], dict[bool, list[dict[str, str]]]]:
    charts: dict[
        tuple[str, str, bool, str], dict[bool, list[dict[str, str]]]
    ] = {}
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if int(row["M"]) != int(row["K"]):
                continue
            key = (
                row["family"],
                row["mode"],
                _is_true(row["rht"]),
                _implementation(row["kernel"], row["family"]),
            )
            stochastic = _is_true(row["stochastic"])
            charts.setdefault(key, {}).setdefault(stochastic, []).append(row)

    if not charts:
        raise ValueError(f"{path} contains no square-shape benchmark rows")
    for variants in charts.values():
        for rows in variants.values():
            rows.sort(key=lambda row: int(row["M"]))
    return charts


def _section_name(variants: dict[bool, list[dict[str, str]]]) -> str:
    kernels = {
        row["kernel"] for rows in variants.values() for row in rows
    }
    if DEPRECATED_KERNEL in kernels:
        return "NVFP4 deprecated"
    family = next(iter(next(iter(variants.values()))))["family"]
    return "MXFP8" if family == "mxfp8" else "NVFP4"


def _plot_chart(
    axis: plt.Axes, variants: dict[bool, list[dict[str, str]]]
) -> None:
    shapes = sorted(
        {int(row["M"]) for rows in variants.values() for row in rows}
    )
    shape_to_x = {shape: index for index, shape in enumerate(shapes)}
    titles = []
    missing_te_rounding = []

    for stochastic in (False, True):
        rows = variants.get(stochastic)
        if not rows:
            continue
        rounding = "SR" if stochastic else "RTNE"
        linestyle = "--" if stochastic else "-"
        titles.append(rows[0]["kernel"])
        x = [shape_to_x[int(row["M"])] for row in rows]
        axis.plot(
            x,
            [float(row["ours_tb_s"]) for row in rows],
            color=OURS_COLOR,
            linestyle=linestyle,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"CuTe hand {rounding}",
        )

        te_rows = [row for row in rows if row["te_tb_s"]]
        if te_rows:
            axis.plot(
                [shape_to_x[int(row["M"])] for row in te_rows],
                [float(row["te_tb_s"]) for row in te_rows],
                color=TE_COLOR,
                linestyle=linestyle,
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"TransformerEngine {rounding}",
            )
        else:
            missing_te_rounding.append(rounding)

    if missing_te_rounding:
        axis.text(
            0.98,
            0.05,
            f"No comparable TE {'/'.join(missing_te_rounding)} implementation",
            color=TE_COLOR,
            fontsize=8,
            ha="right",
            transform=axis.transAxes,
        )

    title_set = set(titles)
    if title_set & DIM_M_RHT_TMA_KERNELS:
        axis.text(
            0.5,
            0.16,
            "we need a _pipelined kernel instead of _tma\nfor this to catch TE, TODO",
            color=TE_COLOR,
            fontsize=8,
            ha="center",
            transform=axis.transAxes,
        )
    if DEPRECATED_KERNEL in title_set:
        axis.text(
            0.5,
            0.16,
            "the _pipelined kernel implements the same thing\nand catches TE",
            color=TE_COLOR,
            fontsize=8,
            ha="center",
            transform=axis.transAxes,
        )

    axis.set_title("\n".join(titles), fontsize=9)
    axis.set_xticks(range(len(shapes)), [str(shape) for shape in shapes])
    axis.set_xlabel("M == K")
    axis.set_ylabel("TB/s")
    axis.set_ylim(0, 8)
    axis.grid(axis="y", alpha=0.3)
    legend = axis.legend(
        loc="upper left",
        fontsize=7,
        ncol=2,
        frameon=False,
        handlelength=3.5,
        numpoints=2,
    )
    for line in legend.get_lines():
        line.set_marker("")


def plot(csv_path: Path, output_path: Path) -> None:
    charts = _read_results(csv_path)
    column_count = 3
    section_order = ("MXFP8", "NVFP4", "NVFP4 deprecated")
    sections = {
        name: [
            variants
            for variants in charts.values()
            if _section_name(variants) == name
        ]
        for name in section_order
    }
    sections = {name: values for name, values in sections.items() if values}

    height_ratios = []
    for variants in sections.values():
        height_ratios.append(0.35)
        height_ratios.extend(
            [CHART_HEIGHT] * math.ceil(len(variants) / column_count)
        )

    figure = plt.figure(
        figsize=(18, sum(height_ratios)), constrained_layout=True
    )
    grid = figure.add_gridspec(
        len(height_ratios), column_count, height_ratios=height_ratios
    )
    grid_row = 0
    for section_name, section_charts in sections.items():
        heading = figure.add_subplot(grid[grid_row, :])
        heading.axis("off")
        heading.text(
            0,
            0.2,
            section_name,
            fontsize=15,
            fontweight="bold",
            transform=heading.transAxes,
        )
        grid_row += 1

        row_count = math.ceil(len(section_charts) / column_count)
        axes = []
        for index in range(row_count * column_count):
            axis = figure.add_subplot(
                grid[grid_row + index // column_count, index % column_count]
            )
            axes.append(axis)
        for axis, variants in zip(axes, section_charts):
            _plot_chart(axis, variants)
        for axis in axes[len(section_charts) :]:
            axis.set_visible(False)
        grid_row += row_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.csv, args.output)


if __name__ == "__main__":
    main()
