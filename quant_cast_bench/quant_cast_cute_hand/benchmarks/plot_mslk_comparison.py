"""Render CuTe-hand versus MSLK benchmark results."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OURS_COLOR = "#1f77b4"
MSLK_COLOR = "#d62728"
CHART_HEIGHT = 3.3 * 1.25 * 1.5
TITLE_FONT_SIZE = 12
LABEL_FONT_SIZE = 13
TICK_FONT_SIZE = 11
LEGEND_FONT_SIZE = 10


def _plot_chart(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    input_dtypes = {row.get("dtype", "bfloat16") for row in rows}
    if input_dtypes != {"bfloat16"}:
        raise ValueError(
            f"README charts require BF16 inputs, got {sorted(input_dtypes)}"
        )

    rows.sort(key=lambda row: int(row["M"]))
    x = list(range(len(rows)))
    axis.plot(
        x,
        [float(row["ours_tb_s"]) for row in rows],
        color=OURS_COLOR,
        linewidth=2,
        marker="o",
        markersize=4,
        label="CuTe hand TMA",
    )
    axis.plot(
        x,
        [float(row["mslk_tb_s"]) for row in rows],
        color=MSLK_COLOR,
        linewidth=2,
        marker="o",
        markersize=4,
        label="MSLK Triton",
    )

    axis.set_title(
        f"{rows[0]['kernel']}\nInput dtype: BF16", fontsize=TITLE_FONT_SIZE
    )
    axis.set_xticks(x, [row["M"] for row in rows])
    axis.set_xlabel("M == K", fontsize=LABEL_FONT_SIZE)
    axis.set_ylabel("TB/s", fontsize=LABEL_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    axis.set_ylim(0, 8)
    axis.grid(axis="y", alpha=0.3)
    legend = axis.legend(
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
        ncol=2,
        frameon=False,
        handlelength=3.5,
        numpoints=2,
    )
    for line in legend.get_lines():
        line.set_marker("")


def plot(csv_path: Path, output_path: Path) -> None:
    with csv_path.open(newline="") as csv_file:
        rows = [
            row
            for row in csv.DictReader(csv_file)
            if int(row["M"]) == int(row["K"])
        ]
    if not rows:
        raise ValueError(f"{csv_path} contains no square-shape benchmark rows")

    grouped_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped_rows.setdefault(row["kernel"], []).append(row)
    family_order = {"mxfp4": 0, "nvfp4": 1}
    charts = sorted(
        grouped_rows.values(),
        key=lambda values: family_order.get(values[0]["family"], 2),
    )

    column_count = 2
    row_count = math.ceil(len(charts) / column_count)
    figure, axes_array = plt.subplots(
        row_count,
        column_count,
        figsize=(18, CHART_HEIGHT * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    axes = list(axes_array.flat)
    for axis, chart_rows in zip(axes, charts):
        _plot_chart(axis, chart_rows)
    for axis in axes[len(charts) :]:
        axis.set_visible(False)

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
