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


def _is_true(value: str) -> bool:
    return value.lower() == "true"


def _implementation(kernel: str, family: str) -> str:
    if family == "mxfp8":
        return kernel.rsplit("_", 1)[-1]
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


def plot(csv_path: Path, output_path: Path) -> None:
    charts = _read_results(csv_path)
    column_count = 3
    row_count = math.ceil(len(charts) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(18, 3.3 * row_count),
        squeeze=False,
    )

    for axis, variants in zip(axes.flat, charts.values()):
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

        axis.set_title("\n".join(titles), fontsize=9)
        axis.set_xticks(range(len(shapes)), [str(shape) for shape in shapes])
        axis.set_xlabel("M == K")
        axis.set_ylabel("TB/s")
        axis.set_ylim(0, 8)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(
            loc="upper left",
            fontsize=7,
            ncol=2,
            frameon=False,
            handlelength=3.5,
            numpoints=2,
        )

    for axis in axes.flat[len(charts) :]:
        axis.set_visible(False)

    figure.tight_layout()
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
