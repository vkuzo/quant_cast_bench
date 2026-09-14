"""Render CuTe-hand versus MSLK benchmark results."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OURS_COLOR = "#1f77b4"
MSLK_COLOR = "#d62728"


def plot(csv_path: Path, output_path: Path) -> None:
    with csv_path.open(newline="") as csv_file:
        rows = [
            row
            for row in csv.DictReader(csv_file)
            if int(row["M"]) == int(row["K"])
        ]
    if not rows:
        raise ValueError(f"{csv_path} contains no square-shape benchmark rows")
    rows.sort(key=lambda row: int(row["M"]))

    figure, axis = plt.subplots(figsize=(6, 3.3))
    x = list(range(len(rows)))
    axis.plot(
        x,
        [float(row["ours_tb_s"]) for row in rows],
        color=OURS_COLOR,
        linewidth=2,
        label="CuTe hand TMA",
    )
    axis.plot(
        x,
        [float(row["mslk_tb_s"]) for row in rows],
        color=MSLK_COLOR,
        linewidth=2,
        label="MSLK Triton",
    )

    axis.set_title(rows[0]["kernel"], fontsize=9)
    axis.set_xticks(x, [row["M"] for row in rows])
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
