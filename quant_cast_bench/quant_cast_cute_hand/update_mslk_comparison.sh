#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
csv_path="${script_dir}/mslk_comparison.csv"
png_path="${script_dir}/mslk_comparison.png"

cd "${repo_root}"
PYTHONUNBUFFERED=1 python "${script_dir}/benchmark_mslk.py" \
  --M 2048,3072,4096,6144,8192,12288,16384,24576 \
  --K 2048,3072,4096,6144,8192,12288,16384,24576 \
  --mk_mode pair \
  --csv_output "${csv_path}"
python "${script_dir}/plot_mslk_comparison.py" \
  --csv "${csv_path}" \
  --output "${png_path}"
