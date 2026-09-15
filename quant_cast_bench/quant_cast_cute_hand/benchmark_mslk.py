"""Compare the CuTe-hand TMA NVFP4 kernel with MSLK's dense NVFP4 kernel.

Both implementations receive the same precomputed global scale. Reported bandwidths use the
same logical BF16 input + packed FP4 qdata + padded Blackwell-scale byte count.

    python -m quant_cast_bench.quant_cast_cute_hand.benchmark_mslk
    python -m quant_cast_bench.quant_cast_cute_hand.benchmark_mslk \
        --M 2048,4096 --K 8192,16384 --mk_mode pair
    python -m quant_cast_bench.quant_cast_cute_hand.benchmark_mslk \
        --shapes_for_model gpt-oss-120b
"""

import csv
import os
from pathlib import Path
import sys

# Set this before importing PyTorch so Kineto does not print USDT messages.
os.environ.setdefault("KINETO_LOG_LEVEL", "6")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fire
from mslk.quantize.triton.fp4_quantize import triton_quantize_nvfp4
from mslk.quantize.triton.fp4_utils import global_scale_nvfp4
import torch
from torch._inductor.utils import _do_bench_using_profiling

from quant_cast_bench.quant_cast_cute_hand.recipes import nvfp4_swizzle_tma
from quant_cast_bench.quant_cast_cute_hand.shape_utils import (
    gpt_oss_120b_m8192_tp8_ep8,
)


SHAPES = (2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576)

_KERNELS = ("nvfp4_swizzle_tma",)

_MODEL_SHAPES = {
    "gpt-oss-120b": gpt_oss_120b_m8192_tp8_ep8,
}

CSV_FIELDS = (
    "kernel",
    "family",
    "mode",
    "M",
    "K",
    "ours_ms",
    "mslk_ms",
    "ours_tb_s",
    "mslk_tb_s",
    "speedup_vs_mslk",
)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _logical_bytes(M: int, K: int) -> int:
    numel = M * K
    qdata_bytes = numel // 2
    scale_bytes = _ceil_div(M, 128) * _ceil_div(_ceil_div(K, 16), 4) * 32 * 16
    return 2 * numel + qdata_bytes + scale_bytes


def _time(run) -> float:
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    return _do_bench_using_profiling(
        run, warmup=2, rep=5, is_vetted_benchmarking=True
    )


def _parse_sizes(value: str, name: str) -> list[int]:
    """Parse one integer or a comma-separated list of integers."""
    items = str(value).split(",")
    if any(not item.strip() for item in items):
        raise ValueError(
            f"{name} must be an integer or comma-separated integers, got {value!r}"
        )
    try:
        sizes = [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer or comma-separated integers, got {value!r}"
        ) from exc
    if any(size <= 0 for size in sizes):
        raise ValueError(f"{name} values must be positive, got {value!r}")
    return sizes


def _parse_kernels(value: str) -> list[str]:
    kernels = [item.strip() for item in str(value).split(",")]
    if any(not kernel for kernel in kernels):
        raise ValueError(f"kernel must be a name or comma-separated names, got {value!r}")
    invalid = [kernel for kernel in kernels if kernel not in _KERNELS]
    if invalid:
        raise ValueError(f"unknown kernels {invalid}; have {list(_KERNELS)}")
    return list(dict.fromkeys(kernels))


@fire.decorators.SetParseFns(
    kernel=str,
    M=str,
    K=str,
    mk_mode=str,
    csv_output=str,
    shapes_for_model=str,
)
def main(
    kernel: str = "nvfp4_swizzle_tma",
    M: str | None = None,
    K: str | None = None,
    mk_mode: str | None = None,
    csv_output: str = "",
    shapes_for_model: str = "",
) -> None:
    """Compare CuTe-hand and MSLK NVFP4 kernels over an M-by-K shape grid."""
    kernels = _parse_kernels(kernel)

    shapes_for_model = shapes_for_model.strip().lower()
    if shapes_for_model:
        specified_shape_args = [
            name
            for name, value in (("M", M), ("K", K), ("mk_mode", mk_mode))
            if value is not None
        ]
        if specified_shape_args:
            raise ValueError(
                "shapes_for_model cannot be combined with "
                + ", ".join(specified_shape_args)
            )
        if shapes_for_model not in _MODEL_SHAPES:
            raise ValueError(
                f"unsupported shapes_for_model {shapes_for_model!r}; "
                f"choose from {tuple(_MODEL_SHAPES)}"
            )
        shapes = _MODEL_SHAPES[shapes_for_model]()
    else:
        default_sizes = ",".join(str(size) for size in SHAPES)
        m_values = _parse_sizes(default_sizes if M is None else M, "M")
        k_values = _parse_sizes(default_sizes if K is None else K, "K")
        mk_mode = "pair" if mk_mode is None else mk_mode.strip().lower()
        if mk_mode not in ("cartesian", "pair"):
            raise ValueError(
                f"unsupported mk_mode {mk_mode!r}; choose from ('cartesian', 'pair')"
            )
        if mk_mode == "pair":
            if len(m_values) != len(k_values):
                raise ValueError(
                    "pair mk_mode requires the same number of M and K values, got "
                    f"{len(m_values)} and {len(k_values)}"
                )
            shapes = list(zip(m_values, k_values))
        else:
            shapes = [(m, k) for m in m_values for k in k_values]

    # Prime Kineto before collecting the first real data point.
    profiler_scratch = torch.empty(1, dtype=torch.int32, device="cuda")
    _do_bench_using_profiling(
        profiler_scratch.zero_, warmup=2, rep=2, is_vetted_benchmarking=True
    )

    csv_file = None
    csv_writer = None
    if csv_output:
        csv_path = Path(csv_output).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        csv_writer.writeheader()

    try:
        for name in kernels:
            for M, K in shapes:
                torch.manual_seed(0)
                x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
                # Match MSLK's normal API usage, but keep this global reduction outside both
                # timed regions because both quantization kernels take it as a precomputed input.
                global_scale = global_scale_nvfp4(x)

                mslk_run = lambda: triton_quantize_nvfp4(x, global_scale)
                ours_run = lambda: nvfp4_swizzle_tma(x, global_scale)

                ours_qdata, ours_scale = ours_run()
                mslk_qdata, mslk_scale = mslk_run()
                torch.cuda.synchronize()
                assert torch.equal(
                    ours_qdata.view(torch.uint8), mslk_qdata.view(torch.uint8)
                ), "qdata mismatch between CuTe-hand and MSLK"
                assert torch.equal(
                    ours_scale.view(torch.uint8).flatten(),
                    mslk_scale.view(torch.uint8).flatten(),
                ), "scale mismatch between CuTe-hand and MSLK"
                del ours_qdata, ours_scale, mslk_qdata, mslk_scale

                mslk_ms = _time(mslk_run)
                ours_ms = _time(ours_run)
                byte_count = _logical_bytes(M, K)
                ours_tb_s = byte_count / (ours_ms * 1e-3) / 1e12
                mslk_tb_s = byte_count / (mslk_ms * 1e-3) / 1e12
                speedup = mslk_ms / ours_ms

                print(
                    f"{name:48s} {M:5d}x{K:<5d} "
                    f"ours={ours_ms:.4f} ms/{ours_tb_s:.3f} TB/s "
                    f"MSLK={mslk_ms:.4f} ms/{mslk_tb_s:.3f} TB/s "
                    f"speedup={speedup:.3f}x",
                    flush=True,
                )

                if csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "kernel": name,
                            "family": "nvfp4",
                            "mode": "dim_k",
                            "M": M,
                            "K": K,
                            "ours_ms": f"{ours_ms:.6f}",
                            "mslk_ms": f"{mslk_ms:.6f}",
                            "ours_tb_s": f"{ours_tb_s:.6f}",
                            "mslk_tb_s": f"{mslk_tb_s:.6f}",
                            "speedup_vs_mslk": f"{speedup:.6f}",
                        }
                    )
                    csv_file.flush()
                del x
    finally:
        if csv_file is not None:
            csv_file.close()

    if csv_output:
        print(f"Wrote {Path(csv_output).expanduser()}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
