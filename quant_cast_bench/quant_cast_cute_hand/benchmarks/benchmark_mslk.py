"""Compare CuTe-hand dim-K FP4 kernels with the corresponding MSLK kernels.

For NVFP4, both implementations receive the same precomputed global scale. Reported bandwidths
use the same logical BF16 input + packed FP4 qdata + padded Blackwell-scale byte count.

The MXFP4 paths use different E8M0 scale-selection conventions: CuTe-hand uses
``ceil_pow2(amax / 6)``, while MSLK uses ``ceil(log2(amax)) - 2``. They perform the same class of
1x32 E2M1/E8M0 cast but are not expected to produce bitwise-identical outputs.

    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark_mslk
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark_mslk \
        --M 2048,4096 --K 8192,16384 --mk_mode pair
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark_mslk \
        --shapes_for_model gpt-oss-120b
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark_mslk \
        --kernel nvfp4_swizzle_tma,mxfp4_swizzle_v2
"""

import csv
import os
from pathlib import Path
import sys

# Set this before importing PyTorch so Kineto does not print USDT messages.
os.environ.setdefault("KINETO_LOG_LEVEL", "6")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import fire
from mslk.quantize.triton.fp4_quantize import (
    triton_quantize_mx4,
    triton_quantize_nvfp4,
)
from mslk.quantize.triton.fp4_utils import global_scale_nvfp4
import torch
from torch._inductor.utils import _do_bench_using_profiling

from quant_cast_bench.quant_cast_cute_hand.recipes import (
    mxfp4_swizzle_v2,
    nvfp4_swizzle_tma,
)
from quant_cast_bench.quant_cast_cute_hand.benchmarks.shape_utils import (
    gpt_oss_120b_m8192_tp8_ep8,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    _compute_error,
    mxfp4_swizzle_dq_f,
)


SHAPES = (2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576)

_KERNELS = ("nvfp4_swizzle_tma", "mxfp4_swizzle_v2")
_MIN_MXFP4_SQNR_DB = 10.0

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
    "ours_sqnr_db",
    "mslk_sqnr_db",
)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _logical_bytes(M: int, K: int, group_size: int) -> int:
    numel = M * K
    qdata_bytes = numel // 2
    scale_bytes = (
        _ceil_div(M, 128)
        * _ceil_div(_ceil_div(K, group_size), 4)
        * 32
        * 16
    )
    return 2 * numel + qdata_bytes + scale_bytes


def _make_runs(name: str, x: torch.Tensor):
    if name == "nvfp4_swizzle_tma":
        # Match MSLK's normal API usage, but keep this global reduction outside both
        # timed regions because both quantization kernels take it as a precomputed input.
        global_scale = global_scale_nvfp4(x)
        return (
            lambda: nvfp4_swizzle_tma(x, global_scale),
            lambda: triton_quantize_nvfp4(x, global_scale),
            "nvfp4",
            16,
        )
    if name == "mxfp4_swizzle_v2":
        return (
            lambda: mxfp4_swizzle_v2(x),
            lambda: triton_quantize_mx4(x),
            "mxfp4",
            32,
        )
    raise AssertionError(f"missing benchmark implementation for {name}")


def _mxfp4_sqnr_db(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
) -> float:
    M, K = x.shape
    blocked_scale = scale.view(torch.uint8).reshape(
        _ceil_div(M, 128),
        _ceil_div(K // 32, 4),
        32,
        16,
    ).view(torch.float8_e8m0fnu)
    dequantized = mxfp4_swizzle_dq_f(
        qdata.view(torch.float4_e2m1fn_x2), blocked_scale
    )
    return _compute_error(x.float(), dequantized.float()).item()


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
    """Compare CuTe-hand and MSLK FP4 kernels over an M-by-K shape grid."""
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
                ours_run, mslk_run, family, group_size = _make_runs(name, x)

                ours_qdata, ours_scale = ours_run()
                mslk_qdata, mslk_scale = mslk_run()
                torch.cuda.synchronize()
                assert ours_qdata.shape == mslk_qdata.shape
                assert ours_scale.numel() == mslk_scale.numel()
                ours_sqnr_db = mslk_sqnr_db = None
                if family == "nvfp4":
                    assert torch.equal(
                        ours_qdata.view(torch.uint8), mslk_qdata.view(torch.uint8)
                    ), "qdata mismatch between CuTe-hand and MSLK"
                    assert torch.equal(
                        ours_scale.view(torch.uint8).flatten(),
                        mslk_scale.view(torch.uint8).flatten(),
                    ), "scale mismatch between CuTe-hand and MSLK"
                else:
                    ours_sqnr_db = _mxfp4_sqnr_db(
                        x, ours_qdata, ours_scale
                    )
                    mslk_sqnr_db = _mxfp4_sqnr_db(
                        x, mslk_qdata, mslk_scale
                    )
                    assert ours_sqnr_db > _MIN_MXFP4_SQNR_DB, (
                        f"CuTe-hand MXFP4 SQNR {ours_sqnr_db:.2f} dB is below "
                        f"{_MIN_MXFP4_SQNR_DB:.1f} dB"
                    )
                    assert mslk_sqnr_db > _MIN_MXFP4_SQNR_DB, (
                        f"MSLK MXFP4 SQNR {mslk_sqnr_db:.2f} dB is below "
                        f"{_MIN_MXFP4_SQNR_DB:.1f} dB"
                    )
                del ours_qdata, ours_scale, mslk_qdata, mslk_scale

                mslk_ms = _time(mslk_run)
                ours_ms = _time(ours_run)
                byte_count = _logical_bytes(M, K, group_size)
                ours_tb_s = byte_count / (ours_ms * 1e-3) / 1e12
                mslk_tb_s = byte_count / (mslk_ms * 1e-3) / 1e12
                speedup = mslk_ms / ours_ms

                sqnr_summary = (
                    f" SQNR=ours:{ours_sqnr_db:.2f}/MSLK:{mslk_sqnr_db:.2f} dB"
                    if ours_sqnr_db is not None
                    else ""
                )
                print(
                    f"{name:48s} {M:5d}x{K:<5d} "
                    f"ours={ours_ms:.4f} ms/{ours_tb_s:.3f} TB/s "
                    f"MSLK={mslk_ms:.4f} ms/{mslk_tb_s:.3f} TB/s "
                    f"speedup={speedup:.3f}x{sqnr_summary}",
                    flush=True,
                )

                if csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "kernel": name,
                            "family": family,
                            "mode": "dim_k",
                            "M": M,
                            "K": K,
                            "ours_ms": f"{ours_ms:.6f}",
                            "mslk_ms": f"{mslk_ms:.6f}",
                            "ours_tb_s": f"{ours_tb_s:.6f}",
                            "mslk_tb_s": f"{mslk_tb_s:.6f}",
                            "speedup_vs_mslk": f"{speedup:.6f}",
                            "ours_sqnr_db": (
                                f"{ours_sqnr_db:.6f}"
                                if ours_sqnr_db is not None
                                else ""
                            ),
                            "mslk_sqnr_db": (
                                f"{mslk_sqnr_db:.6f}"
                                if mslk_sqnr_db is not None
                                else ""
                            ),
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
