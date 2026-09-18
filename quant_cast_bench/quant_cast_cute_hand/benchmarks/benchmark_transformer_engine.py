"""Compare CuTe-hand quantization kernels with TransformerEngine kernels.

The primary TransformerEngine NVFP4 measurements use precomputed global amaxes,
matching the CuTe-hand API contract where outer scales are inputs. All reported
bandwidths use the same logical input + qdata + padded-scale byte count. Kernels
without a comparable TransformerEngine implementation still benchmark our path.
"""

import csv
import os
from pathlib import Path
import sys

# Set these before importing PyTorch or TransformerEngine.
os.environ.setdefault("KINETO_LOG_LEVEL", "6")
os.environ.setdefault("NVTE_USE_FAST_MATH", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.func._random as prng
import fire
import transformer_engine.pytorch as te  # noqa: F401 - must precede transformer_engine_torch
import transformer_engine_torch as tex
from torch._inductor.utils import _do_bench_using_profiling
from transformer_engine.pytorch import MXFP8Quantizer, NVFP4Quantizer

from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma.blockscaled_tma_impl import (
    mxfp8_32x32_swizzle_v2,
    mxfp8_swizzle_v2,
)
from quant_cast_bench.quant_cast_cute_hand.recipes import (
    mxfp8_swizzle,
    mxfp8_swizzle_v3,
    mxfp8_swizzle_v4,
    mxfp8_swizzle_v5,
    nvfp4_dim_km_swizzle_tma,
    nvfp4_dim_m_rht_swizzle_pipelined,
    nvfp4_dim_m_swizzle_rht_sr_pipelined,
    nvfp4_dim_m_swizzle_tma,
    nvfp4_swizzle_dim_k_dim_m_rht_pipelined,
    nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined,
    nvfp4_swizzle_direct,
    nvfp4_swizzle_tma,
)
from quant_cast_bench.quant_cast_cute_hand.benchmarks.shape_utils import (
    gpt_oss_120b_m8192_tp8_ep8,
)


SHAPES = (2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576)

_MODEL_SHAPES = {
    "gpt-oss-120b": gpt_oss_120b_m8192_tp8_ep8,
}


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _scale_bytes(M: int, K: int, group: int, dim_m: bool) -> int:
    rows, groups = (K, _ceil_div(M, group)) if dim_m else (M, _ceil_div(K, group))
    return _ceil_div(rows, 128) * _ceil_div(groups, 4) * 32 * 16


def _logical_bytes(
    M: int, K: int, family: str, mode: str, input_element_size: int
) -> int:
    numel = M * K
    qbytes = numel if family == "mxfp8" else numel // 2
    group = 32 if family == "mxfp8" else 16
    total = input_element_size * numel
    if mode in ("dim_k", "dim_km"):
        total += qbytes + _scale_bytes(M, K, group, dim_m=False)
    if mode in ("dim_m", "dim_km"):
        total += qbytes + _scale_bytes(M, K, group, dim_m=True)
    return total


def _time(run) -> float:
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    # A short profiling window avoids Kineto overhead distorting very small TE kernels. The
    # process-wide profiler warm-up in main handles the unusually slow first profiler invocation.
    return _do_bench_using_profiling(
        run, warmup=2, rep=5, is_vetted_benchmarking=True
    )


def _make_ours(name: str, x: torch.Tensor):
    outer = torch.ones(1, dtype=torch.float32, device=x.device)
    sign = torch.tensor([1, -1] * 8, dtype=torch.bfloat16, device=x.device)
    key = prng.key(0, device=x.device)

    if name == "mxfp8_swizzle":
        return lambda: mxfp8_swizzle(x)
    if name == "mxfp8_swizzle_v2":
        return lambda: mxfp8_swizzle_v2(x)
    if name == "mxfp8_32x32_swizzle_v2":
        return lambda: mxfp8_32x32_swizzle_v2(x)
    if name == "mxfp8_swizzle_v3":
        return lambda: mxfp8_swizzle_v3(x)
    if name == "mxfp8_swizzle_v4":
        return lambda: mxfp8_swizzle_v4(x)
    if name == "mxfp8_swizzle_v5":
        return lambda: mxfp8_swizzle_v5(x)
    if name == "mxfp8_dim_m_swizzle_v2":
        return lambda: mxfp8_swizzle_v2(x, quant_orientation="dim_m")
    if name == "mxfp8_dim_km_swizzle_v2":
        return lambda: mxfp8_swizzle_v2(x, quant_orientation="dim_km")
    if name == "mxfp8_swizzle_sr_v2":
        return lambda: mxfp8_swizzle_v2(
            x, key=key, rounding_mode="stochastic"
        )
    if name == "mxfp8_dim_m_swizzle_sr_v2":
        return lambda: mxfp8_swizzle_v2(
            x, quant_orientation="dim_m", key=key, rounding_mode="stochastic"
        )
    if name == "mxfp8_dim_km_swizzle_sr_v2":
        return lambda: mxfp8_swizzle_v2(
            x, quant_orientation="dim_km", key=key, rounding_mode="stochastic"
        )
    if name == "nvfp4_swizzle_direct":
        return lambda: nvfp4_swizzle_direct(x, outer)
    if name == "nvfp4_swizzle_tma":
        return lambda: nvfp4_swizzle_tma(x, outer)
    if name == "nvfp4_dim_m_swizzle_tma":
        return lambda: nvfp4_dim_m_swizzle_tma(x, outer)
    if name == "nvfp4_dim_km_swizzle_tma":
        return lambda: nvfp4_dim_km_swizzle_tma(x, outer, outer)
    if name == "nvfp4_dim_m_rht_swizzle_pipelined":
        return lambda: nvfp4_dim_m_rht_swizzle_pipelined(x, outer, sign)
    if name == "nvfp4_dim_m_swizzle_rht_sr_pipelined":
        return lambda: nvfp4_dim_m_swizzle_rht_sr_pipelined(
            x, outer, sign, key
        )
    if name == "nvfp4_swizzle_dim_k_dim_m_rht_pipelined":
        return lambda: nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
            x, outer, outer, sign
        )
    if name == "nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined":
        return lambda: nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined(
            x, outer, outer, sign, key
        )
    raise KeyError(name)


def _make_te(
    family: str,
    mode: str,
    rht: bool,
    stochastic: bool,
    x: torch.Tensor,
    *,
    square_scaling: bool = False,
):
    rowwise = mode in ("dim_k", "dim_km")
    columnwise = mode in ("dim_m", "dim_km")
    if family == "mxfp8":
        quantizer = MXFP8Quantizer(
            tex.DType.kFloat8E4M3,
            rowwise=rowwise,
            columnwise=columnwise,
            with_2d_quantization=square_scaling,
        )
        quantizer.optimize_for_gemm = True
        output = quantizer.make_empty(x.shape, dtype=x.dtype, device=x.device)
        return lambda: quantizer.update_quantized(x, output)

    quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=rowwise,
        columnwise=columnwise,
        with_amax_reduction=False,
        with_rht=rht,
        with_post_rht_amax=rht,
        stochastic_rounding=stochastic,
        with_random_sign_mask=True,
    )
    quantizer.optimize_for_gemm = True
    row_amax = torch.ones(1, dtype=torch.float32, device=x.device)
    col_amax = torch.ones(1, dtype=torch.float32, device=x.device)
    return lambda: tex.nvfp4_quantize_with_amax(
        x, quantizer, row_amax, col_amax
    )


# name, family, mode, RHT, stochastic rounding
CASES = (
    ("mxfp8_swizzle_v2", "mxfp8", "dim_k", False, False),
    ("mxfp8_32x32_swizzle_v2", "mxfp8", "dim_k", False, False),
    ("mxfp8_dim_m_swizzle_v2", "mxfp8", "dim_m", False, False),
    ("mxfp8_dim_km_swizzle_v2", "mxfp8", "dim_km", False, False),
    ("nvfp4_swizzle_tma", "nvfp4", "dim_k", False, False),
    ("nvfp4_dim_m_swizzle_tma", "nvfp4", "dim_m", False, False),
    ("nvfp4_dim_km_swizzle_tma", "nvfp4", "dim_km", False, False),
    ("nvfp4_dim_m_rht_swizzle_pipelined", "nvfp4", "dim_m", True, False),
    (
        "nvfp4_dim_m_swizzle_rht_sr_pipelined",
        "nvfp4",
        "dim_m",
        True,
        True,
    ),
    (
        "nvfp4_swizzle_dim_k_dim_m_rht_pipelined",
        "nvfp4",
        "dim_km",
        True,
        False,
    ),
    (
        "nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined",
        "nvfp4",
        "dim_km",
        True,
        True,
    ),
)

# These are useful CuTe-hand measurements, but TransformerEngine has no matching
# stochastic MXFP8 API to put on the other side of the comparison.
NO_TE_CASES = (
    ("mxfp8_swizzle_sr_v2", "mxfp8", "dim_k", False, True),
    ("mxfp8_dim_m_swizzle_sr_v2", "mxfp8", "dim_m", False, True),
    ("mxfp8_dim_km_swizzle_sr_v2", "mxfp8", "dim_km", False, True),
)

ALL_CASES = CASES + NO_TE_CASES

KERNELS_SUPPORTING_FLOAT16_AND_FLOAT32 = frozenset({
    "mxfp8_swizzle_v2",
    "mxfp8_swizzle_sr_v2",
    "mxfp8_32x32_swizzle_v2",
    "mxfp8_dim_m_swizzle_v2",
    "mxfp8_dim_m_swizzle_sr_v2",
    "mxfp8_dim_km_swizzle_v2",
    "mxfp8_dim_km_swizzle_sr_v2",
    "nvfp4_swizzle_tma",
    "nvfp4_dim_m_swizzle_tma",
    "nvfp4_dim_km_swizzle_tma",
})

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

CSV_FIELDS = (
    "kernel",
    "dtype",
    "family",
    "mode",
    "rht",
    "stochastic",
    "M",
    "K",
    "ours_ms",
    "te_ms",
    "ours_tb_s",
    "te_tb_s",
    "speedup_vs_te",
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
    available = {case[0] for case in ALL_CASES}
    invalid = [kernel for kernel in kernels if kernel not in available]
    if invalid:
        raise ValueError(f"unknown kernels {invalid}; have {sorted(available)}")
    return list(dict.fromkeys(kernels))


def _parse_dtype(value: str) -> tuple[str, torch.dtype]:
    name = str(value).strip().lower()
    if name not in DTYPES:
        raise ValueError(f"unsupported dtype {value!r}; choose from {tuple(DTYPES)}")
    return name, DTYPES[name]


@fire.decorators.SetParseFns(
    kernel=str,
    M=str,
    K=str,
    mk_mode=str,
    csv_output=str,
    shapes_for_model=str,
    dtype=str,
)
def main(
    kernel: str = ",".join(case[0] for case in ALL_CASES),
    M: str | None = None,
    K: str | None = None,
    mk_mode: str | None = None,
    csv_output: str = "",
    shapes_for_model: str = "",
    dtype: str = "bfloat16",
) -> None:
    """Compare CuTe-hand and TransformerEngine kernels over an M-by-K shape grid.

    Set ``csv_output`` to additionally save machine-readable results. Empty TE
    fields mean that TransformerEngine has no comparable implementation.
    """
    kernels = _parse_kernels(kernel)
    cases_by_name = {case[0]: case for case in ALL_CASES}
    cases = [cases_by_name[name] for name in kernels]
    dtype_name, input_dtype = _parse_dtype(dtype)
    if input_dtype != torch.bfloat16:
        unsupported = [
            name
            for name in kernels
            if name not in KERNELS_SUPPORTING_FLOAT16_AND_FLOAT32
        ]
        if unsupported:
            raise ValueError(
                f"{dtype_name} input is unsupported by kernels: "
                + ", ".join(unsupported)
            )

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

    te_case_names = {case[0] for case in CASES}
    te_cache = {}
    try:
        for name, family, mode, rht, stochastic in cases:
            for M, K in shapes:
                torch.manual_seed(0)
                x = torch.randn(M, K, dtype=input_dtype, device="cuda")
                te_ms = None
                te_tb_s = None
                if name in te_case_names:
                    square_scaling = name == "mxfp8_32x32_swizzle_v2"
                    key = (
                        family,
                        mode,
                        rht,
                        stochastic,
                        square_scaling,
                        dtype_name,
                        M,
                        K,
                    )
                    if key not in te_cache:
                        te_run = _make_te(
                            family,
                            mode,
                            rht,
                            stochastic,
                            x,
                            square_scaling=square_scaling,
                        )
                        te_cache[key] = _time(te_run)
                    te_ms = te_cache[key]

                ours_ms = _time(_make_ours(name, x))
                byte_count = _logical_bytes(
                    M, K, family, mode, x.element_size()
                )
                ours_tb_s = byte_count / (ours_ms * 1e-3) / 1e12
                if te_ms is not None:
                    te_tb_s = byte_count / (te_ms * 1e-3) / 1e12
                    speedup = te_ms / ours_ms
                    te_summary = (
                        f"TE={te_ms:.4f} ms/{te_tb_s:.3f} TB/s "
                        f"speedup={speedup:.3f}x"
                    )
                else:
                    speedup = None
                    te_summary = "TE=n/a speedup=n/a"
                print(
                    f"{name:48s} {M:5d}x{K:<5d} dtype={dtype_name:8s} "
                    f"ours={ours_ms:.4f} ms/{ours_tb_s:.3f} TB/s "
                    f"{te_summary}",
                    flush=True,
                )

                if csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "kernel": name,
                            "dtype": dtype_name,
                            "family": family,
                            "mode": mode,
                            "rht": rht,
                            "stochastic": stochastic,
                            "M": M,
                            "K": K,
                            "ours_ms": f"{ours_ms:.6f}",
                            "te_ms": "" if te_ms is None else f"{te_ms:.6f}",
                            "ours_tb_s": f"{ours_tb_s:.6f}",
                            "te_tb_s": "" if te_tb_s is None else f"{te_tb_s:.6f}",
                            "speedup_vs_te": "" if speedup is None else f"{speedup:.6f}",
                        }
                    )
                    csv_file.flush()
                del x
    finally:
        if csv_file is not None:
            csv_file.close()

    unavailable = [name for name, *_ in NO_TE_CASES if name in kernels]
    if unavailable:
        print(
            "No comparable TransformerEngine implementation: "
            + ", ".join(unavailable),
            flush=True,
        )
    if csv_output:
        print(f"Wrote {Path(csv_output).expanduser()}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
