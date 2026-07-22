import fire
import torch
import torch.profiler
from torch._inductor.utils import do_bench_using_profiling

from .api import _HopMode, flex_cast_quant_dense
from .api_triton_for_debugging import flex_cast_quant_dense_triton
from .recipe_debug_triton import (
    RecipeTriton,
    deepseek_fp8_128_128_triton,
    deepseek_fp8_1_128_dim_m_triton,
)
from .recipes import (
    Recipe,
    deepseek_fp8_1_128,
    deepseek_fp8_1_128_dim_m,
    deepseek_fp8_128_128,
    mxfp8_floor,
    mxfp8_floor_dim_m,
    mxfp8_floor_swizzle,
    nvfp4_no_gs,
    nvfp4_no_gs_lut,
    nvfp4_no_gs_swizzle,
    nvfp4_with_gs,
)

B200_PEAK_BW_GBPS = 8000.0  # 8 TB/s

# (label, recipe, hop_mode). Recipes that support both HOP and non-HOP routes
# are listed twice with the two modes. Triton-only entries pair a RecipeTriton
# with hop_mode=None and bypass torch.compile.
RECIPES: list[tuple[str, Recipe | RecipeTriton, _HopMode | None]] = [
    ("deepseek_fp8_1_128", deepseek_fp8_1_128, _HopMode.NO_HOP),
    ("deepseek_fp8_1_128_dim_m", deepseek_fp8_1_128_dim_m, _HopMode.NO_HOP),
    ("deepseek_fp8_1_128_dim_m_hop", deepseek_fp8_1_128_dim_m, _HopMode.HOP),
    ("deepseek_fp8_1_128_dim_m_triton", deepseek_fp8_1_128_dim_m_triton, None),
    ("deepseek_fp8_128_128", deepseek_fp8_128_128, _HopMode.NO_HOP),
    ("deepseek_fp8_128_128_hop", deepseek_fp8_128_128, _HopMode.HOP),
    ("deepseek_fp8_128_128_triton", deepseek_fp8_128_128_triton, None),
    ("nvfp4_no_gs", nvfp4_no_gs, _HopMode.NO_HOP),
    ("nvfp4_no_gs_lut", nvfp4_no_gs_lut, _HopMode.NO_HOP),
    ("nvfp4_no_gs_swizzle", nvfp4_no_gs_swizzle, _HopMode.NO_HOP),
    ("nvfp4_with_gs", nvfp4_with_gs, _HopMode.NO_HOP),
    ("mxfp8_floor", mxfp8_floor, _HopMode.NO_HOP),
    ("mxfp8_floor_dim_m", mxfp8_floor_dim_m, _HopMode.NO_HOP),
    ("mxfp8_floor_swizzle", mxfp8_floor_swizzle, _HopMode.NO_HOP),
]
RECIPES_BY_LABEL = {label: (recipe, mode) for label, recipe, mode in RECIPES}


def _bytes_moved(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor | list[torch.Tensor],
) -> int:
    scales = scale if isinstance(scale, list) else [scale]
    scale_bytes = sum(s.numel() * s.element_size() for s in scales)
    return (
        x.numel() * x.element_size()
        + qdata.numel() * qdata.element_size()
        + scale_bytes
    )


def _measure_cpu_time_ms(
    run, n_active: int = 50, trace_path: str | None = None
) -> float:
    """Average host-side CPU time per call, in milliseconds.

    The profiler's first profiled iteration is much slower than steady-state
    (lazy init inside kineto), so we use the schedule API to discard a wait
    + warmup phase and only average the `active` iterations. Caller is
    responsible for warming `run` itself before calling this. If
    `trace_path` is provided, exports a Chrome trace JSON of the active
    window to that path.
    """
    n_wait = 1
    n_warmup = 3
    schedule = torch.profiler.schedule(
        wait=n_wait, warmup=n_warmup, active=n_active, repeat=1
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=False,
    ) as prof:
        for _ in range(n_wait + n_warmup + n_active):
            with torch.profiler.record_function("flex_quant_call"):
                run()
            prof.step()
        torch.cuda.synchronize()

    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    total_cpu_us = 0.0
    for evt in prof.key_averages():
        if evt.key == "flex_quant_call":
            total_cpu_us = evt.cpu_time_total
            break
    return total_cpu_us / n_active / 1e3


def _bench_relu(
    M: int,
    K: int,
    trace_path: str | None = None,
) -> tuple[float, float, float, float]:
    """Eager `torch.relu(x)` baseline: a memory-bound op that moves the same
    volume of bytes as the bfloat16 input (read once, written once)."""
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return torch.relu(x)

    out = run()
    bytes_per_iter = x.numel() * x.element_size() + out.numel() * out.element_size()

    for _ in range(2):
        run()
    torch.cuda.synchronize()

    gpu_time_ms = do_bench_using_profiling(run)
    gpu_gbps = bytes_per_iter / (gpu_time_ms * 1e-3) / 1e9
    gpu_pct_peak = gpu_gbps / B200_PEAK_BW_GBPS * 100
    cpu_time_ms = _measure_cpu_time_ms(run, trace_path=trace_path)
    return gpu_time_ms, gpu_gbps, gpu_pct_peak, cpu_time_ms


def _bench_one(
    recipe_obj: Recipe | RecipeTriton,
    hop_mode: _HopMode | None,
    M: int,
    K: int,
    trace_path: str | None = None,
) -> tuple[float, float, float, float]:
    torch.manual_seed(0)
    torch._dynamo.reset()
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    # Triton-backed recipes skip torch.compile — they're already a kernel.
    if isinstance(recipe_obj, RecipeTriton):
        triton_fn = flex_cast_quant_dense_triton

        def run():
            return triton_fn(
                x,
                block_size=recipe_obj.block_size,
                dim=recipe_obj.dim,
                qdata_dtype=recipe_obj.qdata_dtype,
                scale_dtype=recipe_obj.scale_dtype,
                amax_to_scale_fn_triton=recipe_obj.amax_to_scale_fn,
                cast_to_dtype_fn_triton=recipe_obj.cast_to_dtype_fn,
            )
    else:
        pt_fn = torch.compile(flex_cast_quant_dense, fullgraph=True)

        def run():
            return pt_fn(
                x,
                block_size=recipe_obj.block_size,
                dim=recipe_obj.dim,
                qdata_dtype=recipe_obj.qdata_dtype,
                scale_dtype=recipe_obj.scale_dtype,
                amax_to_scale_fn=recipe_obj.amax_to_scale_fn,
                cast_to_dtype_fn=recipe_obj.cast_to_dtype_fn,
                scale_swizzle=recipe_obj.scale_swizzle,
                _hop_mode=hop_mode,
            )

    qdata, scale = run()
    bytes_per_iter = _bytes_moved(x, qdata, scale)

    # Warmup so first-call costs (compile, autotune, allocator) don't leak
    # into either the GPU timing or the CPU profiler measurement.
    for _ in range(2):
        run()
    torch.cuda.synchronize()

    gpu_time_ms = do_bench_using_profiling(run)
    gpu_gbps = bytes_per_iter / (gpu_time_ms * 1e-3) / 1e9
    gpu_pct_peak = gpu_gbps / B200_PEAK_BW_GBPS * 100
    cpu_time_ms = _measure_cpu_time_ms(run, trace_path=trace_path)
    return gpu_time_ms, gpu_gbps, gpu_pct_peak, cpu_time_ms


def main(
    recipe_filter: str | None = None,
    M: int = 16384,
    K: int = 16384,
    profile_prefix: str | None = None,
) -> None:
    device_name = torch.cuda.get_device_name(0)
    assert "B200" in device_name, f"this benchmark assumes B200, got {device_name!r}"

    if recipe_filter is not None:
        if recipe_filter not in RECIPES_BY_LABEL:
            raise ValueError(
                f"unknown recipe {recipe_filter!r}; available: {list(RECIPES_BY_LABEL)}"
            )
        labels = [recipe_filter]
    else:
        labels = list(RECIPES_BY_LABEL)

    rows = []

    # Baseline: eager `relu` to anchor the bandwidth ceiling.
    if recipe_filter is None:
        baseline_trace = (
            f"{profile_prefix}_relu_M{M}_K{K}.json"
            if profile_prefix is not None
            else None
        )
        gpu_time_ms, gpu_gbps, gpu_pct_peak, cpu_time_ms = _bench_relu(
            M, K, trace_path=baseline_trace
        )
        rows.append(("relu (eager baseline)", gpu_time_ms, gpu_gbps, gpu_pct_peak, cpu_time_ms))

    for label in labels:
        trace_path = (
            f"{profile_prefix}_{label}_M{M}_K{K}.json"
            if profile_prefix is not None
            else None
        )
        recipe_obj, hop_mode = RECIPES_BY_LABEL[label]
        gpu_time_ms, gpu_gbps, gpu_pct_peak, cpu_time_ms = _bench_one(
            recipe_obj, hop_mode, M, K, trace_path=trace_path
        )
        rows.append((label, gpu_time_ms, gpu_gbps, gpu_pct_peak, cpu_time_ms))

    print(f"shape: ({M}, {K}) bfloat16")
    print(
        f"{'recipe':<31} {'gpu_time_ms':>12} {'gpu_gbps':>10} {'gpu_pct_peak':>13} {'cpu_time_ms':>12}"
    )
    for name, gpu_time_ms, gpu_gbps, gpu_pct_peak, cpu_time_ms in rows:
        print(
            f"{name:<31} {gpu_time_ms:>12.4f} {gpu_gbps:>10.1f} {gpu_pct_peak:>12.1f}% {cpu_time_ms:>12.4f}"
        )


if __name__ == "__main__":
    fire.Fire(main)
