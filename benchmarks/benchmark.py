"""Memory-bandwidth benchmark for quant_cast_gold recipes.

Each recipe is a memory-bound cast, so the signal we care about is achieved memory bandwidth
vs. the B200 ceiling (8 TB/s). Per `mode`, we either torch.compile each gold recipe's reference
fn ("compile", the default) or run its hand-written Triton kernel ("triton"), time it with
`do_bench_using_profiling`, and report latency + GB/s + % of peak. Structured after
flexquant/benchmark.py.
"""

import csv as _csv
import os
import sys

import fire
import tabulate
import torch
from torch._inductor.utils import do_bench_using_profiling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_bench.quant_cast_gold.recipes import ALL_RECIPES

B200_PEAK_BW_GBPS = 8000.0  # 8 TB/s

# Recipes excluded from the benchmark entirely -- not relevant here (they still run in the
# gold tests). Filtered out before the sweep, so they never appear in the results table.
_BENCH_SKIP = {
    "mxfp8_floor",
    "nvfp4_blocked_outer",
    "mxfp8_bias",
    "fp8_rowwise_precalc_scale",
    "fp8_colwise_precalc_scale",
}


def _bytes_moved(inputs, outputs):
    # a cast reads its inputs and writes its outputs; bytes moved = element bytes across both.
    tensors = [t for t in (*inputs, *outputs) if isinstance(t, torch.Tensor)]
    return sum(t.numel() * t.element_size() for t in tensors)


def _bench_relu(M, K):
    # eager torch.relu baseline: a trivially memory-bound op (read x, write relu(x), both bf16)
    # that anchors the achievable-bandwidth ceiling for this shape.
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
    gbps = bytes_per_iter / (gpu_time_ms * 1e-3) / 1e9
    pct_peak = gbps / B200_PEAK_BW_GBPS * 100
    return gpu_time_ms, gbps, pct_peak


def _bench_one(recipe, M, K, mode):
    torch.manual_seed(0)
    torch._dynamo.reset()
    inputs = recipe.example_input_fn(M, K)  # (x, *aux)
    # Some recipes (e.g. fp32_to_bf16_sr_global_offsets) consume flex_tile_map framework kwargs
    # naming the tile's global origin + parent row stride. The benchmark runs the whole tensor as a
    # single tile, so origin = (0, 0) and num_col = the full width. Every recipe fn takes **kwargs,
    # so these are ignored by the recipes that don't use them (verified across all benchmarked fns).
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    if mode == "flex_tile_map_triton":
        # drive the flex_tile_map TRITON_TEMPLATE backend: `f` is traced and lowered onto the
        # hand-written Triton template (dim-M group reduction). The caller owns the compile
        # decision (like flex_attention), so we torch.compile the flex_tile_map entrypoint here.
        # Aux inputs aren't wired through the template path yet.
        from quant_cast_bench.flex_tile_map.api import FlexTileMapBackend, flex_tile_map

        assert len(inputs) == 1, "flex_tile_map_triton: single-input recipes only (no aux) so far"
        f = recipe.pt_ref_fn
        compiled = torch.compile(flex_tile_map)

        def run():
            return compiled(inputs[0], f, _backend=FlexTileMapBackend.TRITON_TEMPLATE)
    elif mode == "compile":
        # the generic (no-template) path: drive the reference `f` through flex_tile_map's REFERENCE
        # backend under torch.compile. REFERENCE calls `f(input, *aux, global_row=.., global_col=..,
        # num_col=..)` directly (no HOP), so regular Inductor lowers it -- the same kernel as
        # compiling pt_ref_fn directly, but routed through the flex_tile_map entrypoint so it diffs
        # apples-to-apples against flex_tile_map_triton. aux_inputs are forwarded whole (REFERENCE
        # treats the whole tensor as one tile, so aux_kinds tiling metadata is irrelevant here);
        # the tile_kwargs above are supplied by flex_tile_map itself, not passed in.
        from quant_cast_bench.flex_tile_map.api import FlexTileMapBackend, flex_tile_map

        f = recipe.pt_ref_fn
        compiled = torch.compile(flex_tile_map, fullgraph=True)

        def run():
            return compiled(
                inputs[0],
                f,
                aux_inputs=tuple(inputs[1:]),
                _backend=FlexTileMapBackend.REFERENCE,
            )
    else:
        # "triton"/"cute": run the recipe's hand-written kernel directly.
        fn = recipe.triton_fn if mode == "triton" else recipe.cute_fn

        def run():
            return fn(*inputs, **tile_kwargs)

    outputs = run()
    bytes_per_iter = _bytes_moved(inputs, outputs)

    # warm up so first-call costs (compile, autotune, allocator) don't leak into the timing.
    for _ in range(2):
        run()
    torch.cuda.synchronize()

    gpu_time_ms = do_bench_using_profiling(run)
    gbps = bytes_per_iter / (gpu_time_ms * 1e-3) / 1e9
    pct_peak = gbps / B200_PEAK_BW_GBPS * 100
    return gpu_time_ms, gbps, pct_peak


def main(
    M: int = 16384,
    K: int = 16384,
    recipe_name_filter: str | None = None,
    mode: str | None = None,
    csv: str | None = None,
):
    """Benchmark the quant-cast recipes for one `mode` and print a table.

    Pass `--csv PATH` to also MERGE the numeric results (long format:
    kernel,mode,gpu_time_ms,gbps,mem_bw_pct) into PATH, treated as a small database: existing rows
    are read, the (kernel, mode) keys computed this run are upserted, and the file is rewritten.
    Run once per mode to build the file the chart script (plot_bench.py) reads; re-running a mode
    just updates its rows in place (idempotent, no duplicates).
    """
    device_name = torch.cuda.get_device_name(0)
    assert "B200" in device_name, f"this benchmark assumes B200, got {device_name!r}"

    mode = mode or "compile"
    assert mode in ("compile", "triton", "cute", "flex_tile_map_triton"), (
        f"mode must be 'compile', 'triton', 'cute', or 'flex_tile_map_triton', got {mode!r}"
    )

    # "compile" sweeps the gold recipes through flex_tile_map's REFERENCE backend under
    # torch.compile (regular Inductor lowers `f`); "triton"/"cute" sweep the hand-written kernel
    # sets (triton_fn / cute_fn); "flex_tile_map_triton" sweeps the flex_tile_map RecipeV2 set
    # through the TRITON_TEMPLATE backend (the hand template). All recipe kinds carry
    # example_input_fn / perf_description, so the rest of the sweep is identical.
    if mode == "triton":
        from quant_cast_bench.quant_cast_triton.recipes import ALL_RECIPES as recipes_all
    elif mode == "cute":
        from quant_cast_bench.quant_cast_cute.recipes import ALL_RECIPES as recipes_all
    elif mode == "flex_tile_map_triton":
        from quant_cast_bench.flex_tile_map.recipes import RECIPES_V2
        # wired through the TRITON_TEMPLATE backend: deepseek 1x128 dim-M (the in-fragment
        # group-reduction path, transposed outputs), single-input, no aux. The pointwise relu path
        # was removed -- a pointwise `f` no longer has a template lowering (it raises
        # NotImplementedError), so it's benchmarked via regular Inductor (`--mode compile`) instead.
        _WIRED = {"fp8_deepseek_1x128_dim_m"}
        recipes_all = [(n, r) for n, r in RECIPES_V2 if n in _WIRED]
    else:
        recipes_all = ALL_RECIPES

    recipes = [
        (n, r)
        for n, r in recipes_all
        if n not in _BENCH_SKIP
        and (recipe_name_filter is None or recipe_name_filter in n)
    ]
    if not recipes:
        raise ValueError(
            f"no recipe matched {recipe_name_filter!r}; have {[n for n, _ in recipes_all]}"
        )

    rows = []  # (recipe, gpu_time_ms, gbps, pct_peak, perf_description)
    csv_rows = []  # (kernel, mode, gpu_time_ms, gbps, mem_bw_pct) numeric, successes only

    # relu baseline anchors the bandwidth ceiling; shown on a full sweep (no filter).
    if recipe_name_filter is None:
        ms, gbps, pct = _bench_relu(M, K)
        rows.append(("relu (baseline)", f"{ms:.4f}", f"{gbps:.1f}", f"{pct:.1f}%", ""))
        csv_rows.append(("relu (baseline)", mode, ms, gbps, pct))

    for name, recipe in recipes:
        # TODO: some recipes don't benchmark cleanly yet (e.g. fp4-packed byte accounting under
        # torch.compile, swizzle grids, SR's fp32/const input). Skip failures for now so the
        # sweep still reports the ones that work; revisit each skipped recipe.
        try:
            ms, gbps, pct = _bench_one(recipe, M, K, mode)
        except Exception as e:
            reason = f"SKIPPED: {type(e).__name__}: {str(e).splitlines()[0][:60]}"
            rows.append((name, reason, "", "", recipe.perf_description))
            continue
        rows.append((name, f"{ms:.4f}", f"{gbps:.1f}", f"{pct:.1f}%", recipe.perf_description))
        csv_rows.append((name, mode, ms, gbps, pct))

    print(f"shape: ({M}, {K})  mode: {mode}")
    print(
        tabulate.tabulate(
            rows,
            headers=["recipe", "gpu_time_ms", "gbps", "pct_peak", "perf_description"],
            colalign=("left", "right", "right", "right", "left"),
        )
    )

    # Optionally merge numeric results (long format) into the CSV, treated as a small database:
    # read existing rows, UPSERT the (kernel, mode) keys computed this run, keep everything else,
    # and rewrite the file. Re-running a mode is idempotent (updates in place, no duplicate rows).
    if csv is not None:
        header = ["kernel", "mode", "gpu_time_ms", "gbps", "mem_bw_pct"]
        merged = []           # rows in first-seen order (stable across re-runs)
        index = {}            # (kernel, mode) -> position in `merged`
        if os.path.exists(csv):
            with open(csv, newline="") as f:
                reader = _csv.reader(f)
                next(reader, None)  # skip header
                for r in reader:
                    if len(r) < len(header):
                        continue
                    index[(r[0], r[1])] = len(merged)
                    merged.append(r[: len(header)])
        for kernel, md, ms, gbps, pct in csv_rows:
            row = [kernel, md, f"{ms:.4f}", f"{gbps:.1f}", f"{pct:.2f}"]
            key = (kernel, md)
            if key in index:
                merged[index[key]] = row       # update
            else:
                index[key] = len(merged)        # insert
                merged.append(row)
        with open(csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(header)
            w.writerows(merged)
        print(f"wrote {len(merged)} rows to {csv} ({len(csv_rows)} upserted for mode {mode!r})")


if __name__ == "__main__":
    fire.Fire(main)
