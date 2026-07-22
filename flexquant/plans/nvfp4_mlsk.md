# Plan (DEFERRED): nvfp4 fused-swizzle HOP via ported MSLK kernel

> Status: **parked.** Research is done and captured here; implementation is
> deferred because cleanly inlining the user callbacks into an nvfp4 template
> is more involved than a single sitting (see "Why this is hard"). Pick up from
> the "Open design fork" section.

## Goal

A *fused* nvfp4 cast: one kernel that casts bf16 → nvfp4, computes the
two-level (per-tensor fp32 global + per-block e4m3) scale, and writes the
block scale directly in the NVIDIA 32x4x4 swizzled layout — no separate
`to_blocked` rearrange pass. Expose it as a new HOP option behind a new recipe
`nvfp4_with_gs_swizzle`, routed via `_HopMode.HOP`, with the swizzle hardcoded
to `SWIZZLE_32_4_4`.

Contrast with what already shipped: `nvfp4_no_gs_swizzle` produces the swizzled
scale as a **post-pass** (`swizzle.py::to_blocked_2d`). This task makes it
**fused** in the kernel.

User decisions captured before parking:
- **Extend the existing `flex_cast_quant_dense_with_hop`** (don't add a second HOP).
- **Aligned shapes only** initially (`M % 128 == 0`, `N % 64 == 0`).
- **User callbacks must be correctly inlined** into the new template (not
  hardcoded) — this is the requirement that makes the task non-trivial; see below.

## Why this is hard (the blocker to resolve first)

flexquant's HOP callback-inlining (`hop/dynamo_variable.py::_trace_callback`,
`hop/inductor_lowering.py::build_subgraph_buffer` + `{{ modification }}` holes)
inlines callbacks traced on **0-d scalar placeholders**. That only works for
**pointwise** callbacks — the fp8 recipes' `cast_to_dtype_fn` is pure pointwise,
so it inlines fine.

The nvfp4 `cast_to_dtype_fn` ends in `pack_uint4(f32_to_f4_unpacked(...))`,
which packs **2 fp4 values into 1 byte** — a shape-changing op over neighboring
elements, not pointwise. It cannot be traced/inlined at a scalar modification
hole. So the `f32 → fp4` pack must be hardcoded in the template (PTX
`cvt.rn.satfinite.e2m1x2.f32`); the open question is exactly where the
user-callback / framework boundary sits for the cast.

### Open design fork (decide when resuming)

- **Option A (recommended):** inline both user callbacks but split the cast.
  - Hole 0 inlines `amax_to_scale_fn(amax, outer_scale) -> e4m3 inner_scale`
    (same as fp8).
  - Hole 1 inlines a *scaling-only* cast `(tile, inner_scale, outer_scale) ->
    scaled fp32` (the user's math up to but not including the pack).
  - Template hardcodes the `f32 → fp4` PTX pack + the swizzled scale store
    (both are layout/format concerns, legitimately framework-owned).
  - Requires reshaping the `nvfp4_with_gs` `cast_to_dtype_fn` so its tail
    (pack) is separable; recipe's callback returns scaled fp32, and a shared
    pack helper is used by the eager body + reference for bitwise parity.
- **Option B:** inline only `amax_to_scale_fn`; hardcode the entire cast
  (scaling + pack) from MSLK. Simpler template, but the cast math is not
  user-controlled — weaker on the "inline the user callbacks" requirement.
- **Option C:** extend the modification machinery to operate on element-pairs
  so `pack_uint4` is expressible. Most faithful, but changes flexquant's
  subgraph/placeholder infra (currently 0-d scalar only) — significant infra
  work, high risk. Not recommended for a first cut.

## Research findings (de-risk; verified)

1. **MSLK kernel is pure Triton**, not CUDA C++.
   `/home/dev/MSLK/mslk/quantize/triton/fp4_quantize.py`:
   - host `triton_quantize_nvfp4` — L5558
   - kernel `triton_quantize_nvfp4_kernel` — L5647
   - `nvfp4_scale_swizzle` — L5782
   - `convert_fp32_to_fp4_packed` (PTX `cvt.rn.satfinite.e2m1x2.f32`) — L5804
   torchao calls it via `mslk_quantize_nvfp4`
   (`/home/dev/ao/torchao/prototype/mx_formats/kernels.py` L1179-1227) →
   `from mslk.quantize.triton.fp4_quantize import triton_quantize_nvfp4`.

2. **MSLK swizzle byte-layout == torchao `to_blocked` == our `to_blocked_2d`.**
   Hand-traced `nvfp4_scale_swizzle`: row `r = a*32 + b` (b∈[0,32), a∈[0,4))
   → flat offset `b*16 + a*4 + col`, tile-major in (row_block, col_block) order.
   This matches torchao `to_blocked`'s flat order, which we already verified is
   bitwise-equal to `swizzle.py::to_blocked_2d(...).flatten()`. So a kernel that
   writes MSLK flat offsets into a contiguous `(32*n_row_blocks, 16*n_col_blocks)`
   e4m3 buffer yields a tensor `torch.equal` to `to_blocked_2d(row_major_scale)`.
   The new recipe should reuse that `(32*nrb, 16*ncb)` 2D shape convention from
   `nvfp4_no_gs_swizzle`.

3. **MSLK two-level math == flexquant's, modulo op order.**
   MSLK (precise-math path, kernel L5705-5765):
   - `global_scale = reciprocal(per_tensor_scale)`  (host L1192 in torchao)
   - `block_scale = clamp(div_rn(block_amax, 6.0) * global_scale, E4M3_EPS, 448) -> e4m3`
   - `total_scale = div_rn(global_scale, block_scale.to(fp32))`
   - `x_q = round_to_nearest_even_e2m1(x * total_scale)`
   flexquant `outer_scale = amax/(448*6)`, so `global_scale = 1/outer_scale`.
   **For bitwise eager-vs-compile, the recipe reference AND the HOP eager body
   must replicate MSLK's exact op order (`* global_scale`, `div_rn`), not reuse
   `nvfp4_with_gs`'s `/ outer_scale` callbacks verbatim.** Constants:
   `E4M3_EPS = 1.5258789e-05`, `FP8_E4M3_MAX = 448.0`, `FP4_E2M1_MAX = 6.0`.

4. **MSLK aligned path is small.** With `M_PER_BLOCK=128`, `N` tiled by 64,
   no mask, fp32 indexing, precise math, e4m3 (not e8m0): the kernel body is
   ~50 lines (L5692-5778). Drop `USE_MASK`, tail-zeroing (L5678-5690),
   `USE_INT64_INDEXING`, `USE_E8M0_SCALE`, `USE_PRECISE_MATH=False`.

## Files to touch (when resumed)

Existing HOP infra to mirror:
- `flexquant/hop/hop.py` — HOP def, dispatch impls, `register_fake`, tiling keys
  (`_TILING_128_128`, `_TILING_1_128_DIM_M`).
- `flexquant/hop/dynamo_variable.py` — captures callbacks as FX subgraphs.
- `flexquant/hop/inductor_lowering.py` — per-tiling `TritonTemplate` + lowering.
- `flexquant/hop/template_{128_128,1_128_dim_m}.py.jinja` — template patterns
  (note `{{ modification(...) }}` holes and `store_output`).

Changes:
1. **`flexquant/hop/template_nvfp4_swizzle.py.jinja` (NEW)** — port MSLK aligned
   path. Inline callback(s) per the chosen fork option at `{{ modification }}`
   holes; hardcode the `convert_fp32_to_fp4_packed` PTX pack and the
   `nvfp4_scale_swizzle` store (hardcoded swizzle == the `SWIZZLE_32_4_4`
   commitment). Inputs `[x, global_scale, scale]`; primary output `qdata`;
   `mutated_inputs=[scale]`. Attribute the port to MSLK in a header comment.
2. **`flexquant/hop/hop.py`** — add `_TILING_NVFP4_SWIZZLE = (16, -1)`; thread a
   new `global_scale` tensor arg (right after `x`, `None` for fp8 tilings)
   through `__call__` + all dispatch impls + fake. Add
   `_eager_body_nvfp4_swizzle(x, global_scale, <callbacks>)` transcribing
   MSLK's exact arithmetic (finding #3), reusing `nvfp4_utils`
   (`f32_to_f4_unpacked`, `pack_uint4`, `F4_E2M1_MAX`) and `to_blocked_2d`.
   Fake: `qdata = (M, N//2) float4_e2m1fn_x2`,
   `scale = (32*cdiv(M,128), 16*cdiv(N//16,4)) float8_e4m3fn`.
3. **`flexquant/hop/dynamo_variable.py`** — unpack the new `global_scale` slot
   and proxy it as a tensor (like `x`) when not `None`.
4. **`flexquant/hop/inductor_lowering.py`** — new `TritonTemplate` +
   `@SymbolicGridFn` grid `(cdiv(N,64), cdiv(M,128))`; `_lower_nvfp4_swizzle`
   with `qdata_layout = FixedLayout(device, float4_e2m1fn_x2, [M, N//2],
   stride=[N//2,1])`, `scale = empty_strided([32*cdiv(M,128),
   16*cdiv(N//16,4)], None, e4m3, device)`, `input_nodes=[x, global_scale,
   scale]`, `mutated_inputs=[scale]`, subgraphs per fork option,
   `maybe_realize([x, global_scale])`. Relax dtype asserts for the new tiling.
   Watch the qdata store dtype: PTX yields `uint8`; if `store_output` rejects
   the `float4_e2m1fn_x2` layout, allocate `uint8` and `.view(float4_e2m1fn_x2)`
   in the return.
5. **`flexquant/api.py`** — relax the `scale_swizzle` early guard to allow
   `two-level + SWIZZLE_32_4_4 + _hop_mode == HOP`. In the two-level branch for
   that combo: compute `global_scale = outer_scale.reciprocal()`, call the HOP
   with `(x, global_scale, callbacks, block_size=16, dim=-1, qdata_dtype,
   scale_dtype=inner_scale_dtype)`, return `(qdata, [swizzled_inner_scale,
   outer_scale])`. All existing fp8 HOP call sites add `None` in the
   `global_scale` slot.
6. **`flexquant/recipes.py`** — add `nvfp4_with_gs_swizzle` (mirror
   `nvfp4_with_gs` two-level config + `scale_swizzle=SWIZZLE_32_4_4`). Its
   `_reference_fn` transcribes MSLK's exact op order (finding #3) and is the
   shared oracle; factor a shared helper so the HOP eager body and the
   reference call one implementation (guarantees bitwise eager-vs-reference).
7. **`flexquant/test.py`** — import + add
   `("nvfp4_with_gs_swizzle", nvfp4_with_gs_swizzle, _HopMode.HOP)` to
   `RECIPES_PT`. `scale_swizzle` already threads via `_call_pt`.
8. **`flexquant/benchmark.py`** — import + `RECIPES` entry with `_HopMode.HOP`.

## Primary risk

**Bitwise `test_eager_vs_compile`.** Eager body is a PyTorch transcription;
template is the ported PTX kernel — two implementations of one formula, must
agree bitwise. Mitigate by matching op order (finding #3), round-to-nearest-even
on both sides (`f32_to_f4_unpacked` vs PTX `cvt.rn`), `div_rn` both sides,
matched e4m3 clamp bounds. **Do not reintroduce a tolerance** — the suite's
contract is now bitwise (see the earlier tolerance-removal change). If a 1-ULP
mismatch appears, align the specific op in the eager body to the template.

## Verification (when resumed)

```bash
pytest test.py -v -k nvfp4_with_gs_swizzle   # eager-vs-ref, eager-vs-compile (bitwise), cuda-graph
pytest test.py -v -k hop                      # fp8 HOPs still pass (global_scale=None threading)
pytest test.py                                # full suite: expect 31 -> 34
python benchmark.py --recipe_filter nvfp4_with_gs_swizzle   # B200, compare vs nvfp4_with_gs / nvfp4_no_gs_swizzle
```
Pre-test sanity on 256×256 bf16: HOP-compiled output `torch.equal` to the
reference (qdata via `.view(uint8)`, swizzled scale, `outer_scale`); swizzled
scale `(64,64)` e4m3; qdata `(256,128)` float4_e2m1fn_x2; optionally confirm
fused scale `.flatten()` == `to_blocked_2d(row_major_inner_scale).flatten()`.
