# quant_cast_bench

Memory-bandwidth benchmark for the `quant_cast_gold` recipes. Each recipe is a memory-bound
cast, so the signal is achieved memory bandwidth vs. the B200 ceiling (8 TB/s). Per `mode`, the
benchmark either `torch.compile`s each gold recipe's reference fn (`compile`, the default) or
runs its hand-written kernels (`triton` / `cute`), times each with `do_bench_using_profiling`,
and reports latency + GB/s + % of peak. The `relu (baseline)` row anchors the achievable ceiling
for the shape.

## torchinductor gaps vs triton

* square quant block sizes (32x32, 128x128, etc)
  - For example, on fp8_deepseek_128x128, inductor 44.3% peak mem -> triton 77.1% peak mem
* reductions across M-dim, or K-dim and M-dim in the same kernel
  - For example, on mxfp8_floor_dim_m, inductor 17.5% peak mem -> triton 59.9% peak mem
* fp4
  - nvfp4_swizzle: inductor 23.3% peak mem -> triton 62.6% peak mem

## triton gaps vs SOL (CUDA / CUTLASS / cute)

* reductions across M-dim, or K-dim and M-dim in the same kernel
  - For example, on mxfp8_floor_dim_m, triton 59.9% peak mem -> CUDA 67.7% peak mem

    - The CUDA kernel writes quantized values directly into a transposed smem layout (out_colwise_sh[col][row]) and TMA-stores that smem tile. Triton has no
      user-facing __shared__ + __syncthreads(), so every transpose goes through the compiler's tl.trans — which either (a) produces the uncoalesced 21-sectors/request store, or (b) if you TMA-store it, pays a register→smem
      transpose tax. This is the crux: CUDA decouples "coalesced transposed store" from "small register footprint," and Triton cannot.
    - Efficient TMA transfers and high occupancy at the same time. CUDA gets both because it manages smem by hand and uses tiny 64-thread CTAs. In Triton, TMA transfer size and occupancy are both governed by the tile size, so
      you're forced to choose — big tiles (efficient TMA, low occupancy) or small tiles (high occupancy, tiny inefficient TMA transfers).

## Repro

```bash
cd /home/dev/pytorch_scripts

# torch.compile the gold reference fns (default mode)
python quant_cast_bench/benchmark.py --mode compile

# hand-written Triton kernels
python quant_cast_bench/benchmark.py --mode triton

# optional: single shape / single recipe
python quant_cast_bench/benchmark.py --mode triton --M 16384 --K 16384
python quant_cast_bench/benchmark.py --mode triton --recipe_name_filter mxfp8_floor_dim_m
```

Default shape is `(M, K) = (16384, 16384)`. Assumes a B200 (peak 8 TB/s).

## Output

### `--mode compile`

```
shape: (16384, 16384)  mode: compile
recipe                            gpu_time_ms    gbps    pct_peak  perf_description
------------------------------  -------------  ------  ----------  -------------------------------------------------
relu (baseline)                        0.1792  5993.2       74.9%
fp8_tensorwise_precalc_scale            0.143  5632.1       70.4%  elementwise
mxfp8_floor_swizzle                    0.1301  6256.2       78.2%  (1,32) block, swizzle
mxfp8_floor_dim_m                      0.5811  1400.3       17.5%  (32,1) block, t-contig
mxfp8_floor_dim_km                     0.7127  1530.1       19.1%  (1,32) dim-k + (32,1) dim-m, one pass, t-contig
mxfp8_32x32_floor                      0.3793  2123.7       26.5%  (32,32) block
fp8_deepseek_1x128                     0.1318  6171.9       77.1%  (1,128) block
fp8_deepseek_1x128_dim_m               0.2498  3257.7       40.7%  (128,1) block, t-contig
fp8_deepseek_1x128_dim_km              0.3808  2863.4       35.8%  (1,128) dim-k + (128,1) dim-m, one pass, t-contig
fp8_deepseek_128x128                    0.227    3548       44.4%  (128,128) block
fp8_rowwise                            0.1224  6577.4       82.2%  (1,-1) block
fp8_colwise                            0.3928  2050.4       25.6%  (-1,1) block, t-contig
nvfp4_swizzle                          0.3658  1880.2       23.5%  (1,16) block, fp4 qdata, swizzle
bf16_rht                               0.4596  2336.1       29.2%  elementwise RHT
fp32_to_bf16_sr                        0.6822    2361       29.5%
fp32_to_bf16_sr_global_offsets         2.8956   556.2        7.0%  elementwise SR with stateless RNG
```

### `--mode triton`

```
shape: (16384, 16384)  mode: triton
recipe                          gpu_time_ms    gbps    pct_peak  perf_description
----------------------------  -------------  ------  ----------  -------------------------------------------------
relu (baseline)                      0.1792  5992.8       74.9%
fp8_tensorwise_precalc_scale         0.1422    5663       70.8%  elementwise
mxfp8_floor_swizzle                  0.1248  6518.6       81.5%  (1,32) block, swizzle
mxfp8_floor_dim_m                    0.1692  4810.2       60.1%  (32,1) block, t-contig
mxfp8_floor_dim_km                   0.2893  3769.3       47.1%  (1,32) dim-k + (32,1) dim-m, one pass, t-contig
mxfp8_32x32_floor                    0.1287  6257.8       78.2%  (32,32) block
fp8_deepseek_1x128                   0.1341    6068       75.9%  (1,128) block
fp8_deepseek_1x128_dim_m             0.1506  5404.2       67.6%  (128,1) block, t-contig
fp8_deepseek_1x128_dim_km            0.2358  4625.7       57.8%  (1,128) dim-k + (128,1) dim-m, one pass, t-contig
fp8_deepseek_128x128                 0.1306  6166.2       77.1%  (128,128) block
fp8_rowwise                          0.1291  6238.8       78.0%  (1,-1) block
fp8_colwise                           0.276  2917.8       36.5%  (-1,1) block, t-contig
nvfp4_swizzle                        0.1371  5017.3       62.7%  (1,16) block, fp4 qdata, swizzle
```

## Known issues

* `fp32_to_bf16_sr` (compile) reports only ~19.6% peak, but this understates the real bandwidth.
  The stochastic-rounding uniform is drawn via `torch.func._random.uniform` → `aten._philox_uniform`,
  which inductor treats as an opaque extern op rather than a fusible in-kernel RNG. So it runs as
  two DRAM passes: kernel 1 materializes a full-size fp32 random tensor (~1.07 GB write, ~63% of
  the runtime), kernel 2 reads it back alongside `x` to dither+truncate. Real traffic is ~3.76 GB
  (write u + read x + read u + write out) ≈ 46% of peak; the benchmark only counts input+output
  (~1.61 GB), so the wasted RNG round-trip shows up as the low 19.6%. Fix: fuse the Philox RNG into
  the dither kernel (generate uniforms in-register, never materialize) — as inductor already does
  for `torch.rand`/dropout — which would cut traffic to ~1.61 GB and approach the relu ceiling
  (~2–3× speedup).

* `bf16_rht` (compile) runs at only ~29% peak, and here the traffic is not wasted (the whole 1.07 GB
  is useful read x + write out) — it's GEMM-kernel inefficiency. The 16×16 RHT `x.reshape(..., 16) @ rht`
  is lowered to a single cuBLAS GEMM via `extern_kernels.mm`, shape `(M·N/16, 16) @ (16, 16)` — i.e.
  `K=16, N=16`. That GEMM is ~99.5% of the runtime. The op is really memory-bound (~4 flop/byte), but
  cuBLAS runs it as a compute-oriented matmul, and the skinny `K=N=16` shape tiles terribly (N-tiling
  wasted, no K-reuse to amortize), so it fails to saturate DRAM — 29% vs the ~75% relu ceiling for the
  same 1.07 GB (~2.6× slower than bandwidth-bound). Fix direction: a fused kernel that loads a 16-vector,
  applies the transform in registers, and writes 16 (or a Triton matmul template tuned for the skinny
  shape) would approach the relu ceiling.

* `nvfp4_swizzle` (compile) runs at only ~23% peak (vs the Triton kernel's 62.6%), because inductor
  splits it into **3 separate kernels** instead of one fused pass:
  1. per-16-block `amax` reduction (reads `x` → block amaxes),
  2. quantize: reads **`x` again** + outer scale + amaxes, computes the inner e4m3 scale and the fp4
     data, writes packed nvfp4 (this kernel does contain the hardware `cvt.rn.satfinite.e2m1x2.f32`
     fp4 encode — see `_f32_to_packed_fp4`),
  3. a `permute/transpose` scatter that writes the swizzled inner scale.
  So `x` is streamed **twice** (the quantize can't fuse with the reduction it depends on) and the
  swizzle is a separate scatter. The hand-written Triton kernel collapses all of this into **one**
  pass — load each block once, reduce to amax in-register, quantize from those registers, and write
  both the fp4 data and the swizzled scale — which is the bulk of the 62.6% vs 23% gap. Note the fp4
  encode itself is *not* the bottleneck (~60% in isolation); adding the hardware `cvt` (gated to
  compile via `inline_asm_elementwise`, like torchao's `_to_mx_rceil`) only moved it 20.7% → 23%.
  Fix direction: a single fused reduce+quantize+swizzle kernel (what Triton/CUDA do), which inductor
  won't generate here.

* `fp8_deepseek_1x128_dim_km` (compile) runs at only ~35.8% peak. The gold recipe expresses a
  **single pass** that reads `x` once and reduces it both ways (dim-K = 1×128 along columns, dim-M =
  128×1 along rows) to emit all four outputs — but inductor generates **3 kernels that each read `x`
  (x streamed 3×)**:
  1. dim-K, fully fused — a persistent reduction that reads `x`, computes the per-128 amax + scale,
     and quantizes in one kernel → `qdata_k` + `scale_k`;
  2. dim-M `amax` reduction — reads `x` again, reduces over the 128 rows → `scale_m`;
  3. dim-M quantize + transpose — reads `x` a **third** time + the dim-M scale → `qdata_m` (transposed).
  Two structural reasons: dim-K and dim-M are treated as independent subgraphs so they don't share
  the load of `x`, and dim-M splits reduce-from-normalize (the quantize depends on the reduction, same
  pattern as `nvfp4_swizzle`). So the ~35.8% is roughly the cost of ~3 passes over `x` plus the
  transposed dim-M store, not the intended single pass. All 3 are Triton (no cuBLAS/extern).
  **The hand-written Triton kernel realizes the single pass (57.8%, ~1.6× the compile 35.8%)**: one
  128×128 tile of `x` is loaded once, reduced both ways in-register (128 columns for dim-K, 128 rows
  for dim-M), and all four outputs are written (dim-M transposed). It lands near the standalone
  `fp8_deepseek_1x128_dim_m` (~68%) rather than the standalone dim-K (~76%), because the transposed
  dim-M store is the binding cost — dim-K rides the already-loaded tile essentially for free.

* `mxfp8_floor_dim_km` (compile ~19.1%, triton ~47.1%) — the mxfp8-floor analog of the above (1×32
  dim-K + 32×1 dim-M, e8m0 scales). Same story under compile: inductor generates **3 kernels reading
  `x` 3×** (dim-K fused reduce+quantize; dim-M amax reduction; dim-M quantize+transpose). The
  hand-written Triton kernel does the single pass. A first fixed **32×32 tile** version only reached
  ~30.9% (the transposed dim-M store is only 32-wide, poorly coalesced, and each program does little
  work); switching to **blocked tiles** (autotuned `RB` 32-row blocks × `BN` cols, reshape per
  direction — the same lever that fixed `mxfp8_floor_dim_m`) widens the dim-M store and raises
  occupancy, reaching **47.1%**. Still below deepseek's dim_km (57.8%) because the 32-block
  granularity means 4× as many e8m0 scales (M/32 vs M/128) plus the per-scale e8m0 bit-math, and the
  transposed dim-M store remains the binding cost.

* `fp32_to_bf16_sr_global_offsets` (compile) runs at only ~6.2% peak — ~3.2× slower than
  `fp32_to_bf16_sr` (19.6%) for identical dithering math. The difference is how the Philox draw is
  keyed. Both use `torch.func._random.uniform` (experimental stateless Philox → unfused
  `aten._philox_uniform`, so both share the same materialized-`u` ~46%-real-BW ceiling). The plain
  variant keys on tile-LOCAL position (one shared key, counter = flat index within the call), which
  is cheap but changes with tiling. The global variant is tile-INVARIANT: it keys each draw on the
  element's GLOBAL index, which — because `uniform` only exposes a single scalar starting offset (the
  `(seed, offset)` key pair, fine for 1D/full-width tiles but not 2D sub-blocks) — forces
  materializing a per-element `(numel, 2)` uint64 key tensor = **4.29 GB** (16 B/element, 8× the
  0.54 GB bf16 output), written then read back by a batched Philox. That ~8.5 GB key round-trip
  triples total traffic (~12.3 GB vs ~3.76 GB), matching the 3.2× slowdown.

### Fix direction: key by global index without materializing keys

The global index only needs to reach Philox as a *counter*. Today the sole knob is the key's single
scalar `offset` — one value per call, which can only shift a 1-D contiguous stream, so it cannot
express a 2-D sub-block's global index and the recipe is forced to fold the index into a
**per-element key tensor** (`(numel, 2)` uint64, 4.29 GB). If `uniform` instead accepted a per-element
**affine counter** (a `base` plus per-dim `strides`), element `(i, j)` could take
`counter = base + i·num_col + j` computed *in-kernel from its own indices* — one shared key, zero
materialized index/key tensors. Combined with a fusible in-kernel Philox (as Triton's
`tl.rand(seed, offset)` already allows), the whole tile-invariant SR becomes one fused kernel that
never materializes `u` either — approaching the relu ceiling.

<table>
<tr><th>Current — per-element key tensor (4.29 GB)</th><th>Ideal — shared key + in-kernel affine counter</th></tr>
<tr><td>

```python
# to key on the GLOBAL index, fold it into a
# distinct key per element:
i = (global_row + arange(M)).view(-1, 1)
j = (global_col + arange(N)).view(1, -1)
gidx = (i * num_col + j).reshape(-1)   # global index
seed = key[0:1].expand(gidx.numel())
keys = stack([seed, gidx], -1).to(uint64)
#      ^ (numel, 2) uint64 = 4.29 GB  <-- materialized
u = uniform(keys, (gidx.numel(),))
#   ^ batched philox reads 4.29 GB keys back,
#     writes u (1.07 GB)  <-- also materialized
rand16 = (u * 65536).to(int32)
```

</td><td>

```python
# one shared key; per-element counter is an affine
# map of the element's coords, computed in-kernel:
u = uniform(
    key,                       # single (seed, offset) key
    (M, N),
    counter_base=global_row * num_col + global_col,
    counter_strides=(num_col, 1),
)   # counter(i,j) = base + i*num_col + j
#   no key tensor; fusible -> u never materialized
rand16 = (u * 65536).to(int32)
```

</td></tr>
</table>

## Why `mxfp8_floor_dim_m` (59.9%) is slower than `fp8_deepseek_1x128_dim_m` (71.2%)

Both are the same shape of kernel — load a bf16 tile, reduce down M per column, scale, `.to(fp8)`,
transposed store — and both are memory-bound. The gap is entirely about **register pressure**, which
decides whether you can afford a *tall* tile (good transposed-store coalescing) at high occupancy.

* **deepseek is light:** one `amax` over the whole 128-row block → one fp32 scale/column, `x/scale`.
  At a **128-row tile** it uses ~72 reg/thread → 40% occupancy **and** 128-wide coalesced stores
  (~17.8 sectors/req) → **68.6% DRAM**. It gets both at once.
* **mxfp8 is heavier:** it reshapes `(RB·32, BN) → (RB, 32, BN)`, reduces per **32-row sub-block**
  (4 scales/column at RB=4, vs deepseek's 1), and quantizes to e8m0. At the same 128-row tile that
  needs ~121–127 reg/thread → only ~23% occupancy → ~40–44% DRAM. So the autotuner is forced to a
  **32-row tile**, which restores occupancy (~40%) but narrows the transposed store to 32-wide
  (~21.3 sectors/req) → **~60% DRAM**. mxfp8 is stuck in an occupancy-vs-coalescing bind that
  deepseek's lighter math avoids.

ncu at a matched 128-row tile (`RB=4, BN=64`):

| | reg/thread | occupancy | store sectors/req | DRAM % |
|---|---|---|---|---|
| deepseek (its real tile) | 72 | 40.5% | 17.8 | 68.6% |
| mxfp8 (forced small tile) | 69 | 40.5% | 21.3 | 57–60% |
| mxfp8 at deepseek's tile | 121–127 | 23.3% | 16.0 | 39–44% |

The register cost is **structural, not the e8m0 math**: replacing the manual e8m0-floor bit
extraction with the hardware `cvt.rz.satfinite.ue8m0x2.f32` instruction (see the mxfp8 kernel) only
freed ~6 registers (127→121) and moved the number 57.9% → 59.9% — the bulk of the pressure is the
fp32 working tile plus the `(RB,32,BN)` reshape and `tl.trans` transpose staging, plus holding 4×
the per-column scale state. Closing the gap to deepseek would require cutting *that* (e.g. a
shared-memory transpose to decouple store coalescing from tile height), not cheaper scale math.
