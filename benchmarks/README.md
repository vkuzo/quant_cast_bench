# quant_cast_bench

Memory-bandwidth benchmark for the `quant_cast_gold` recipes.

## Repro

```bash
cd /home/dev/quant_cast_bench

# torch.compile the gold reference fns (default mode)
# modes: compile, triton, cute, helion
python benchmarks/benchmark.py --mode compile

# optional: single shape / single recipe
python benchmarks/benchmark.py --mode triton --M 16384 --K 16384
python benchmarks/benchmark.py --mode triton --recipe_name_filter mxfp8_dim_m
```

Default shape is `(M, K) = (16384, 16384)`. Assumes a B200 (peak 8 TB/s).

## Output

![Memory bandwidth by mode](mem_bw.png)

### `--mode compile`

```
shape: (16384, 16384)  mode: compile
versions: torch 2.14.0.dev20260720+cu130, helion 1.2.0, cutlass-dsl 4.5.2
recipe                                                                     gpu_time_ms    gbps    pct_peak  perf_description
--------------------------------------------  ----------------------------------------  ------  ----------  ------------------------------------------------------------------------------------------------------
relu (baseline)                                                                 0.1773  6057.3       75.7%
fp8_tensorwise_precalc_scale                                                    0.1413  5698.4       71.2%  elementwise
mxfp8_swizzle                                                                   0.1341  6069.3       75.9%  (1,32) block, swizzle
fp8_deepseek_1x128                                                              0.1310  6211.8       77.6%  (1,128) block
mxfp8_dim_m                                                                     0.6089  1336.4       16.7%  (32,1) block, t-contig
mxfp8_dim_m_swizzle                                                             0.4147  1962.1       24.5%  (32,1) block, t-contig, swizzle
fp8_deepseek_1x128_dim_m                                                        0.2534  3210.6       40.1%  (128,1) block, t-contig
mxfp8_dim_km                                                                    0.7366  1480.4       18.5%  (1,32) dim-k + (32,1) dim-m, one pass, t-contig
mxfp8_dim_km_swizzle                                                            0.5562  1960.7       24.5%  (1,32) dim-k + (32,1) dim-m, one pass, t-contig, swizzle
fp8_deepseek_1x128_dim_km                                                       0.3843  2837.4       35.5%  (1,128) dim-k + (128,1) dim-m, one pass, t-contig
mxfp8_32x32                                                                     0.3591  2243.0       28.0%  (32,32) block
mxfp8_32x32_swizzle                                                             0.3605  2257.3       28.2%  (32,32) block, swizzle
mxfp8_32x32_dim_m_swizzle                                                       0.3963  2053.2       25.7%  (32,32) block, t-contig, swizzle
mxfp8_32x32_dim_km_swizzle                                                      0.5347  2039.5       25.5%  (32,32) block, one pass, t-contig, swizzle
mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle                                    0.3665  2243.2       28.0%  (32,32) block, one pass, dim-k qdata + km scales, swizzle
fp8_deepseek_128x128                                                            0.2257  3568.9       44.6%  (128,128) block
fp8_rowwise                                                                     0.1235  6522.2       81.5%  (1,-1) block
fp8_colwise                                                                     0.3892  2069.2       25.9%  (-1,1) block, t-contig
nvfp4                                                                           0.3698  1860.1       23.3%  (1,16) block, fp4 qdata, no swizzle
nvfp4_swizzle                                                                   0.3819  1800.9       22.5%  (1,16) block, fp4 qdata, swizzle
nvfp4_dim_m_swizzle                                                             0.5123  1342.8       16.8%  (1,16) block, fp4 qdata, t-contig, swizzle; dim-m
nvfp4_dim_km_swizzle                                                            0.8910   941.5       11.8%  (1,16) block, fp4 qdata, swizzle; dim-k + dim-m (no RHT), per-orientation outer scale
nvfp4_dim_m_rht_swizzle                                                         1.0437   659.1        8.2%  (1,16) block, fp4 qdata, swizzle; dim-m (RHT), one outer scale
nvfp4_sr_swizzle                                                                0.7745   888.1       11.1%  (1,16) block, fp4 qdata (stochastic rounding), swizzle
nvfp4_dim_m_rht_sr_swizzle                                                      1.4366   478.8        6.0%  (1,16) block, fp4 qdata (stochastic rounding), swizzle; dim-m (RHT), one outer scale
nvfp4_swizzle_dim_k_dim_m_rht                                                   1.3914   602.9        7.5%  (1,16) block, fp4 qdata, swizzle; dim-k (no RHT) + dim-m (RHT), two outer scales
nvfp4_swizzle_dim_k_sr_dim_m_rht_sr                                             2.2085   379.8        4.7%  (1,16) block, fp4 qdata (stochastic rounding), swizzle; dim-k (no RHT) + dim-m (RHT), two outer scales
bf16_rht                                                                        0.4583  2342.9       29.3%  elementwise RHT
fp32_to_bf16_sr                                                                 0.6831  2357.9       29.5%
fp32_to_bf16_sr_global_offsets                SKIPPED: Unsupported: Observed exception                      elementwise SR with stateless RNG
debug_relu                                                                      0.1639  6552.3       81.9%  debug: relu, elementwise, no quant
```

### `--mode triton`

```
shape: (16384, 16384)  mode: triton
versions: torch 2.14.0.dev20260720+cu130, helion 1.2.0, cutlass-dsl 4.5.2
recipe                                          gpu_time_ms    gbps    pct_peak  perf_description
--------------------------------------------  -------------  ------  ----------  ---------------------------------------------------------
relu (baseline)                                      0.1773  6057.6       75.7%
fp8_tensorwise_precalc_scale                          0.141  5711.5       71.4%  elementwise
mxfp8_swizzle                                        0.1225  6643.9       83.0%  (1,32) block, swizzle
fp8_deepseek_1x128                                    0.135  6028.7       75.4%  (1,128) block
mxfp8_dim_m                                          0.1628  4999.1       62.5%  (32,1) block, t-contig
mxfp8_dim_m_swizzle                                  0.1534  5305.7       66.3%  (32,1) block, t-contig, swizzle
fp8_deepseek_1x128_dim_m                             0.1439  5655.8       70.7%  (128,1) block, t-contig
mxfp8_dim_km                                         0.2556  4267.2       53.3%  (1,32) dim-k + (32,1) dim-m, one pass, t-contig
mxfp8_dim_km_swizzle                                 0.2239  4870.6       60.9%  (1,32) dim-k + (32,1) dim-m, one pass, t-contig, swizzle
fp8_deepseek_1x128_dim_km                             0.235  4641.3       58.0%  (1,128) dim-k + (128,1) dim-m, one pass, t-contig
mxfp8_32x32                                          0.1278  6302.9       78.8%  (32,32) block
mxfp8_32x32_swizzle                                  0.1312  6200.1       77.5%  (32,32) block, swizzle
mxfp8_32x32_dim_m_swizzle                            0.1454    5598       70.0%  (32,32) block, t-contig, swizzle
mxfp8_32x32_dim_km_swizzle                           0.1854  5883.2       73.5%  (32,32) block, one pass, t-contig, swizzle
mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle          0.134  6133.2       76.7%  (32,32) block, one pass, dim-k qdata + km scales, swizzle
fp8_deepseek_128x128                                 0.1315  6124.2       76.6%  (128,128) block
fp8_rowwise                                          0.1294  6222.7       77.8%  (1,-1) block
fp8_colwise                                          0.2217  3632.2       45.4%  (-1,1) block, t-contig
nvfp4                                                0.1284  5355.6       66.9%  (1,16) block, fp4 qdata, no swizzle
nvfp4_swizzle                                        0.1375  5004.3       62.6%  (1,16) block, fp4 qdata, swizzle
bf16_rht                                             0.1991  5392.2       67.4%  elementwise RHT
fp32_to_bf16_sr                                      0.2734  5891.8       73.6%
fp32_to_bf16_sr_global_offsets                       0.2565    6280       78.5%  elementwise SR with stateless RNG
```

### `--mode cute`

* `fp8_tensorwise_precalc_scale` (85.8%) — vectorized 128-bit copy atoms (`num_bits_per_copy` +
  `assumed_align=16`) hit DRAM speed-of-light.
* `mxfp8_swizzle` (78.7%) — one e8m0 scale per 1×32 block, scattered to the swizzled 4D
  `(nrb, ncb, 32, 16)` scale grid. The **warp-per-row ("wpr") mapping** (ported from
  `_nvfp4_swizzle_kernel`) is what lifts it from the old 1-D-flatten kernel's ~67% to match triton
  (77.5%): warp `w` owns row `bidy*WARPS+w`; its 32 lanes + a `grid.x` column split (XSPLIT) + ILP
  stripe that row's N/32 blocks, all 128-bit vectorized loads issued first for memory-level
  parallelism. ncu on the old kernel showed it was **ALU-pipe bound (~68%), not DRAM bound**: a 1-D
  flatten recomputed the full 4D swizzle offset (a 6-op div/mod chain, `_swizzle_flat`) *per block
  per thread*. Because wpr fixes the row per warp, the row-dependent term
  `row_base = ((row//128)*ncb*32 + (r128%32))*16 + (r128//32)*4` is computed **once per warp** and
  amortized over every block the lane visits — only the cheap `+(gc//4)*512 + gc%4` remains per
  block. Tuned WARPS=2, XSPLIT=4, ILP=4. Bit-exact vs gold; edges triton at peak. (This is the same
  wpr + hoisted-swizzle recipe that fixed `nvfp4_swizzle` below.)
* `fp8_deepseek_1x128` (72.2%) — the vectorized-copy recipe applied to 1×128 blocks: flatten to 1-D,
  128 threads/CTA, VPT=32 with 128-bit vectorized load/store. A 1×128 block spans 4 contiguous
  threads (4×32=128), so the per-thread abs-max is combined across the group with
  `warp_reduction_max(threads_in_group=4)` and the group leader scatters the fp32 scale. The
  per-block reduction forces the full 32-wide f32 vector live (48 reg/thread → ~48% occupancy, vs
  the 29 reg / 80% occ of the pure-elementwise tensorwise), which caps it near the triton parity
  (75.7%); DRAM is ~76%, near the practical ceiling.
* `mxfp8_dim_m` (60.3%) — warp-specialized **TMA**: TMA-load a (64,256) tile, reduce 32-row
  blocks per column to the e8m0 scale, quantize, transpose in the register→smem write, and
  TMA-store the (256,64) tile to the row-major (N,M) output. Beats the triton kernel (60.1%) and
  approaches the CUDA SOL (67.7%). See [`quant_cast_cute/recipes.py`](../quant_cast_bench/quant_cast_cute/recipes.py).
* `mxfp8_dim_m_swizzle` (72.3%) — identical to `mxfp8_dim_m` (same TMA load → per-column
  e8m0 reduce → transposed register→smem write → TMA store), except the e8m0 byte is scattered
  into the NVIDIA-swizzled 4D `(nrb, ncb, 32, 16)` scale grid (acting on the transposed-frame scale
  `(N, M//32)`) instead of the plain 2D buffer, using the same flatten as `mxfp8_swizzle`.
  **Surprisingly it's ~10 pts *faster* than the plain kernel (62.1%), not equal**: the qdata TMA
  pipeline is byte-identical, so the only difference is the scale store — and the plain `(N, M//32)`
  store is a strided scatter (adjacent columns/threads write `M//32`-apart addresses), whereas the
  swizzled write packs neighboring rows/blocks into compact, coalesced offsets, relieving the scale
  store rather than costing anything. The swizzle row-part (`row_base`) is loop-invariant per thread,
  so it's hoisted out of the 32-row-block loop. Also beats triton (71.8%).
* `fp8_deepseek_1x128_dim_m` (61.7%) — the same TMA path as `mxfp8_dim_m`, with a 128-row
  block (not 32) and an fp32 `amax/448` scale (not an e8m0 byte). TMA-load a (128,128) tile, each
  thread scans its 128-row column for the amax in four 32-wide chunks (vector reduce, only 32 f32
  live → low registers), then re-reads to quantize and write the transposed contiguous run into
  sOUT for the TMA store. Tile is (128,128)/4 warps: a (128,256) tile needs 96 KB smem → only 2
  CTAs/SM, which halved bandwidth (38.7%); dropping to (128,128) restores 48 KB/4 CTAs → 61.7%,
  matching `mxfp8_dim_m`'s footprint. Beats the triton dim-M kernel and approaches its
  compile-mode SOL. (Replaces the old scalar `x.t()` path at ~7%.)
* `fp8_deepseek_1x128_dim_km` (41.2%) — the one-pass both-directions deepseek cast (dim-K `qk (M,N)`
  + `sk (M,N//128)`; dim-M `qm (N,M)` + `sm (N,M//128)`, transposed), the fp32-scale/128-block analog
  of `mxfp8_dim_km`. Same fused TMA template: TMA-load one (128,128) tile, reduce both ways —
  dim-M writes the transposed run into sOUT for a TMA store; dim-K keeps `x`'s layout so it quantizes
  in-register and stores each 32-chunk **directly to gmem** with a 128-bit copy. Beats compile
  (35.8%) but sits **below the 1×32 sibling (57.4%) and triton (57.8%)**, for two reasons ncu makes
  clear (both intrinsic to the 128-block): (1) the dim-K row reads are **32-way** bank-conflicted (vs
  16-way at 1×32) — a 128-col bf16 block is bank-aligned, so a thread-per-block read has every lane on
  the same bank regardless of tile/mapping; (2) doing both 128-reductions per thread needs 154
  reg/thread → 3 CTAs/SM. Occupancy is *not* the lever, though (L1/TEX ~86%): warp-splitting the two
  directions (one dir/thread, ~72 reg, 2× occupancy) made it **worse** (35%), as did non-unrolled
  chunk loops (30%) — both cost ILP. The real fix for the conflict is a **swizzled sIN smem layout**
  (as for the 1×32 sibling), left as future work.
* `fp8_deepseek_128x128` (70.1%) — non-transposing **TMA** with a block-wide reduction (one fp32
  scale per whole 128×128 block). One CTA/block: TMA-load the (128,128) tile to smem, each thread
  reduces its share to a local amax, then a warp-reduce + smem block-reduce gives the block amax;
  re-read, quantize, TMA-store the (128,128) tile (no transpose). The **crux is the smem access
  pattern**: giving each thread a *contiguous* run puts all 32 warp lanes on one bank (32-way
  conflict → 4.3%, worse than the old kernel); switching to a *strided* assignment (thread `t` owns
  `{t + i·THREADS}`) so consecutive lanes hit consecutive banks lifts it to 70.1%. 128 threads/CTA
  beat 256/512 (higher VPT → more memory-level parallelism per thread).
* `mxfp8_32x32` (70.8%) — one e8m0 scale per 32×32 block; non-transposing, so the same
  TMA path as `fp8_deepseek_128x128`. A 32×32 block = 1024 elems = a full warp (32 lanes × 32 rows),
  so **one warp owns one block** and the block amax is a single `warp_reduction_max` — no cross-warp
  scratch. A 128×128 TMA tile holds 16 blocks; 8 warps each loop over 2 of them (lane `l` owns
  column `l` → consecutive lanes hit consecutive smem banks). Two findings each moved it ~38%→70%:
  (1) `v / sfp` on the 32-wide vector emits 32 per-element **divisions** (168M insts, 38%) — since
  the e8m0 scale is a power of two, `inv = 1/sfp; v * inv` is bit-exact and cuts to 95M insts (matches
  deepseek); (2) using 8 warps/CTA (not 4) doubles resident warps at the same smem-capped 4 CTAs/SM,
  hiding the TMA-load latency the kernel is bound on. A direct coalesced fp8 *global* store (drop the
  sOUT smem to raise occupancy) was tried but is worse (50.5%) — the whole-tile TMA store beats
  scattered 32-byte fp8 sectors. Matches the deepseek_128x128 sibling's ~70% ceiling (triton 76.5%).
* `fp8_rowwise` (79.7%) — one fp32 scale per row, amax over the whole row. The naive kernel held the
  whole row live (one 512-thread CTA/row), which pinned it at 58 reg/thread → 43% occupancy → 60%
  DRAM (3.0% of peak here since it was also unvectorized). The fix mirrors triton/inductor: a small
  256-thread CTA per row that **loops over the row in BN=4096 blocks** (VPT=16, 128-bit vectorized
  ld/st) accumulating a per-thread abs-max (only 16 elems live/iter), warp+smem block-reduce for the
  row amax, then a second loop re-reads each block — warm in L2 from pass 1, like triton's
  `evict_last`/`evict_first` hints — to quantize and store. Registers drop to 32, occupancy to 90%,
  DRAM to 82.6%. Beats triton (76.7%) and matches compile (79.0%); the extra L2 read pass costs
  little at this occupancy. (Replaces the old one-warp-per-row scalar kernel at 3.0%.)
* `fp8_colwise` (46.0%) — one fp32 scale per column, amax over all rows, transposed (N,M) output.
  The reduction is *down* a column (the strided axis of row-major x), so a naive kernel is forced
  into uncoalesced reads (1.6%). Mirror triton's two coalesced passes but drive both with **TMA**:
  pass 1 TMA-loads (128,256) tiles, each thread reduces its column's rows in smem to a partial amax,
  then `atomic_max_float32` into a (N,) scratch (combining across the M-grid); pass 2 TMA-loads
  (64,256) tiles, quantizes each column with the precomputed `amax/448` scale, transposes in the
  register→smem write (like `mxfp8_dim_m`), and TMA-stores the (256,64) tile. The TMA engine
  streams the strided tiles at DRAM speed — a hand-rolled strided row-segment read of x caps the amax
  pass at ~42% (152 µs) vs TMA's ~67% (93 µs). Beats triton (43.8%) and compile (25.6%). The ~51%
  ceiling is structural: a full-column amax forces reading x *twice* and, unlike rowwise, the quant
  re-read misses L2 (a full column is 32 KB·M, far larger than L2; the whole 512 MB streams between
  the two kernels). L2-panel tiling doesn't help — separate TMA kernels don't retain the panel, and
  per-CTA reuse needs <6 concurrent CTAs. (Replaces the old scalar `x.t()` path at 1.6%.)

* `nvfp4_swizzle` (~62%) — the two-level nvfp4 cast (per-tensor outer scale × per-16-block e4m3
  inner scale, fp4-packed qdata, e4m3 scale scattered to the swizzled 4D grid), modeled on the
  human-optimized torchao fp4 CuTeDSL cast (pytorch/ao#4517). The unit of work is a "group" of 32
  elems = two 1×16 blocks = one 128-bit fp4 store. The old naive kernel (8 thr/CTA, scalar
  per-element loads) was 7.9%. ncu shows it's **ALU-pipe bound** (~73%), not DRAM bound, and three
  fixes lifted it: (1) hardware inline-PTX cvts — `cvt.rn.satfinite.e2m1x2.f32` packs 8 f32 → 4 fp4
  bytes/call (one `mov.b32`, no per-byte masking) and `cvt e4m3x2` / `f16x2.e4m3x2` do the two-level
  scale as single instructions, vs the 4-lane-broadcast e4m3 fragments + `_maybe_recast_from_f4`;
  (2) hoisting the swizzle offset — the 4D flatten factors as `row_base + (col//4)*512 + (col%4)`
  (the per-row div/mod chain was the dominant ALU term); (3) a **warp-per-row** mapping (ao#4517's
  "wpr") — warp `w` owns row `bidy*WARPS+w`; its 32 lanes + a `grid.x` column split + ILP stripe
  that row's groups, all loads issued first for MLP. Because the row is fixed per warp, `row_base`
  is computed *once* and amortized over every group the lane visits (vs 2 in a 1-D-flatten mapping),
  and a whole row's scale bytes land in one 128-row swizzle atom. This beat a 1-D striped mapping
  (~58%, long-scoreboard-bound at 44% occ). Tuned WARPS=2, XSPLIT=4, ILP=4. Beats compile (23.5%)
  and edges the repo's triton (62.7%) at peak; the identical ao#4517 kernel on this same swizzle
  layout measures 58.5% (its striped mapping) / 63.1% (its wpr) here. (The swizzle scale-scatter is
  the ceiling: the *linear* scale layout hits ~70% on the same kernel, but our recipe needs the
  blocked swizzle.) (`nvfp4_blocked_outer` keeps the naive kernel — it wasn't the target.)

* `mxfp8_dim_km` (57.4%) — the one-pass both-directions mxfp8 cast: read `x` once and
  emit four outputs, dim-K (`qk (M,N)` + `sk (M,N//32)`, 1×32 blocks along columns) and dim-M
  (`qm (N,M)` + `sm (N,M//32)`, 32×1 blocks down rows, transposed). The **fused TMA BM×BN template**
  (mirrors `mxfp8_dim_m`): TMA-load one (64,256) row-major tile into smem, read it once and
  reduce BOTH ways, then emit the two quantized tiles. dim-M (the binding half): each of the
  TM·TN/32 (col, 32-row-block) groups reads its 32 rows *down* a column, e8m0-scales, quantizes, and
  writes the run into an sOUT laid out (TN,TM) — the transpose is the register→smem write — for a TMA
  store to the `(N,M)` output. dim-K rides the loaded tile nearly for free: each (row, 32-col-block)
  group reads its 32 *along* a row, and since `qk` keeps `x`'s layout it quantizes in-register and
  stores the contiguous 32-run **directly to gmem with a 128-bit vectorized copy** (adjacent threads
  = adjacent col-blocks → coalesced). Went **18.9% → 57.4%** (naive 32×32/1-warp kernel → this),
  beating triton (47.1%) and nearing the standalone `mxfp8_dim_m` (60.3%). Keeping `qk` out of
  smem was worth +5 pts alone (52% → 57%): it frees 16 KB → +1 CTA/SM (occupancy 37.5% → 50%) and
  drops the dim-K transpose-store bank conflicts. **The remaining ceiling is L1/TEX (ncu ~82%)**: the
  dim-K row reads are ≥16-way bank-conflicted because a 32-col bf16 block is exactly 16 banks wide,
  so thread-per-block reads collapse to 2 bank groups regardless of tile/mapping (bank depends only
  on column when TN is a multiple of 64). Killing that needs a **swizzled smem layout** for the input
  tile (XOR swizzle, as CUTLASS GEMM uses) that *also* keeps the dim-M column reads conflict-free —
  the real next step, left as future work.

* `mxfp8_dim_km_swizzle` (62.1%) — same one-pass both-directions cast as `mxfp8_dim_km`
  (byte-identical TMA load, dim-M transposed store, dim-K direct-to-gmem store), except **both** e8m0
  scales are scattered into the swizzled 4D `(nrb, ncb, 32, 16)` grid: `sk (M, N//32)` (pre-swizzle
  row = m, col = 32-col-block over N//32) and the transposed `sm (N, M//32)` (pre-swizzle row = n,
  col = 32-row-block over M//32). Like `mxfp8_dim_m_swizzle`, the swizzle is *faster* than the
  plain kernel (57.4% → 62.1%, +4.7 pts) — the qdata paths are unchanged, so the win is entirely the
  two scale stores: the plain `(M, N//32)` / `(N, M//32)` writes are strided scatters (adjacent
  threads land `N//32` / `M//32` apart), whereas the swizzled writes pack neighboring rows/blocks into
  compact, coalesced offsets. The gain is larger here than for `dim_m_swizzle` because there are *two*
  such scatters to relieve. The swizzle offset is recomputed per group (unlike `dim_m_swizzle`, the
  row isn't loop-invariant here). Also edges triton (45.7%).

* `bf16_rht` (68.2%) — the 16×16 randomized Hadamard transform, run on **tensor cores**. A scalar
  fp32 dot-product cute kernel is *compute*-bound at ~36% (256 fp32 MACs/group saturate the CUDA
  cores before DRAM), and torch.compile's cuBLAS GEMM stalls at 29.3% (a skinny K=N=16 GEMM tiles
  terribly). The fix is the SM80 warp-level bf16 `mma.sync` atom (m16n8k16): flatten `x` to groups
  of 16 and run the transform as a batched `D[m,n] = Σ_k A[m,k]·B[n,k]` with A = a 16-group×16 tile
  and B = `rht`ᵀ, so K=16 is a single k-step — the CuTeDSL analog of the Triton `tl.dot` kernel.
  Global↔smem transfers are coalesced 128-bit vectorized copies; smem↔MMA-register fragments go
  through the tiled-MMA partition; `rht` is staged in smem once per block (transposed on the fly, so
  the wrapper passes it row-major with no runtime transpose). Tuned WARPS=4 × 2 tiles/warp (the
  per-warp tile loop gives the memory-level parallelism that lifts a 1-tile/warp version 56% → 68%).
  Beats compile (29.3%) and matches the triton kernel (67.7%), near the ~75% relu ceiling. Bit-exact
  vs the torch reference: bf16×bf16 is exact in fp32, so the tensor-core fp32 accumulation reproduces
  torch's bf16 matmul (the cute test compares bf16 outputs to ~1 ULP, i.e. demands exactness).

* `fp32_to_bf16_sr` (83.5%) — stochastic-rounding fp32→bf16, a pure elementwise streaming cast (read
  fp32, write bf16), so it takes the same DRAM-speed-of-light recipe as `fp8_tensorwise`: flatten to
  1-D, 256 threads/CTA, VPT=8 with 128-bit vectorized load/store (8 fp32 = 2×128b in, 8 bf16 = 1×128b
  out; VPT=4 drops to 70.5% — the 64-bit store loses the vectorization). CuTeDSL exposes no
  counter-based PRNG intrinsic (unlike Triton's `tl.randint4x`), so the dither comes from a
  **hand-written Philox-4×32-10** (`_philox_4x32`) — the same generator torch/triton use, built out of
  the DSL's integer ops (the mulhilo step widens to `Uint64`; verified bit-exact vs the Random123
  reference). It's keyed like the global SR kernel (counter = flat index // 4, one call dithers 4
  consecutive elements from its 4 outputs' top 16 bits). The test only checks the SR *property*
  (unbiased, lands on the two bracketing bf16 grid points — mean error ~1e-5 vs the 1e-3 tolerance),
  not a bit-match. This is the **fastest SR path** — beats triton (72.3%, `tl.randint4x` in-register)
  and is ~2.8× compile (29.3%, which wastes a DRAM round-trip materializing the uniforms; see Known
  issues). The 10-round Philox mix costs ~5 pts vs a cheap MurmurHash3 fmix32 dither (88.0%), which
  also passes — the extra ALU of a full Philox is the price of matching the standard generator.
  fp32-in/bf16-out is 6 bytes/element, so the 83.5% isn't directly comparable to the bf16 relu ceiling.

* `fp32_to_bf16_sr_global_offsets` (82.9%) — the tiling-invariant SR counterpart. In the gold/triton
  pair the two SR recipes differ in RNG keying: the plain one keys the dither on the *tile-local*
  element order (so tiling changes the rounding — the deliberate counterexample), the global one keys
  on each element's *global* flat index (tile-invariant). But the CuTeDSL SR kernel is built on
  `cute.make_identity_tensor` global coordinates, so it **already** keys Philox on the global flat
  index (counter = flat index // 4, stream = flat index % 4), independent of the tile shape. That is
  exactly the "global offsets" scheme, so this recipe reuses the same kernel — there is no separate
  tile-local cute kernel to contrast against, and the ±0.6-pt gap from `fp32_to_bf16_sr` is run-to-run
  variance on identical code.

```
shape: (16384, 16384)  mode: cute
versions: torch 2.14.0.dev20260720+cu130, helion 1.2.0, cutlass-dsl 4.5.2
recipe                            gpu_time_ms    gbps    pct_peak  perf_description
------------------------------  -------------  ------  ----------  --------------------------------------------------------
relu (baseline)                        0.1772  6059.2       75.7%
fp8_tensorwise_precalc_scale           0.1169  6887.7       86.1%  elementwise
mxfp8_swizzle                          0.1298  6267.6       78.3%  (1,32) block, swizzle
fp8_deepseek_1x128                     0.1334  6101.3       76.3%  (1,128) block
mxfp8_dim_m                            0.1568  5188.5       64.9%  (32,1) block, t-contig
mxfp8_dim_m_swizzle                     0.136    5981       74.8%  (32,1) block, t-contig, swizzle
fp8_deepseek_1x128_dim_m               0.1561  5211.5       65.1%  (128,1) block, t-contig
mxfp8_dim_km                           0.2483  4392.3       54.9%  (1,32) dim-k + (32,1) dim-m, one pass, t-contig
mxfp8_dim_km_swizzle                   0.2137  5103.1       63.8%  (1,32) dim-k + (32,1) dim-m, one pass, t-contig, swizzle
fp8_deepseek_1x128_dim_km              0.3279  3326.1       41.6%  (1,128) dim-k + (128,1) dim-m, one pass, t-contig
mxfp8_32x32                            0.1403  5740.8       71.8%  (32,32) block
fp8_deepseek_128x128                   0.1414    5697       71.2%  (128,128) block
fp8_rowwise                            0.1263  6375.1       79.7%  (1,-1) block
fp8_colwise                            0.2196  3667.4       45.8%  (-1,1) block, t-contig
nvfp4_swizzle                          0.1407  4888.7       61.1%  (1,16) block, fp4 qdata, swizzle
bf16_rht                               0.1963  5469.5       68.4%  elementwise RHT
fp32_to_bf16_sr                        0.2372  6790.8       84.9%
fp32_to_bf16_sr_global_offsets         0.2371  6791.8       84.9%  elementwise SR with stateless RNG
```

### `--mode flex_tile_map_triton`

The `flex_tile_map` TRITON_TEMPLATE backend: the user's plain-PyTorch `f` is traced, walked by
`FxTritonEmitter`, and spliced onto a hand-written Triton template (one per reduction shape). Only
the group-reduction recipes are wired (a pointwise `f` has no template lowering — it goes through
`--mode compile`). Each row here is generated by the *same* emitter + template machinery, so it's
the "generic backend" number to compare against the bespoke `--mode triton` kernels above:

```
shape: (16384, 16384)  mode: flex_tile_map_triton
versions: torch 2.14.0.dev20260720+cu130, helion 1.2.0, cutlass-dsl 4.5.2
recipe                      gpu_time_ms    gbps    pct_peak  perf_description
------------------------  -------------  ------  ----------  -----------------------------------
relu (baseline)                  0.1773  6057.6       75.7%
fp8_deepseek_1x128_dim_m         0.1461  5568.5       69.6%  (128,1) block, t-contig
mxfp8_dim_m                      0.1968  4133.6       51.7%  (32,1) block, t-contig
mxfp8_32x32                      0.1266  6360.6       79.5%  (32,32) block
nvfp4                            0.1333  5161.9       64.5%  (1,16) block, fp4 qdata, no swizzle
```

* `mxfp8_32x32` (76.6%) — the block_2d template (`template_mxfp8_32x32.py.jinja`): the
  traced `f` splits both dims into 32×32 blocks (a rank-4 reshape + a `permute` swapping the two
  middle axes), flattens each block to 1024 elements, reduces the whole-block amax to an e8m0 scale,
  then un-blocks the fp8 qdata back to `(M, N)` — no transpose. It matches the bespoke `--mode
  triton` kernel (77.0%) and beats the CuTeDSL kernel (70.4%): a non-transposing block cast is
  exactly what Triton's default blocking handles well, so the emitted 4D `tl.reshape`/`tl.trans`
  register shuffles cost nothing over a straight elementwise store.
* The dim-M rows (`fp8_deepseek_1x128_dim_m` 68.3%, `mxfp8_dim_m` 50.1%) and `nvfp4` (61.7%)
  are in the same ballpark as their bespoke `--mode triton` counterparts (70.1% / 56.9% / 64.5%):
  the emitter reproduces the hand kernels' structure and the template autotunes its own block sizes,
  so the residual gaps are config/autotune spread rather than a fundamental backend penalty.

### `--mode helion`

```
shape: (16384, 16384)  mode: helion
versions: torch 2.14.0.dev20260720+cu130, helion 1.2.0, cutlass-dsl 4.5.2
recipe                          gpu_time_ms    gbps    pct_peak  perf_description
----------------------------  -------------  ------  ----------  --------------------------------------------------------
relu (baseline)                      0.1773  6057.5       75.7%
fp8_tensorwise_precalc_scale         0.1168  6896.8       86.2%  elementwise
fp8_deepseek_1x128                   0.1171  6950.2       86.9%  (1,128) block
mxfp8_swizzle                        0.1347  6038.7       75.5%  (1,32) block, swizzle
fp8_deepseek_1x128_dim_m             0.1513  5379.5       67.2%  (128,1) block, t-contig
mxfp8_dim_m                          0.2092  3889.1       48.6%  (32,1) block, t-contig
mxfp8_dim_m_swizzle                  0.2139  3803.9       47.5%  (32,1) block, t-contig, swizzle
mxfp8_dim_km                         0.5175  2107.4       26.3%  (1,32) dim-k + (32,1) dim-m, one pass, t-contig
mxfp8_dim_km_swizzle                  0.383  2847.4       35.6%  (1,32) dim-k + (32,1) dim-m, one pass, t-contig, swizzle
fp8_deepseek_1x128_dim_km              0.29  3760.7       47.0%  (1,128) dim-k + (128,1) dim-m, one pass, t-contig
mxfp8_32x32                          0.1895  4251.6       53.1%  (32,32) block
fp8_deepseek_128x128                 0.1331    6051       75.6%  (128,128) block
nvfp4                                 0.481  1430.1       17.9%  (1,16) block, fp4 qdata, no swizzle
nvfp4_swizzle                        0.3806  1807.2       22.6%  (1,16) block, fp4 qdata, swizzle
bf16_rht                             0.1571  6832.6       85.4%  elementwise RHT
fp32_to_bf16_sr                      0.2429  6631.7       82.9%
```

* `fp8_tensorwise_precalc_scale` (85.3%, above the relu ceiling), `fp8_deepseek_1x128` (87.3%) and
  `mxfp8_swizzle` (79.4%) are the **three non-transposing casts optimized per the full
  process** (correctness with `autotune_effort="none"`, then a `autotune_effort="full"` search at
  16384², then the winning config hardcoded), so unlike the correctness-first rows below these are
  actually tuned — all three land at or above their bespoke triton/cute siblings. `fp8_tensorwise`
  (elementwise, flatten to 1-D) autotuned to a wide `block_sizes=[8192]` with a `tensor_descriptor`
  load, matching cute (86.1%) and beating triton/compile (~70.8/70.5%). `fp8_deepseek_1x128` (1×128
  block reduced in-kernel, no transpose) autotuned to `block_sizes=[64, 2]` with a persistent
  `reduction_loops=[None]`; the full search needed `HELION_AUTOTUNE_IGNORE_ERRORS=1` to *prune* (not
  abort on) one candidate that miscompiled (`flatten_loops` + a reduction over a flattened
  accumulator), and the result beats triton/cute/compile (75.8/76.7/77.2%).
* `mxfp8_swizzle` (79.4%) is the one of the three where **autotune could not find a config**:
  the default `[32, 32]` config overflows triton's 1,048,576-element per-tensor cap while computing
  the autotune baseline (this kernel tiles the block-*count* dims, so the per-program tile numel is
  `block₀·block₁·16384`), so a custom `autotune_baseline_fn` was supplied — after which the search
  still returned `NoConfigFound` because the space is dominated by those same overflowing configs. Per
  the task's fallback clause it was pinned to a manual `block_sizes=[1, 1]` (one 128×128 block/program),
  which is bit-exact and still reaches 79.4% — matching triton (81.5%) and edging cute (76.5%). This is
  the same block-count-dim tiling constraint the correctness-first swizzle rows below run into.
* `fp8_deepseek_1x128_dim_m` (66.8%, i.e. ~89% of this run's relu ceiling) and `mxfp8_dim_m`
  (55.2%) both do the dim-M reduction with an **in-kernel transposed store** — view the input as
  `(rb, group, N)` and the `(N, M)` output as `(N, rb, group)`, then store `y.permute(...)` inside
  the kernel (Helion lowers the register permute like `tl.trans`), which is ~4× faster than writing
  the natural frame and transposing in the wrapper. Both use hand-pinned configs (deepseek keeps the
  128-row reduction persistent via `reduction_loops=[None]` so `x` is loaded once, not 3×; the
  autotuner's timing was too noisy to trust here). They land in the same ballpark as their bespoke
  `--mode triton` counterparts once the depressed baseline is accounted for.
* `mxfp8_dim_m_swizzle` (36.7%) is `mxfp8_dim_m` with the e8m0 scale scattered **directly
  into the NVIDIA 32×4×4 swizzled block grid in-kernel** (not stored plain then swizzled in a
  wrapper). To make the swizzled store a plain index, it tiles over the *block-count* dims so the
  tile indices are the block ordinals, which forces the within-block 32/128 axes into 5D register
  tiles. That structure can't use `autotune_effort="none"` — the default block-size heuristic would
  blow the register tile past triton's 1,048,576-element per-tensor cap at this shape — so it's
  pinned to `block_sizes=[1, 1]` (one 128×128 block/program). That correctness-first tile is why the
  bandwidth trails the plain dim-M kernel and the triton/cute swizzle versions (72.7% / 73.9%);
  raising the block sizes is the perf follow-up.
* `mxfp8_dim_km` (24.9%) does **both** mxfp8 reductions in one pass over `x` and emits four
  outputs — the dim-K pair `(M, N)` / `(M, N//32)` in the natural frame *and* the dim-M pair `(N, M)` /
  `(N, M//32)` transposed. It reuses the 32×32 block-grid view `(rb, 32, cb, 32)` and takes both
  reductions off the one loaded block (reduce the trailing 32 for dim-K, the leading 32 for dim-M),
  so like `mxfp8_32x32` it's pinned to `block_sizes=[1, 1]`. `autotune_effort="none"` is *not*
  usable here (tested): the default heuristic scales the register tile by `block²·32·32`, which didn't
  compile at 512² in 10 min and would overflow the 1,048,576 per-tensor cap at this shape. The low
  bandwidth is expected — it's one read of `x` but ~2× the output writes (four tensors) on the
  correctness-first tile.
* `mxfp8_dim_km_swizzle` (27.1%) is `mxfp8_dim_km` with **both** e8m0 scales scattered
  directly into the NVIDIA 32×4×4 swizzled block grid in-kernel (qdata is byte-identical). To make
  both swizzled stores plain 5D indices it tiles over the 128×128 block grid and views `x` with both
  within-128 axes fully split (M-rows into `(c4, w32)`, N-cols into `(a, b)`), so one 6D loaded block
  feeds both reductions and both scale stores via pure permutes — no in-kernel reshape across a tiled
  axis (unlike the dim-M-only swizzle kernel). Same `block_sizes=[1, 1]` pin / autotune-none overflow
  as the other dim-km kernels. It's actually a hair *faster* than the plain `mxfp8_dim_km`
  (24.9%), consistent with the swizzled scale store beating the plain strided one seen elsewhere.
* `fp8_deepseek_1x128_dim_km` (46.9%) is the deepseek analog — same one-pass four-output structure as
  `mxfp8_dim_km` but 128×128 blocks with an fp32 `amax/448` reciprocal scale instead of e8m0
  bit-math. It's ~2× the bandwidth of the mxfp8 dim-km (24.9%) because 128-blocks write 4× fewer scale
  values than 32-blocks (and in larger contiguous chunks). Pinned `block_sizes=[1, 1]` for the same
  reason as the others.
* `mxfp8_32x32` (52.2%) is deliberately pinned to a **tiny `block_sizes=[1, 1]`** (one 32×32
  block per program). The kernel views `x` as 4D `(rb, 32, cb, 32)` and tiles only `[rb, cb]`, so the
  block sizes multiply the untiled 32×32 register tile — the default heuristic's `[16, 16]`
  materializes a 1 MB fp32 tile per program that takes ptxas ~18 s to compile cold. `[1, 1]` keeps it
  at 4 KB → ~0.5 s cold compile. That fast-debug tile is the reason the bandwidth is low here (raise
  `block_sizes` or switch to `autotune_effort="full"` if this kernel's perf becomes the point).
* `fp8_deepseek_128x128` (76.3%) is the deepseek square-block analog of `mxfp8_32x32` — same
  in-place `(rb, 128, cb, 128)` block view but 128×128 blocks with an fp32 `amax/448` reciprocal scale.
  It sits right at the relu ceiling (74.9%, and edges past it run-to-run): 128×128 blocks write the
  fewest scale values of any recipe here (one fp32 per 16384 elements) and there's no transpose, so
  it's essentially pinned to the input-read + qdata-write bandwidth floor even on the same
  correctness-first `block_sizes=[1, 1]`.
* `nvfp4` (26.1%) is the two-level cast (per-tensor outer scale × per-16-block e4m3 inner scale,
  fp4-packed qdata) on Helion's default `autotune_effort="none"` config. The fp4 encode+pack uses the
  **hardware `cvt.rn.satfinite.e2m1x2.f32` PTX**, emitted via `hl.inline_asm_elementwise` (Helion's
  own inline-asm HOP — the gold's Inductor `inline_asm_...` is unreachable from a Helion kernel, and
  the native `.to(torch.float4_e2m1fn_x2)` cast mis-lowers). Bit-exact vs gold but un-tuned; the low
  bandwidth is the default config plus the extra even/odd split work, not the encode itself.
* `nvfp4_swizzle` (37.4%) is `nvfp4` with the per-16 e4m3 inner scale scattered **directly into the
  NVIDIA 32×4×4 swizzled block grid in-kernel** (qdata byte-identical). Like the mxfp8 swizzle kernels
  it tiles over the block-count dims and views `x` with the within-128/64 axes fully split (a 7D block),
  so the fp4 encode, the even/odd pack, and the swizzled scale store are all pure permutes on one loaded
  block. It's ~1.4× the plain `nvfp4` (26.1%) — the swizzled scale store beats the plain strided one,
  same effect seen in the mxfp8 dim-km pair. Pinned `block_sizes=[1, 1]`; `autotune_effort="none"` is
  *not* usable here for two reasons (the inline-asm autotuner crash `nvfp4` also hits, and the swizzle
  tile-shape overflow the other swizzle kernels hit).
* `bf16_rht` (85.2%, i.e. above this run's relu ceiling) is the 16×16 randomized Hadamard transform —
  bf16 in/out, no scale. It flattens `(M, N)` to `(n_groups, 16)` and gives each program a
  `(BLOCK_G, 16)` tile, so it's a batch of `(BLOCK_G, 16) @ (16, 16)` matmuls; the matmul is done in
  fp32 (upcasting the bf16 inputs is exact, so fp32 accumulation reproduces torch's bf16 gemm) and cast
  back. It's bandwidth-bound (read + write bf16, the K=16 dot is tiny). `autotune_effort="none"` picks
  too small a block (`block_sizes=[32]`, ~40% peak); a pinned wider block matching the triton kernel's
  `BLOCK_G=512` lifts it to the top of the table (blocks ≥1024 groups overflow tensor memory since the
  matmul lowers to `tl.dot`/tmem).
* `fp32_to_bf16_sr` (81.9%, above this run's relu ceiling) is the stochastic-rounding fp32→bf16 dither
  (add a uniform 16-bit value to the mantissa, then truncate). It never materializes a random tensor
  the way `compile` mode does, and it's bit-unrelated to the torch reference (only the SR *property* —
  unbiased, lands on the two bracketing bf16 grid points — is checked). The dither is drawn by calling
  **`tl.randint4x` directly through `hl.inline_triton`** (Helion's raw-Triton HOP) rather than
  `hl.rand`: `hl.rand` runs a full 10-round Philox per element and throws away 3 of every 4 outputs,
  which capped this kernel at ~4.3%. Instead the input is viewed as `(n//4, 4)` and one Philox round
  dithers 4 elements (its four blocks scattered across the size-4 minor axis via a one-hot sum, since
  Helion rejects strided column indexing). `tl.randint4x` returns `uint32`, so each block is
  `bitcast` to int32. Because `inline_triton` is a raw-Triton HOP that aborts `autotune_effort="full"`,
  the config (`block_sizes=[1024]`, `num_warps=8`, `num_stages=7`) was found under
  `autotune_effort="none"` and hand-pinned. This lands at/above the bespoke triton (73.0%) and cute
  (84.8%) SR kernels.

## Known issues

* `fp32_to_bf16_sr` (compile) reports only ~29.3% peak, but this understates the real bandwidth.
  The stochastic-rounding uniform is drawn via `torch.func._random.uniform` → `aten._philox_uniform`,
  which inductor treats as an opaque extern op rather than a fusible in-kernel RNG. So it runs as
  two DRAM passes: kernel 1 materializes a full-size fp32 random tensor (~1.07 GB write, ~63% of
  the runtime), kernel 2 reads it back alongside `x` to dither+truncate. Real traffic is ~3.76 GB
  (write u + read x + read u + write out) ≈ 46% of peak; the benchmark only counts input+output
  (~1.61 GB), so the wasted RNG round-trip shows up as the low 29.3%. Fix: fuse the Philox RNG into
  the dither kernel (generate uniforms in-register, never materialize) — as inductor already does
  for `torch.rand`/dropout — which would cut traffic to ~1.61 GB and approach the relu ceiling
  (~2–3× speedup). **The hand-written Triton kernel does exactly this and confirms the prediction:
  `tl.randint4x` generates the Philox uniforms in-register (never materializing `u`), so it moves
  ~1.61 GB in one pass and hits 72.2% — ~2.5× the compile mode's 29.3% and near the relu ceiling.**
  The CuTeDSL kernel goes further (83.5%): the same one-pass, in-register dither with 128-bit
  vectorized fp32-load/bf16-store, using a hand-written Philox-4×32-10 (CuTeDSL has no PRNG intrinsic,
  so it's built from the DSL's integer ops — bit-exact vs the Random123 reference).

* `bf16_rht` (compile) runs at only ~29% peak, and here the traffic is not wasted (the whole 1.07 GB
  is useful read x + write out) — it's GEMM-kernel inefficiency. The 16×16 RHT `x.reshape(..., 16) @ rht`
  is lowered to a single cuBLAS GEMM via `extern_kernels.mm`, shape `(M·N/16, 16) @ (16, 16)` — i.e.
  `K=16, N=16`. That GEMM is ~99.5% of the runtime. The op is really memory-bound (~4 flop/byte), but
  cuBLAS runs it as a compute-oriented matmul, and the skinny `K=N=16` shape tiles terribly (N-tiling
  wasted, no K-reuse to amortize), so it fails to saturate DRAM — 29% vs the ~75% relu ceiling for the
  same 1.07 GB (~2.6× slower than bandwidth-bound). Fix direction: a fused kernel that loads a 16-vector,
  applies the transform in registers, and writes 16 (or a Triton matmul template tuned for the skinny
  shape) would approach the relu ceiling. **The hand-written Triton kernel confirms this and hits 67.7%**
  (~2.3× the compile 29.3%, near the 74.9% relu ceiling): it flattens `x` to `(n_groups, 16)` and runs a
  batch of `(BLOCK_G, 16) @ (16, 16)` `tl.dot`s (fp32 accum → bf16) in one pass. The win comes from a
  large `BLOCK_G=512` — each program does a `512×16` tile, amortizing the tiny `K=16` dot that cuBLAS
  can't tile well and turning the load/store into big coalesced runs. **The CuTeDSL kernel matches it
  at 68.2%** on the SM80 warp `mma.sync` atom (m16n8k16, K=16 in one step); notably a *scalar* fp32
  cute kernel is only ~36% (compute-bound on the CUDA cores) — the tensor cores are what make this
  memory-bound. Both `tl.dot` and `mma.sync` are bit-exact vs the reference (bf16×bf16 is exact in
  fp32, so the fp32 accumulation reproduces torch's bf16 matmul).

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

* `mxfp8_dim_km` (compile ~19.1%, triton ~47.1%) — the mxfp8 analog of the above (1×32
  dim-K + 32×1 dim-M, e8m0 scales). Same story under compile: inductor generates **3 kernels reading
  `x` 3×** (dim-K fused reduce+quantize; dim-M amax reduction; dim-M quantize+transpose). The
  hand-written Triton kernel does the single pass. A first fixed **32×32 tile** version only reached
  ~30.9% (the transposed dim-M store is only 32-wide, poorly coalesced, and each program does little
  work); switching to **blocked tiles** (autotuned `RB` 32-row blocks × `BN` cols, reshape per
  direction — the same lever that fixed `mxfp8_dim_m`) widens the dim-M store and raises
  occupancy, reaching **47.1%**. Still below deepseek's dim_km (57.8%) because the 32-block
  granularity means 4× as many e8m0 scales (M/32 vs M/128) plus the per-scale e8m0 bit-math, and the
  transposed dim-M store remains the binding cost.

* `fp32_to_bf16_sr_global_offsets` (compile) runs at only ~7.0% peak — ~4.2× slower (wall-clock) than
  `fp32_to_bf16_sr` (29.3%) for identical dithering math. The difference is how the Philox draw is
  keyed. Both use `torch.func._random.uniform` (experimental stateless Philox → unfused
  `aten._philox_uniform`, so both share the same materialized-`u` ~46%-real-BW ceiling). The plain
  variant keys on tile-LOCAL position (one shared key, counter = flat index within the call), which
  is cheap but changes with tiling. The global variant is tile-INVARIANT: it keys each draw on the
  element's GLOBAL index, which — because `uniform` only exposes a single scalar starting offset (the
  `(seed, offset)` key pair, fine for 1D/full-width tiles but not 2D sub-blocks) — forces
  materializing a per-element `(numel, 2)` uint64 key tensor = **4.29 GB** (16 B/element, 8× the
  0.54 GB bf16 output), written then read back by a batched Philox. That ~8.5 GB key round-trip
  roughly triples total traffic (~12.3 GB vs ~3.76 GB), the bulk of the slowdown.

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

**The hand-written Triton kernel realizes this "ideal" and hits 77.0% — the fastest SR variant in the
suite (vs the 7.0% compile mode, ~11×).** It's even simpler than the sketch above: because a standalone
Triton kernel is handed the *whole* tensor and owns its own blocking (rather than a flex_tile_map
sub-tile), an element's global index is just its flat position `f` in `x` — so there's no
`global_row`/`global_col`/`num_col` to thread through, and no `base`/`strides` needed. It keys Philox on
`counter = f >> 2` (via `tl.randint4x`, so one counter's 4 streams serve 4 consecutive elements),
computed in-register from `f` alone. This makes the result **invariant to the internal block size**
(the meaningful sense of "tile-invariant" for a standalone kernel — change `BLOCK` and every element
still draws the same dither; the tile-*local* `fp32_to_bf16_sr` kernel, keyed on `pid*BLOCK+lane`, is
not), with zero materialized key/uniform tensors — exactly the fused, never-materialize path the
compile mode can't generate.

The CuTeDSL kernel reaches the same 82.9% as its plain SR sibling because it is *literally the same
kernel*: the cute SR kernel is built on `cute.make_identity_tensor` global coordinates, so it already
keys its hand-written Philox on each element's global flat index (`counter = f >> 2`) regardless of
the tile shape — it was tile-invariant from the start, so the "global offsets" recipe just reuses it
(there is no separate tile-local cute kernel to contrast against, unlike the Triton pair).

## Why `mxfp8_dim_m` (59.9%) is slower than `fp8_deepseek_1x128_dim_m` (71.2%)

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

The register cost is **structural, not the e8m0 math**: replacing the manual e8m0 bit
extraction with the hardware `cvt.rz.satfinite.ue8m0x2.f32` instruction (see the mxfp8 kernel) only
freed ~6 registers (127→121) and moved the number 57.9% → 59.9% — the bulk of the pressure is the
fp32 working tile plus the `(RB,32,BN)` reshape and `tl.trans` transpose staging, plus holding 4×
the per-column scale state. Closing the gap to deepseek would require cutting *that* (e.g. a
shared-memory transpose to decouple store coalescing from tile height), not cheaper scale math.

## cuteDSL notes

### How `mxfp8_swizzle` was optimized (67.6% → 78.7%, matching triton)

The original cute kernel used the tensorwise vectorized-copy recipe (1-D flatten, 128 thr/CTA, each
thread owns one contiguous 1×32 block, 128-bit ld/st) and plateaued at ~67.6%. ncu on that kernel
showed the ceiling was **ALU-pipe bound (~68%), not DRAM bound**: the 1-D flatten recomputed the full
4D swizzle offset (a 6-op div/mod chain, `_swizzle_flat`) *per block per thread*.

The fix that closed the gap was **not** the TMA + smem-reduction kernel originally sketched here —
it was the much simpler **warp-per-row ("wpr") + hoisted-swizzle** mapping ported from
`_nvfp4_swizzle_kernel` (the nvfp4 cast, which had already been optimized the same way): warp `w`
owns a fixed row, so the row-dependent swizzle term `row_base` is computed **once per warp** and
amortized over every 1×32 block the lane visits (only the cheap `+(gc//4)*512 + gc%4` remains
per-block), and all 128-bit loads for the ILP group are issued first for memory-level parallelism.
Tuned WARPS=2, XSPLIT=4, ILP=4. That alone took it to 78.7%, edging triton (77.5%) — no TMA or smem
staging needed. See `_mxfp8_swizzle_kernel` in
[`quant_cast_cute/recipes.py`](../quant_cast_bench/quant_cast_cute/recipes.py).

Two dead ends ruled out along the way (don't repeat), both on the old 1-D-flatten kernel:

* **Reduce in bf16** to shrink the live vector — the compiler CSEs the store's f32 back, regs stay 48.
* **Paired-lane VPT=16** (two lanes share the block amax via `warp_reduction_max(threads_in_group=2)`)
  — cuts regs to 32 and lifts occupancy to 83%, but DRAM *drops* to 64%: reducing from a smaller
  register fragment loses memory-level parallelism. For a reduction kernel, **bigger VPT wins** even
  at lower occupancy. (The wpr rewrite sidesteps this entirely — it attacked the ALU tax, not
  occupancy.)
