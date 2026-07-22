# quant_cast_gold

LLM-friendly gold **plain-PyTorch reference** recipes for single-kernel quant-casts (fp8 / mxfp8 / nvfp4 /
stochastic-rounding / RHT), written in a tile-invariant-friendly way.  The signature:

```python
@dataclass(frozen=True)
class QuantCastSingleKernelGold:
    # Callable[*inputs] -> *outputs - PyTorch native definition of a quant cast recipe
    pt_ref_fn: Callable
    # Callable[inputs, computed_outputs] - A function to test correctness of the outputs, throws on error
    correctness_fn: Callable
    # Convenience function to generate valid inputs
    example_input_fn: Callable[[int, int], Tuple[torch.Tensor, ...]]
    # String for humans to reason about performance considerations
    perf_description: str
```

Coverage so far:
* quant recipes: fp8 (deepseek, tensorwise, rowwise), mxfp8, nvfp4, RHT, RS
* input orientation: x, x.t(), both
* scale swizzles: none, blackwell_32_4_4

Note: these are all **single kernel**.  For recipes such as nvfp4 with global outer scale, we assume that
the outer scale is computed elsewhere and the user is responsible for composing the pieces together.

## motivation and tl;dr;

I did this to:
1. see what is missing in PT Core eager to express modern quantization recipes
2. evaluate options for quant API frontend
3. evaluate options for near-SOL quant backend

Findings:
1. missing in PT Core eager for modern quant casting:
   - fast RTNE casting to FP4 in eager
   - stochastic rounding in eager
     - expose integer randomness directly (https://github.com/pytorch/pytorch/pull/190253)
     - align on a design for a stochastic rounding op in eager (https://github.com/pytorch/pytorch/issues/175409)
   - TODO work through the MoE case
2. options for quant cast API frontend
   - (preferred) eager PyTorch, express quantization casting as a composition of primitives on plain tensors
   - (acceptable) flex_gemm/flex_ep/flex_moe, taking tile-invariant quantization function callbacks + tiling metadata
   - (acceptable) flex_tile_map with a dummy backend, to enable fusion of gemm + f to flex_gemm(..., f)
     - Note: a reason for this to exist with today's tooling is line up torch.autograd.Function boundaries for CODA flex_gemm
     - to get good performance with flex_tile_map + general case of quant, we need a compiler that outputs cuteDSL
     - TODO talk about new input broadcasting cases
     - TODO talk about example flex_tile_map API
3. options for quant cast API backend
   - (acceptable) per-recipe human written gold reference + LLM kernel gen
   - lowering target
     - cuteDSL - needed for near-SOL training
     - triton - good baseline, but SOL not reachable for training (quantizing `input.t()`)
   - compilers
     - torchinductor (we already have this)
       - triton backend
         - works well for quant inference in 8 bits
         - various gaps at 4 bits and for training that can be improved
       - cuteDSL backend
         - could extend to quant patterns (currently GEMM only) to cover training better
     - (?) mini-compiler for quantization cast reductions to cuteDSL
       - can expand quant recipe coverage for flex* family of products to cover training better

## tile invariance and quantization

**tile invariance** of a function `f` is desireable as it gives a backend maximium
flexibility in tiling and fusing the calculation.  **Some but not all** quantization
casts are expressible by a fully tile invariant `f`:

* **input orientation**: quantizing `input` can be tile invariant, but quantizing `input.t()` is not
* **quantization block_size** and **local output transform**: can be tile invariant with a constraint on the tiling
* **global element offset** for RS is not tile invariant

At a high level:
* inference (no `input.t() quant, no SR) - tile-invariant, with tiling constraints
  - exception: rowwise/colwise scaling is not 2d tile invariant
* training - not tile invariant (input_orientation `input.t()` breaks it)
* stochastic rounding - not tile invariant (global offsets break it)

## parametrization of single-kernel quantization casts

We parametrize the space of popular single-kernel quantization casts by the following properties

1. **input orientation**: whether we are quantizing `input`, `input.t()` or both in the same kernel
2. **quantization block size**: for example (1, 16), (32, 32), (1, 128), etc
3. presence of **sub-byte output packing**: for example packing qdata to fp4
4. presence of **tile-local output transform**: for example "nothing", or "blackwell 32_4_4 scale swizzle"
5. presence of **global offset** information: for example, for proper stochastic rounding we
   need to generate (M, K) local offsets to generate per-element randomness

There are other considerations such as general aux inputs/outputs, RHT, 4over6, etc which 
are not included in the analysis ^. We give more context on each of these below:

### input orientation

Quantizing both `input` and `input.t()` is commonly needed for quantized training,
to supply the backward quantized gemms with the operands in the right orientation. 
There are two considerations here:

1. the `input.t()` operation is not tile invariant, as it is a global axis flip. Note that we **can**
   express an equivalent transform with a composition of an (a) in-tile transpose and (b)
   transposing the tile positions in the output tensor.

   Example: `input.t()` of a 4x4 matrix (numbers are source positions), tiled into 2x2 tiles.

   ```text
   input:              goal (input.t()):
    0  1  2  3           0  4  8 12
    4  5  6  7           1  5  9 13
    8  9 10 11           2  6 10 14
   12 13 14 15           3  7 11 15

    input 2x2         transpose tiles    transpose tile positions
    0  1 |  2  3      0  4 | 2  6        0  4 |  8 12
    4  5 |  6  7      1  5 | 3  7        1  5 |  9 13
    -----+------  =>  -----+-----    =>  -----+------
    8  9 | 10 11      8 12 |10 14        2  6 | 10 14
   12 13 | 14 15      9 13 |11 15        3  7 | 11 15
   ```

   So the global transpose factors into a tile-invariant `f` (the local per-tile transpose) plus a
   grid-level swap of tile positions. The grid-level swap of tile positions must be handled by
   a backend that consumes a tile invariant `f`, and it requires extra information (for example,
   `OutputKinds.GRID_SWAP`, or a more general version of it).


3. performant kernels to quantize `input.t()` are not currently at SOL performance in triton
   due to limitations of triton itself (TODO link to details). We need a lower level 
   DSL (for example, cuteDSL) to reach near-SOL.

### quantization block size

Given a 2d (M, K) shapes input tensor, quantization block size denotes how we partition the tensor
to calculate the scales.

For example, for an (M, K) input:
* float8 rowwise: (1, -1) block size, (M, K) qdata, (M, 1) scale
* mxfp8 unswizzled: (1, 32) block size, (M, K) qdata, (M, K // 32) scale
* nvfp4 unswizzled: (1, 16) block size, (M, K // 2) qdata, (M, K // 16) scale
* deepseek 128x128 weight: (128, 128) block size, (M, K) qdata, (M // 128, K // 128) scale

This matters for kernel authoring because:
* block size determines if the reduction can be done within an SSA (easiest), within a CTA (harder) or cross-CTA (hardest)
* block size is correlated with register pressure (smaller blocks - more scales - more scale calculation instructions per CTA)

### sub-byte output packing

An (M, K) fp4 output is stored as (M, K // 2) bytes.

This matters for kernel authoring because:
* the output can't be addressed per element -- the kernel must **pack pairs** before storing. On
  Blackwell the hardware `cvt.rn.satfinite.e2m1x2.f32` converts and packs two fp32 into one fp4 byte
  in a single instruction.
* sub-byte packing doesn't cleanly map to hardware instructions such as `cvt.rn.satfinite.e2m1x2.f32`
  in either PyTorch or triton today (we should improve this)
* ^ is for 4 bits, mxfp6 with 6 bits and non-powers-of-two packing is relevant as well

### tile-local output transform

Some recipes rearrange the **scale** output into a hardware-specific layout that is local to each
scale "atom" -- e.g. Blackwell's block-scaled MMA wants the scales for each 128x4 block delivered in
a swizzled (32, 16) layout. 

```python
# scale: (rows, cols) row-major block-scales, rows % 128 == 0, cols % 4 == 0
n_row_blocks, n_col_blocks = rows // 128, cols // 4
# gather each 128x4 block, keeping the two block axes separate: (nrb, ncb, 128, 4)
blocks = scale.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
# 128x4 -> 32x16: split the 128 rows into (4, 32), move that 4 beside the trailing 4
rearranged = blocks.reshape(n_row_blocks, n_col_blocks, 4, 32, 4).transpose(-3, -2)
swizzled = rearranged.reshape(n_row_blocks, n_col_blocks, 32, 16)  # per-atom (32, 16) blocks
```

### global offset

Stochastic rounding needs each element's **global position** in the parent tensor, not just its position
within the tile to be unbiased *and* deterministic per element, so a given element gets the same
draw no matter how the tensor is tiled.

## supported recipes

Each recipe classified by the five properties above. `x` / `x.t()` / `both` is the input
orientation; `—` means "none / no".

| recipe | orientation | block size | sub-byte packing | tile-local transform | global offset | precalc_out_scale |
|---|---|---|---|---|---|---|
| `fp8_tensorwise_precalc_scale`     | x     | (-1, -1)        | —   | —       | —   | true |
| `fp8_rowwise_precalc_scale`        | x     | (1, -1)         | —   | —       | —   | true |
| `fp8_colwise_precalc_scale`        | x     | (-1, 1)         | —   | —       | —   | true |
| `mxfp8_floor`                      | x     | (1, 32)         | —   | —       | —   | — |
| `mxfp8_floor_swizzle`              | x     | (1, 32)         | —   | swizzle | —   | — |
| `mxfp8_floor_dim_m`                | x.t() | (32, 1)         | —   | —       | —   | — |
| `mxfp8_floor_dim_km`               | both  | (1,32)+(32,1)   | —   | —       | —   | — |
| `mxfp8_32x32_floor`                | x     | (32, 32)        | —   | —       | —   | — |
| `fp8_deepseek_1x128`               | x     | (1, 128)        | —   | —       | —   | — |
| `fp8_deepseek_1x128_dim_m`         | x.t() | (128, 1)        | —   | —       | —   | — |
| `fp8_deepseek_1x128_dim_km`        | both  | (1,128)+(128,1) | —   | —       | —   | — |
| `fp8_deepseek_128x128`             | x     | (128, 128)      | —   | —       | —   | — |
| `fp8_rowwise`                      | x     | (1, -1)         | —   | —       | —   | — |
| `fp8_colwise`                      | x.t() | (-1, 1)         | —   | —       | —   | — |
| `nvfp4_swizzle`                    | x     | (1, 16)         | fp4 | swizzle | —   | — |
| `nvfp4_blocked_outer`              | x     | (1, 16)         | fp4 | swizzle | —   | — |
| `bf16_rht`                         | x     | — (16-wide RHT) | —   | RHT     | —   | — |
| `fp32_to_bf16_sr`                  | x     | — (elementwise) | —   | —       | —   | — |
| `fp32_to_bf16_sr_global_offsets`   | x     | — (elementwise) | —   | —       | yes | — |
| `mxfp8_bias`                       | x     | (1, 32)         | —   | —       | —   | — |

Notes:
* the `_precalc_scale` recipes take the scale as an input (a plain elementwise divide + cast), so
  the block size only labels how that scale was computed (tensor / row / col).
* `fp8_colwise` and the `_dim_m` recipes write **transposed** output (`x.t()`); the `_dim_km`
  recipes emit both orientations in one pass (four outputs).
* `nvfp4_swizzle` vs `nvfp4_blocked_outer` are identical under these five axes — they differ only in
  the outer-scale source (per-tensor vs 128×128-blocked), which is a scale-scheme detail, not one of
  the five axes.
* `bf16_rht` (a 16-wide randomized Hadamard transform) and the `fp32_to_bf16_sr*` recipes are the
  non-block-quant examples; `mxfp8_bias` is a debug recipe.

## Code structure

- **`recipes.py`** — the recipes. Each is a `QuantCastSingleKernelGold` (frozen dataclass) with:
  - `pt_ref_fn(*inputs, **kwargs) -> outputs` — the reference cast in plain PyTorch.
  - `correctness_fn(inputs, outputs) -> None` — asserts a candidate set of outputs is valid (e.g.
    dequant recovers `x` above an SQNR threshold). Used by the backends' tests to accept
    hardware-rounding divergence from the exact reference.
  - `example_input_fn(M, K) -> (x, *aux)` — one representative input set.
  - `perf_description` — free-form note surfaced in the benchmark.

  All recipes are collected in **`ALL_RECIPES`** (list of `(name, gold)`), consumed by the tests and
  by the benchmark (`benchmarks/benchmark.py --mode compile` `torch.compile`s each `pt_ref_fn`).
- **`utils.py`** — fp4 (e2m1) helpers: `f32_to_f4_unpacked` / `f4_unpacked_to_f32`, `pack_uint4` /
  `unpack_uint4`.
- **`test/test_quant_cast_gold.py`** — parametrizes `ALL_RECIPES`, runs `pt_ref_fn` on a fixed shape,
  and asserts `correctness_fn`. Makes no assumption about the number of outputs (recipes return 1-,
  2-, or 4-tuples).

## Recipe families

fp8 elementwise (tensorwise / rowwise / colwise precalc-scale), mxfp8-floor (1×32, plus swizzle,
dim-m, 32×32, and the one-pass dim-k+dim-m combo), deepseek fp8 (1×128, 128×128, dim-m, and the
one-pass dim-k+dim-m combo), rowwise/colwise fp8, nvfp4 (1×16 with swizzled scale), the randomized
Hadamard transform, and fp32→bf16 stochastic rounding. See `ALL_RECIPES` for the full list.

## Test

```bash
cd /home/dev/quant_cast_bench
pytest test/test_quant_cast_gold.py -s
```
