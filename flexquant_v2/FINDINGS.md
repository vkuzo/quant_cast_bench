# FINDINGS

Running log of deltas from Ed's toy `tile_map` and Helion holes hit, as we add
quant recipes one by one.

Ed's toy primitive:

```python
def tile_map(f, x):
  m, n = x.size()
  out, aux = alloc_empty_out_aux(f, x)
  for tile_m, tile_n in hl.tile([m, n]):
    acc = x[tile_m, tile_n]
    out[tile_m, tile_n], aux[tile_m, tile_n] = f(acc)
  return out, aux
```

## Recipe 1: deepseek fp8 1x128 (`deepseek_1x128_f`)

Status: **runs, bit-exact vs plain-PyTorch reference** (256x256 bf16, B200,
`pytest test.py` passes in ~5s with a fixed `helion.Config`).

### Delta 1 — aux is coarser than out (block space vs element space) — SOLVED via option D

The scale output is `(M, N//128)`, not `(M, N)`. Ed's `aux[tile_m, tile_n] = ...`
assumes aux shares x's element grid and does **not** type-check for a real quant
recipe. So `tile_map` must own a **tile→output coordinate mapping**.

**Resolution: option D — every output is written at `tile_coords // divisor`.** There
is no special "out vs aux": `out` has n-divisor 1 (element grid), `aux` has n-divisor
BLOCK_N (block grid). One rule (`_output_index`) covers both, and will also cover
nvfp4's shrinking qdata (divisor 2) and 128x128 scales (divisor 128,128) without new
special-casing. The divisor comes from the probe (option b).

```python
qdata[tile_m, blk]                   # divisor 1 -> element grid
scale[tile_m, blk.begin // BLOCK_N]  # divisor BLOCK_N -> block grid
```

### Delta 2 — Helion forces option (b) -> (c): block size can't come from f

Original plan: infer the block from `f`'s grouped output shape
(`acc.reshape(m, -1, 128).amax(-1)`), option (b). This **does not survive contact
with Helion**, because making the block inferable requires `f` to *reshape*, and
Helion cannot trace a reshape of a symbolic tile in-kernel:

```
helion.exc.TorchOpTracingError: RuntimeError:
  shape '[u1, -1, u0]' invalid for input of size u1*u2      # f called in-kernel
AssertionError: 64 not in VR[128, 128]                      # same, via probe VR
```

So `f` must be authored in **per-block form** (reduce `dim=-1` on an already-sliced
block, no reshape). That form lowers fine — but it no longer encodes the block size.
Result: **block size must be passed explicitly**, `tile_map(f, x, block_shape=(1,128))`
(option c). `_probe_specs` then runs `f` on one `(1,128)` block to recover output
divisors/dtypes. Net: **Helion's no-symbolic-reshape rule collapses (b) into (c)** —
the exact axis we debated, decided by the compiler, not preference.

### Delta 3 — tile > block: nested tiling, NOT in-tile reshape (Hole B, resolved)

We now run tile_n = 2 blocks (`TILE_BLOCKS=2`) to decouple tile from block.

**In-tile reshape is a hard Helion hole.** Reshaping a symbolic tile-N dim into
`(blocks, BLOCK_N)` fails, both with `-1` and with an explicit middle dim:

```
helion.exc.TorchOpTracingError: RuntimeError:
  shape '[u1, u3, u0]' is invalid for input of size u1*u2
  at:  x_b = acc.reshape(bm, TILE_BLOCKS, BLOCK_N)
```

Helion won't accept that the symbolic tile extent `u2` equals `TILE_BLOCKS*BLOCK_N`.

**Resolution: nested tiling.** Outer autotuned tile spans TILE_BLOCKS blocks; an
inner `hl.tile(tile_n.begin, tile_n.end, block_size=BLOCK_N)` pins each block, so the
per-block reduction needs no reshape and each block writes its own aux column:

```python
for tile_m, tile_n in hl.tile([M, N]):
    for blk in hl.tile(tile_n.begin, tile_n.end, block_size=BLOCK_N):
        acc = x[tile_m, blk]                       # (BM, BLOCK_N), no reshape
        s = acc.abs().amax(-1, keepdim=True)...    # per-block reduce
        qdata[tile_m, blk] = ...                    # divisor 1
        scale[tile_m, blk.begin // BLOCK_N] = ...   # divisor BLOCK_N
```

Takeaway for the abstraction: **the tile>block sub-reduction must be a nested
`hl.tile` pinned to the block, not an in-tile reshape.** This is how `tile_map`
should lower "reduction granularity < tile size" in Helion.

**Re-confirmed after Delta 4** (with `f` as an argument, "single loop + reshape inside
`f`"): still fails. Moving the reshape into `f` doesn't help because `f` is inlined
during tracing, so it's the same symbolic reshape. Tested the full matrix — tile==block
and tile>block, `-1` and explicit middle dim, `static_shapes` False and True — **all
four/six variants fail identically** (`shape '[u1, u4, u5]' invalid for input of size
u1*u2`). The tile block size is a symbolic `u2` at trace time even when pinned in the
config, and Helion won't prove `u2 == nblk * BLOCK_N`. So the single-loop form is not
available; nested tiling is mandatory for any tile that spans >1 block.

### Delta 4 — `tile_map` calls `f` as a kernel ARGUMENT (Hole A)

The kernel no longer inlines the recipe: the loop body calls `f(acc)` and the
deepseek math lives only in `deepseek_1x128_f`. `f` is passed as a **kernel
parameter** (`_tile_map_kernel(x, f, aux_n_div)`), not captured.

**Correction of an earlier mistake in this file.** I first concluded "Helion bans
closures" and routed `f` through module globals. That was wrong / an over-reading.
The precise rule: the **decorated kernel function** may not have free variables
(`fn.__code__.co_freevars`; the check is `output_header.py:113`). Defining the kernel
*inside* `tile_map` so it closed over `f` tripped that
(`ClosuresNotSupported: ('block_n','f','specs')`). But a **callable passed as an
argument** is fully supported and is the intended pattern — see upstream
`examples/matmul.py` and `examples/matmul_split_k.py`, which take
`epilogue: Callable = lambda acc, tile: acc` and call it in the loop. The arg-callable
may itself close over tensors in the *caller's* scope (that's the blog's "f can close
over Tensors"); Helion inlines its body during tracing. So: kernel-with-free-vars =
banned; callable-argument-that-closes-over-tensors = blessed. The globals hack is
gone; no reentrancy wart.

Two smaller Helion lowering constraints hit while wiring this up (still true):

1. **No heterogeneous-dtype output list in-trace.** Allocating outputs with a
   `for (dm,dn,dt) in specs: torch.empty(...)` comprehension fails to merge fp8 vs
   fp32 across the loop (`Can't combine types from control flow: fp8 and fp32`).
   Allocate each output explicitly instead. (This is why `_tile_map_kernel` currently
   hardcodes the two output dtypes rather than looping over `specs` — a generality
   limitation to revisit for multi-output recipes.)
2. **No integer tensor indexing.** `aux_local[:, 0]` -> `InvalidIndexingType: got 0`.
   Use `aux_local.squeeze(-1)` to drop the reduced dim instead.

So `tile_map` is a real HOF (swap `f`, get a different cast), matching the upstream
epilogue-callable pattern.

## Open holes / next steps

- **Hole A — generic f.** RESOLVED (Delta 4): `f` is a kernel argument, deepseek math
  not inlined, no globals. Follow-up: output allocation still hardcodes 2 dtypes
  (constraint 1 above) — generalize when a multi-output recipe (nvfp4 two-level) lands.
- **Hole B — decouple tile from block.** RESOLVED via nested tiling (see Delta 3):
  outer autotuned tile + inner block-pinned `hl.tile`. In-tile reshape does not lower.
  Still to do: run this under real autotuning (currently a fixed outer config) to
  confirm the outer tile can vary while aux shape stays config-independent.
- **Hole C — autotuning + aux shape.** Aux shape here is config-independent (derived
  from x.shape + inferred divisor), so it should survive Helion's autotuner baseline
  accuracy check. Not yet exercised with real autotuning (fixed config used).
- **Next recipes to stress the abstraction:** dim=-2 (reduce across M, transposed
  output), 128x128 2D blocks, nvfp4 (sub-byte packing -> qdata N-dim also shrinks),
  mxfp8 (e8m0 scale dtype), then two-level/tensorwise (partial reduction + finish
  kernel) and padding/swizzle.
