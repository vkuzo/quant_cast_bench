# nvidia_rs_bit_probe

Reverse-engineers **how NVIDIA's `cvt.rs` x4 conversion intrinsics slice their single 32-bit random
word across the 4 output lanes**, for each of:

| intrinsic | cast | out | discarded f32 mantissa bits (D) |
|---|---|---|---|
| `cvt.rs.satfinite.e4m3x4.f32` | 4×f32 → 4×fp8 e4m3 | b32 (8-bit lanes) | 20 |
| `cvt.rs.satfinite.e5m2x4.f32` | 4×f32 → 4×fp8 e5m2 | b32 (8-bit lanes) | 21 |
| `cvt.rs.satfinite.e2m1x4.f32` | 4×f32 → 4×fp4 e2m1 | b16 (4-bit lanes) | 22 |

`.rs` is a rounding mode (PTX ISA 9.7.9.22): the instruction adds the lane's random slice to the
**discarded mantissa bits** of the f32 input and rounds up on the carry-out. All four lanes share a
single 32-bit random operand, so the question is which bits of that word each lane uses, and with
what weight.

Run (needs Blackwell sm_100a, cuda capability `(10, 0)`):

```
python probe.py
```

## Method

Take a base grid point `B = 1.0` whose next representable neighbor is `U`, and set the `D` discarded
mantissa bits of the f32 input to a controllable value `F`, giving fractional distance
`frac = F / 2^D` toward `U`. Then

```
lane rounds UP (B -> U)  iff  R_lane(W) / 2^16  >=  1 - frac
```

where `R_lane(W)` is the lane's 16-bit slice of the word `W`. Sweeping `F` for a fixed `W` locates
each lane's up/down transition, hence `R_lane(W)`. Feeding one-hot words `W = 1<<b` reads out, per
lane, **which physical bits it uses** (membership) and **their weight order** (a higher-weight bit
flips the lane at a smaller `F`, so ranking a lane's bits by transition-`F` ascending gives MSB→LSB —
this is what exposes forward vs bit-reversed slices).

Everything format-specific is discovered empirically, not hardcoded: the up/down output codes (from
the all-down `F=0` and all-up `F=max, W=all-ones` extremes) and the source-arg→output-element packing
(feed 4 increasing grid points, sort the output codes). A final cross-check confirms the additive
`round-up ⟺ R ≥ (1-frac)·2¹⁶` model on random full-width words.

## Result

**Shared structure (all three formats):** each output lane reads a **16-bit** slice, every physical
bit is reused by **exactly two** lanes, and the two lanes sharing a slice consume it with **opposite
weight order** (one forward, one bit-reversed). Only the *grouping* of bits differs between fp8 and
fp4.

Notation below: `a→b` means physical word-bit `a` is the **MSB (weight 2¹⁵)** of that lane's slice
and the run continues to bit `b` as the **LSB (weight 2⁰)**. Source args `$1..$4` pack into output
elements `3..0` (little-endian), so both are shown.

### e4m3 and e5m2 (identical) — two contiguous 16-bit halves

| source arg | output elem | bits (MSB→LSB) | order |
|---|---|---|---|
| `$1` | elem 3 | `16→31` | forward |
| `$2` | elem 2 | `31→16` | reversed |
| `$3` | elem 1 | `0→15` | forward |
| `$4` | elem 0 | `15→0` | reversed |

`$1`/`$2` share the high half (bits `16..31`); `$3`/`$4` share the low half (bits `0..15`).

### e2m1 (fp4) — byte-interleaved (each slice = two bytes)

| source arg | output elem | bits (MSB→LSB) | order |
|---|---|---|---|
| `$1` | elem 3 | `8→15, 24→31` | forward |
| `$2` | elem 2 | `31→24, 15→8` | reversed |
| `$3` | elem 1 | `0→7, 16→23` | forward |
| `$4` | elem 0 | `23→16, 7→0` | reversed |

`$1`/`$2` share bytes 1 & 3 (bits `8..15` + `24..31`); `$3`/`$4` share bytes 0 & 2
(bits `0..7` + `16..23`). Within each pair the reversed lane is the exact bit-reversal of the forward
lane (e.g. `$2`'s `31→24, 15→8` is `$1`'s `8→15, 24→31` reversed).

### Note on lane numbering

Results are reported by **intrinsic source arg** `$1..$4`. The x4 result packs `$1` into the
most-significant lane, so a little-endian view of the packed output reverses the order: source arg
`$i` lands in output element `4-i` (mapping `[3,2,1,0]`, also discovered empirically). A kernel that
wants output elements in memory order should load its 4 f32 lanes reversed into `$1..$4`.
