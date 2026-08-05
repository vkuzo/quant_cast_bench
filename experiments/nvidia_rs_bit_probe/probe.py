"""Reverse-engineer how NVIDIA's `cvt.rs` x4 conversion intrinsics slice their single 32-bit random
word across the 4 output lanes, for each of:

    cvt.rs.satfinite.e4m3x4.f32    4 f32 -> 4 fp8 e4m3   (b32 out, 8-bit lanes)
    cvt.rs.satfinite.e5m2x4.f32    4 f32 -> 4 fp8 e5m2   (b32 out, 8-bit lanes)
    cvt.rs.satfinite.e2m1x4.f32    4 f32 -> 4 fp4 e2m1   (b16 out, 4-bit lanes)

Mechanism (`.rs`, PTX ISA 9.7.9.22): the instruction adds the lane's random slice to the DISCARDED
mantissa bits of the f32 input and rounds up (away from truncation) on the carry-out. So with a base
grid point B whose next representable neighbor is U, an input x = B with its D discarded mantissa bits
set to a controllable value F has fractional distance frac = F / 2^D to U, and:

    lane rounds UP (B -> U)  iff  R_lane(W) / 2^k  >=  1 - frac       (R_lane = the lane's slice of W)

For a fixed random word W, sweeping F over [0, 2^D) locates each lane's up/down transition -> R_lane(W).
We use one-hot words W = 1<<b to read out, per lane:
  * membership -- which physical bits b feed the lane (a bit triggers a round-up for some F iff it is
    in the lane's slice); and
  * weight order -- a higher-weight bit flips the lane at a SMALLER F, so ranking a lane's bits by
    their transition-F ascending gives MSB -> LSB, exposing forward vs bit-reversed slices.

Everything format-specific (the up/down output codes, the source-arg -> output-position packing) is
DISCOVERED empirically here, not hardcoded: up/down codes from the all-down (F=0) and all-up
(F=max, W=all-ones) extremes; the packing permutation by feeding 4 monotically increasing grid points
and sorting the output codes. The only structural assumption is that D = 23 - (format mantissa bits).

Empirically all three share the same structure: each lane reads a 16-bit slice, every physical bit is
reused by exactly two lanes, and the two lanes sharing a slice consume it with OPPOSITE weight order
(one forward, one bit-reversed). Only the grouping differs -- the fp8 formats (e4m3/e5m2) split the
word into two contiguous 16-bit halves (lanes 0,1 <- bits[16:31]; lanes 2,3 <- bits[0:15]), whereas
e2m1 (fp4) splits it byte-interleaved (lanes 0,1 <- bytes 1&3; lanes 2,3 <- bytes 0&2).
Run: `python probe.py` (needs Blackwell sm_100a).
"""

import torch
import triton
import triton.language as tl

DEV = "cuda"
BLOCK = 256
SWEEP_BITS = 18  # F-sweep points = 2^18: > 2^k for k<=18, enough to resolve every bit's weight


# two kernels, differing only in the output register width (b32 for fp8, b16 for fp4); the asm
# string / constraints are threaded in as constexpr (the `inline_asm` dtype must be a literal).
@triton.jit
def _probe_b32(x_ptr, rb_ptr, y_ptr, num_groups, ASM: tl.constexpr, CONSTR: tl.constexpr,
               BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    g = pid * BLOCK + tl.arange(0, BLOCK)
    mask = g < num_groups
    base = g * 4
    a = tl.load(x_ptr + base + 0, mask=mask)  # $1 = source lane 0
    b = tl.load(x_ptr + base + 1, mask=mask)  # $2 = source lane 1
    c = tl.load(x_ptr + base + 2, mask=mask)  # $3 = source lane 2
    d = tl.load(x_ptr + base + 3, mask=mask)  # $4 = source lane 3
    rb = tl.load(rb_ptr + g, mask=mask)
    q = tl.inline_asm_elementwise(asm=ASM, constraints=CONSTR, args=[a, b, c, d, rb],
                                  dtype=tl.int32, is_pure=True, pack=1)
    tl.store(y_ptr + g, q, mask=mask)


@triton.jit
def _probe_b16(x_ptr, rb_ptr, y_ptr, num_groups, ASM: tl.constexpr, CONSTR: tl.constexpr,
               BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    g = pid * BLOCK + tl.arange(0, BLOCK)
    mask = g < num_groups
    base = g * 4
    a = tl.load(x_ptr + base + 0, mask=mask)
    b = tl.load(x_ptr + base + 1, mask=mask)
    c = tl.load(x_ptr + base + 2, mask=mask)
    d = tl.load(x_ptr + base + 3, mask=mask)
    rb = tl.load(rb_ptr + g, mask=mask)
    q = tl.inline_asm_elementwise(asm=ASM, constraints=CONSTR, args=[a, b, c, d, rb],
                                  dtype=tl.int16, is_pure=True, pack=1)
    tl.store(y_ptr + g, q, mask=mask)


# one spec per intrinsic. mant = format mantissa bits -> D = 23 - mant discarded f32 mantissa bits;
# lane_bits = width of each packed output element; up = next e-format value above the base 1.0.
FORMATS = [
    dict(name="e4m3", asm="cvt.rs.satfinite.e4m3x4.f32 $0, {$1, $2, $3, $4}, $5;",
         constr="=r,f,f,f,f,r", mant=3, lane_bits=8, out_dt=torch.int32, up=1.125),
    dict(name="e5m2", asm="cvt.rs.satfinite.e5m2x4.f32 $0, {$1, $2, $3, $4}, $5;",
         constr="=r,f,f,f,f,r", mant=2, lane_bits=8, out_dt=torch.int32, up=1.25),
    dict(name="e2m1", asm="cvt.rs.satfinite.e2m1x4.f32 $0, {$1, $2, $3, $4}, $5;",
         constr="=h,f,f,f,f,r", mant=1, lane_bits=4, out_dt=torch.int16, up=1.5),
]


def _launch(x, rb, spec):
    """Run the intrinsic over len(x)//4 groups; return the packed output (int32 or int16), one per group."""
    m = x.numel() // 4
    out = torch.empty(m, dtype=spec["out_dt"], device=DEV)
    grid = (triton.cdiv(m, BLOCK),)
    kernel = _probe_b16 if spec["out_dt"] == torch.int16 else _probe_b32
    kernel[grid](x, rb, out, m, ASM=spec["asm"], CONSTR=spec["constr"], BLOCK=BLOCK)
    return out


def _codes(out, spec):
    """Unpack the packed output into per-output-position integer codes, shape (num_groups, 4).
    Position 0 is the least-significant lane_bits of the word."""
    lb = spec["lane_bits"]
    width = 4 * lb  # 32 for fp8, 16 for fp4
    q = out.to(torch.int64) & ((1 << width) - 1)  # treat the packed word as unsigned
    mask = (1 << lb) - 1
    return torch.stack([(q >> (lb * p)) & mask for p in range(4)], dim=1)  # (num_groups, 4)


def _int32_word(W):
    return W - (1 << 32) if W >= (1 << 31) else W  # python int -> int32 two's complement


def _fmt_runs(bits):
    """Collapse an ordered MSB->LSB bit list into 'a->b' runs of consecutive (+/-1) bits, e.g.
    [16,17,..,31] -> '16->31', [8,..,15,24,..,31] -> '8->15, 24->31'."""
    runs, i, n = [], 0, len(bits)
    while i < n:
        j, step = i, (bits[i + 1] - bits[i] if i + 1 < n else 0)
        if step in (1, -1):
            while j + 1 < n and bits[j + 1] - bits[j] == step:
                j += 1
        runs.append(str(bits[i]) if j == i else f"{bits[i]}->{bits[j]}")
        i = j + 1
    return ", ".join(runs)


def probe(spec):
    name, D = spec["name"], 23 - spec["mant"]
    G = 1 << SWEEP_BITS
    f_step = 1 << (D - SWEEP_BITS) if D > SWEEP_BITS else 1
    n_sweep = 1 << min(D, SWEEP_BITS)

    # base grid point 1.0 (0x3F800000) with its D discarded mantissa bits set to F = i*f_step; all 4
    # lanes identical, so lanes differ only in which random bits they read.
    F = (torch.arange(n_sweep, dtype=torch.int32, device=DEV) * f_step)
    x_sweep = (0x3F800000 | F).view(torch.float32).repeat_interleave(4)  # (4*n_sweep,)

    def run(W):
        rb = torch.full((n_sweep,), _int32_word(W), dtype=torch.int32, device=DEV)
        return _codes(_launch(x_sweep, rb, spec), spec)  # (n_sweep, 4)

    # discover the down/up output codes: F=0 always truncates (round down); F=max with an all-ones
    # word always carries (round up). Both must be uniform across the 4 positions (same grid point).
    down = run(0)[0]
    up = run(0xFFFFFFFF)[-1]
    assert (down == down[0]).all() and (up == up[0]).all(), "down/up codes not uniform across lanes"
    down_code, up_code = int(down[0]), int(up[0])
    assert down_code != up_code, "F=0 and F=max produced the same code; base/up choice is wrong"

    # discover source-arg ($1..$4) -> output-position packing: feed 4 monotonically increasing grid
    # points as the 4 source lanes (F=0, W=0 -> each rounds to itself). Larger value -> larger code,
    # so the position holding the L-th smallest code is source lane L's position.
    perm_x = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32, device=DEV)
    perm_codes = _codes(_launch(perm_x, torch.zeros(1, dtype=torch.int32, device=DEV), spec), spec)[0]
    src_to_pos = torch.argsort(perm_codes).tolist()  # src_to_pos[L] = output position of source lane L

    # one-hot sweep: for each physical bit b, the first F (index) at which each position rounds up.
    thr = torch.full((32, 4), -1, dtype=torch.long)
    for b in range(32):
        up_mask = run(1 << b) == up_code  # (n_sweep, 4)
        anyup = up_mask.any(0)
        thr[b] = torch.where(anyup, up_mask.int().argmax(0), torch.full((4,), -1, device=DEV)).cpu()

    print(f"=== {name}: cvt.rs.satfinite.{name}x4.f32  (D={D} discarded bits, lane_bits={spec['lane_bits']}) ===")
    print(f"    output codes: down(1.0)=0x{down_code:x}  up({spec['up']})=0x{up_code:x}")
    print(f"    source arg $1..$4 -> output elements {src_to_pos}")
    print("    each row lists the 16 word-bits a lane reads, MSB (weight 2^15) -> LSB (weight 2^0):\n")

    weight = [dict() for _ in range(4)]  # weight[src][physical_bit] = weight within the slice
    order = [""] * 4
    bitset = [frozenset()] * 4
    for src in range(4):
        col = thr[:, src_to_pos[src]]
        members = sorted([(b, int(col[b])) for b in range(32) if col[b] >= 0], key=lambda t: t[1])
        bits = [b for b, _ in members]  # MSB -> LSB
        k = len(members)
        for rank, (b, _) in enumerate(members):
            weight[src][b] = 1 << (k - 1 - rank)  # MSB = 2^(k-1) ... LSB = 2^0
        order[src] = "forward " if bits == sorted(bits) else ("reversed" if bits == sorted(bits, reverse=True) else "mixed   ")
        bitset[src] = frozenset(bits)
        print(f"    $ {src + 1} (elem {src_to_pos[src]}):  MSB->LSB = {_fmt_runs(bits):<22}  [{order[src].strip()}]")

    # pairs: the two lanes sharing the same 16-bit set, forward first (its reverse is the partner)
    seen = set()
    for i in range(4):
        j = next((k for k in range(4) if k != i and bitset[k] == bitset[i]), None)
        if j is not None and i not in seen:
            seen.update({i, j})
            fwd, rev = (i, j) if order[i].strip() == "forward" else (j, i)
            print(f"    -> $ {fwd + 1} and $ {rev + 1} share the same 16 bits (one forward, one bit-reversed)")

    # cross-check the additive linear model on random full-width words: round up iff R>=(1-frac)*2^k.
    gen = torch.Generator().manual_seed(1)
    ok = True
    for _ in range(8):
        W = int(torch.randint(0, 1 << 32, (1,), generator=gen, dtype=torch.int64).item())
        up_mask = run(W) == up_code
        for src in range(4):
            k = len(weight[src])
            R = sum(w for b, w in weight[src].items() if (W >> b) & 1)
            pred = min(max(int(-(-G * (1 - R / (1 << k)))) if k else 0, 0), n_sweep)  # ceil(G*(1-R/2^k))
            col = up_mask[:, src_to_pos[src]]
            meas = int(col.int().argmax(0)) if col.any() else n_sweep
            ok = ok and abs(pred - meas) <= 2
    print(f"    additive bit->weight model matches 8 random words: {ok}\n")
    return ok


def main():
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)):
        raise SystemExit("cvt.rs emits Blackwell-only PTX; needs cuda capability (10, 0)")
    all_ok = all(probe(spec) for spec in FORMATS)
    print("ALL FORMATS: additive slice model verified" if all_ok else "MISMATCH in at least one format")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
