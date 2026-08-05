"""Bit-for-bit reproduction of Triton's `tl.randint4x` dither in eager PyTorch.

Three dither generators, keyed on each element's GLOBAL flat index:

- `rand16_triton`:  the dither the Triton recipe kernel `_sr_bf16_global_kernel`
  (quant_cast_triton/recipes.py) computes -- `tl.randint4x` raw uint32 words, low 16 bits --
  here laid across elements in contiguous `(r0,r1,r2,r3)` lane order. Stored out instead of
  applied to `x`. (The repo recipe uses a different, equivalent interleave lane order; the
  order is a free choice, see below.)
- `rand16_bits`:  NEW eager reference using `torch.func._random.bits` (PR pytorch#190253),
  which exposes Philox's RAW uint32 words. This bit-matches `rand16_triton` -- see below.
- `rand16_uniform`:  the OLD eager path (gold recipe `sr_bf16_global_f`, minus the
  dither/truncate) built on `prng.uniform`, a FLOAT uniform. Kept as a contrast: it does
  NOT match, because a float's top fractional bits != a raw uint32's low 16 bits.

Why `bits` matches `randint4x` (both are Philox4x32-10 with the same seed->key /
offset->counter mapping):
  * `bits(key, .., uint32)` returns philox words contiguously: element `4*c + lane` is
    lane `lane` of counter `c`, i.e. exactly `tl.randint4x(seed, c)[lane]`.
  * This kernel spreads a counter's 4 lanes across 4 elements in that same contiguous
    `(r0, r1, r2, r3)` order (via `interleave(interleave(r0,r2), interleave(r1,r3))`), so
    the eager side is just `bits` with the low 16 bits taken -- no permutation needed.

The within-group lane order is a free choice: it costs nothing in the generated PTX (the 4
words of one counter live in one thread's registers and map to 4 contiguous, thread-local
output addresses, so reordering is pure register relabeling -- verified: identical stores, no
smem/shuffle/barrier). We use the contiguous `(r0,r1,r2,r3)` order here for the simplest match.

`test.py` feeds all three the same inputs and compares. Requires a build with pytorch#190253.

Usage: python experiments/prng_match/test.py
"""

import torch
import torch.func._random as prng
import triton
import triton.language as tl


# --- old eager path: PyTorch FLOAT uniform (gold `sr_bf16_global_f`, minus dither/truncate) ---
def rand16_uniform(M, N, key, global_row=0, global_col=0, num_col=None, device="cuda"):
    """Return the (M, N) int32 dither the gold recipe would use via `prng.uniform`, keyed on
    each element's GLOBAL flat index. With the defaults (origins 0, num_col=N) the global index
    equals the plain row-major flat position `f`, matching the standalone Triton kernel.

    This does NOT bit-match the Triton kernel: `int(u * 2**16)` keeps the float's TOP fractional
    bits, whereas the recipe uses the LOW 16 bits of a raw uint32."""
    if num_col is None:
        num_col = N
    seed = key.reshape(-1)[0].item()
    # per-element global flat index (int64; uint64 mul is unsupported on cuda).
    i = (global_row + torch.arange(M, device=device)).view(-1, 1)
    j = (global_col + torch.arange(N, device=device)).view(1, -1)
    gidx = (i * num_col + j).reshape(-1).to(torch.int64)
    # per-element Philox key [seed, global_index]; one uniform drawn each.
    seed_t = torch.tensor([seed], device=device)[0:1].to(torch.int64).expand(gidx.numel())
    keys = torch.stack([seed_t, gidx], dim=-1).to(torch.uint64)
    u = prng.uniform(keys, (gidx.numel(),)).reshape(M, N)
    # float uniform -> integer dither: keeps the TOP fractional bits of the float.
    return (u * (1 << 16)).to(torch.int32)


# --- NEW eager path: raw Philox uint32 words via `prng.bits`, bit-matching the recipe kernel ---
def rand16_bits(n, key):
    """Return the (n,) int32 dither the Triton kernel computes, reproduced in eager PyTorch
    from the raw Philox words `prng.bits` exposes.

    Layout: `bits(key, 4*C, uint32)[4*c + lane] == tl.randint4x(seed, c)[lane]`. The kernel
    lays a counter's 4 lanes across 4 elements in that same contiguous `(r0,r1,r2,r3)` order,
    so `bits` already lines up flat -- just take the low 16 bits, no permutation.

    Note: exact only when the kernel's `seed` (low 32 bits of the key word) equals the full key
    word and the key's offset word is 0 -- the case for `prng.key(seed)` with a 32-bit seed.

    Requires the raw-bits API from pytorch#190253; raises a clear error if this torch build
    predates it."""
    if not hasattr(prng, "bits"):
        raise RuntimeError(
            "torch.func._random.bits is unavailable in this torch build. The raw-uint32 "
            "Philox API is required to bit-match tl.randint4x; build/install a torch that "
            "includes https://github.com/pytorch/pytorch/pull/190253."
        )
    b = prng.bits(key, n, dtype=torch.uint32).view(torch.int32)  # flat philox words, contiguous
    return (b & 0xFFFF).to(torch.int32)  # low 16 bits of a raw uint32; the kernel's rand16


# --- experiment: Triton `tl.randint4x`, using the recipe kernel's exact dither computation ---
@triton.jit
def _rand16_kernel(r16_ptr, seed_ptr, n_elements, BLOCK: tl.constexpr):
    # `_sr_bf16_global_kernel` (quant_cast_triton/recipes.py) with the load/add/truncate/
    # store-of-y replaced by storing rand16 directly, and the lane order set to contiguous
    # (r0,r1,r2,r3) -- a free choice vs the recipe's interleave order (identical PTX stores).
    seed = tl.load(seed_ptr)  # on-device, no host sync
    pid = tl.program_id(0)
    grp = pid * BLOCK + tl.arange(0, BLOCK)  # (BLOCK,) counter = global flat index >> 2
    r0, r1, r2, r3 = tl.randint4x(seed, grp)  # 4 streams; the group's 4 elements each take one
    rand = tl.interleave(tl.interleave(r0, r2), tl.interleave(r1, r3))  # (4*BLOCK,) (r0,r1,r2,r3)
    offs = pid * (4 * BLOCK) + tl.arange(0, 4 * BLOCK)  # contiguous global flat indices
    mask = offs < n_elements
    rand16 = (rand & 0xFFFF).to(tl.int32)  # low 16 bits of a raw uint32; the kernel's rand16
    tl.store(r16_ptr + offs, rand16, mask=mask)


def rand16_triton(n, key):
    """Return the (n,) int32 dither the Triton recipe kernel computes, one per flat index."""
    out = torch.empty(n, dtype=torch.int32, device="cuda")
    seed = key.reshape(-1)[:1].view(torch.int32)  # first 32 bits of the key, stays on-device
    BLOCK = 1024

    def grid(meta):
        return (triton.cdiv(n, 4 * meta["BLOCK"]),)

    _rand16_kernel[grid](out, seed, n, BLOCK=BLOCK)
    return out
