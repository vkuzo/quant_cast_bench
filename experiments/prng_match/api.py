"""Two dither generators, for a bit-for-bit comparison of PyTorch's stateless Philox PRNG
against Triton's `tl.randint4x`.

- `baseline_rand16`: the RNG body of the gold recipe `sr_bf16_global_f`
  (quant_cast_gold/recipes.py), stopped one step early -- it returns the integer dither
  `rand16` directly instead of applying it to `x` and truncating to bf16.
- `rand16_triton`: the *exact* dither the Triton recipe kernel `_sr_bf16_global_kernel`
  (quant_cast_triton/recipes.py) computes, stored out instead of applied to `x`.

`test.py` feeds both the same inputs and compares. They do NOT match today; see test.py.

Usage: python experiments/prng_match/test.py
"""

import torch
import torch.func._random as prng
import triton
import triton.language as tl


# --- baseline: PyTorch stateless Philox (gold `sr_bf16_global_f`, minus the dither/truncate) ---
def baseline_rand16(M, N, key, global_row=0, global_col=0, num_col=None, device="cuda"):
    """Return the (M, N) int32 dither `rand16` the gold recipe would use, keyed on each
    element's GLOBAL flat index. With the defaults (origins 0, num_col=N) the global index
    equals the plain row-major flat position `f`, matching the standalone Triton kernel."""
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


# --- experiment: Triton `tl.randint4x`, using the recipe kernel's exact dither computation ---
@triton.jit
def _rand16_kernel(r16_ptr, seed_ptr, n_elements, BLOCK: tl.constexpr):
    # This is `_sr_bf16_global_kernel` (quant_cast_triton/recipes.py) with the load/add/
    # truncate/store-of-y replaced by storing rand16 directly.
    seed = tl.load(seed_ptr)  # on-device, no host sync
    pid = tl.program_id(0)
    grp = pid * BLOCK + tl.arange(0, BLOCK)  # (BLOCK,) counter = global flat index >> 2
    r0, r1, r2, r3 = tl.randint4x(seed, grp)  # 4 streams; the group's 4 elements each take one
    rand = tl.interleave(tl.interleave(r0, r1), tl.interleave(r2, r3))  # (4*BLOCK,)
    offs = pid * (4 * BLOCK) + tl.arange(0, 4 * BLOCK)  # contiguous global flat indices
    mask = offs < n_elements
    rand16 = (rand & 0xFFFF).to(tl.int32)  # low 16 bits of a raw uint32; the recipe's rand16
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
