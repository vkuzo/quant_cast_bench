"""Triton kernels backing the two stochastic-rounding modes of `api.to`, fp32 -> bf16 only.

Two kernels, both keyed on each element's GLOBAL flat index so their output is invariant to the
launch tiling (block size):

  * `_sr_bf16_software_kernel` -- software SR. Dither the 16 mantissa bits that fp32->bf16 drops
    with a uniform 16-bit random value, then truncate. Portable to any GPU.
  * `_sr_bf16_hardware_kernel` -- hardware SR via the Blackwell-only PTX intrinsic
    `cvt.rs.bf16x2.f32`, inlined with `tl.inline_asm_elementwise`. The rounding happens inside the
    instruction (it adds `rbits` to the truncated mantissa bits and rounds on the carry-out), so it
    is faster but NOT reproducible from eager PyTorch. Requires cuda capability (10, 0).

Both take the Philox `seed` as an on-device int32 tensor (no host sync); `api.py` resolves a
key/generator down to that seed. Randomness is `tl.randint4x`, which returns four independent uint32
Philox words per counter; we key the counter on `global_flat_index >> 2` so one Philox call feeds 4
consecutive elements (software) / the interleave lays 4 words across 4 groups (hardware), wasting no
words and keeping the counter->element map a pure function of the global index.
"""

import torch
import triton
import triton.language as tl


# --- software SR: copy of quant_cast_triton/recipes.py `_sr_bf16_global_kernel` -----------------
@triton.jit
def _sr_bf16_software_kernel(x_ptr, y_ptr, seed_ptr, n_elements, BLOCK: tl.constexpr):
    seed = tl.load(seed_ptr)  # on-device, no host sync
    pid = tl.program_id(0)
    grp = pid * BLOCK + tl.arange(0, BLOCK)  # (BLOCK,) group index = global flat index >> 2
    r0, r1, r2, r3 = tl.randint4x(seed, grp)  # 4 streams; the group's 4 elements each take one
    # interleave the 4 streams back to the contiguous 4*BLOCK element span -> coalesced ld/st.
    # Element at flat index f gets counter f>>2 (independent of BLOCK) and a stream fixed by f&4 --
    # a pure function of f, so the dither is invariant to the launch tiling.
    rand = tl.interleave(tl.interleave(r0, r1), tl.interleave(r2, r3))  # (4*BLOCK,)
    offs = pid * (4 * BLOCK) + tl.arange(0, 4 * BLOCK)  # contiguous global flat indices
    mask = offs < n_elements
    xi = tl.load(x_ptr + offs, mask=mask).to(tl.int32, bitcast=True)
    rand16 = (rand & 0xFFFF).to(tl.int32)  # uniform 16-bit dither; randint4x is uint32
    xi = (xi + rand16) & -65536  # add dither, then truncate the low 16 mantissa bits
    y = xi.to(tl.float32, bitcast=True).to(tl.bfloat16)  # exact: low 16 bits are zero
    tl.store(y_ptr + offs, y, mask=mask)


def sr_bf16_software_triton(x, seed, block=1024):
    """Tiling-invariant software fp32 -> bf16 stochastic rounding. `seed` is an on-device int32
    tensor; `block` is exposed so callers can prove tile-invariance across launch sizes."""
    assert x.dtype == torch.float32, f"SR bf16 expects fp32 input, got {x.dtype}"
    assert x.is_contiguous()
    out = torch.empty_like(x, dtype=torch.bfloat16)
    n = x.numel()

    def grid(meta):
        return (triton.cdiv(n, 4 * meta["BLOCK"]),)

    _sr_bf16_software_kernel[grid](x, out, seed, n, BLOCK=block)
    return out


# --- hardware SR: copy of nvidia_rs_demo/api.py `_rs_pair_kernel` for bf16x2 ---------------------
# The `cvt.rs.bf16x2.f32` intrinsic converts 2 f32 lanes -> one b32 (two packed bf16), consuming
# one 32-bit `rbits` word (16 dither bits per element). Requires sm_100a / sm_103a (Blackwell).
@triton.jit
def _sr_bf16_hardware_kernel(x_ptr, y_ptr, seed_ptr, num_groups, BLOCK: tl.constexpr):
    # BLOCK = input f32 elements per program. Each group is 2 elements needing one 32-bit rbits
    # word, and randint4x yields 4 words per counter, so this program needs:
    #     groups   = BLOCK // 2
    #     counters = groups // 4 = BLOCK // 8   (one randint4x call each, all 4 words used)
    seed = tl.load(seed_ptr)  # Philox seed, on-device (no host sync)
    pid = tl.program_id(0)
    ctr = pid * (BLOCK // 8) + tl.arange(0, BLOCK // 8)  # unique Philox counters for this program
    r0, r1, r2, r3 = tl.randint4x(seed, ctr)             # 4 words per counter, (BLOCK//8,) each
    # interleave lays the words down so 4 consecutive groups take one counter's 4 words: no word is
    # recomputed and none discarded. Element f -> group f>>1 -> counter f>>3 (independent of BLOCK).
    rbits = tl.interleave(tl.interleave(r0, r1), tl.interleave(r2, r3))  # (BLOCK//2,) one word/group
    g = pid * (BLOCK // 2) + tl.arange(0, BLOCK // 2)     # group index (each group = 2 elements)
    mask = g < num_groups
    base = g * 2
    # a -> upper half of the packed result, b -> lower half (little-endian view yields [b, a]);
    # load a from the odd lane and b from the even lane so the stored pair is [even, odd].
    a = tl.load(x_ptr + base + 1, mask=mask)
    b = tl.load(x_ptr + base + 0, mask=mask)
    q = tl.inline_asm_elementwise(
        asm="cvt.rs.bf16x2.f32 $0, $1, $2, $3;", constraints="=r,f,f,r", args=[a, b, rbits],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    tl.store(y_ptr + g, q, mask=mask)


def sr_bf16_hardware_triton(x, seed, block=1024):
    """Tiling-invariant hardware fp32 -> bf16 stochastic rounding via `cvt.rs.bf16x2.f32`.
    Blackwell-only (the caller must gate on cuda capability). `seed` is an on-device int32 tensor."""
    assert x.dtype == torch.float32 and x.is_contiguous()
    n = x.numel()
    assert n % 2 == 0, "bf16x2 needs an even element count"
    x = x.reshape(-1)
    num_groups = n // 2
    out_i32 = torch.empty(num_groups, dtype=torch.int32, device=x.device)
    grid = (triton.cdiv(n, block),)  # block = input elements per program
    _sr_bf16_hardware_kernel[grid](x, out_i32, seed, num_groups, BLOCK=block)
    return out_i32.view(torch.bfloat16).reshape(x.shape)
