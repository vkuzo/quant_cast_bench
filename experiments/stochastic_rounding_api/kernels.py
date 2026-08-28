"""Triton kernels backing the stochastic-rounding modes of `api.to`.

All kernels are keyed on each element's GLOBAL flat index so their output is invariant to the launch
tiling (block size):

  * `_sr_bf16_software_kernel` -- software SR fp32->bf16. Dither the 16 mantissa bits that the cast
    drops with a uniform 16-bit random value, then truncate. Portable to any GPU.
  * `_sr_fp8_software_kernel` -- software SR fp32->float8_e4m3fn. The same add-dither-then-truncate
    trick over the 20 mantissa bits that fp32->fp8 drops. Portable to any GPU.
  * `_sr_fp4_software_kernel` -- software SR fp32->float4_e2m1fn_x2 (packed nvfp4). Same dither trick
    over the 22 mantissa bits fp32->fp4 drops, plus a +-6 clamp; the e2m1 pack (two codes per byte)
    then uses the Blackwell `cvt.rn.satfinite.e2m1x2.f32` intrinsic (like the repo's `_nvfp4_kernel`),
    so it needs cuda capability (10, 0). Round-to-nearest cvt on the already-SR'd grid points, so the
    stochastic rounding is still fully in software.
  * `_sr_bf16_hardware_kernel` / `_sr_fp8_hardware_kernel` -- hardware SR via the Blackwell-only PTX
    intrinsics `cvt.rs.bf16x2.f32` / `cvt.rs.satfinite.e4m3x4.f32`, inlined with
    `tl.inline_asm_elementwise`. The rounding happens inside the instruction (it adds `rbits` to the
    truncated mantissa bits and rounds on the carry-out), so it is faster but NOT reproducible from
    eager PyTorch. Requires cuda capability (10, 0).

All take the Philox `seed` as an on-device int32 tensor (no host sync); `api.py` resolves a
key down to that seed. Randomness is `tl.randint4x`, which returns four independent uint32
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
    # lay the 4 streams across the contiguous 4*BLOCK element span in (r0,r1,r2,r3) lane order:
    # element f takes counter f>>2 (independent of BLOCK -> tile-invariant) and lane f&3, so
    # element 4c+lane == tl.randint4x(seed,c)[lane] == prng.bits(key,..,uint32)[4c+lane]. The lane
    # order is a free choice (identical PTX); this one lets api.py's eager reference bit-match
    # prng.bits with no permutation (see experiments/prng_match).
    rand = tl.interleave(tl.interleave(r0, r2), tl.interleave(r1, r3))  # (4*BLOCK,) (r0,r1,r2,r3)
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


# --- software SR for fp8_e4m3fn: the same add-dither-then-truncate trick, just wider ------------
# fp32 keeps its top 3 mantissa bits for a float8_e4m3fn normal (bias-7 exponent, 3-bit mantissa),
# so fp32->fp8 drops the low 20 mantissa bits (vs 16 for bf16). Dither those 20 bits with a uniform
# random then truncate toward the kept bits: the result is one of the two fp8 grid points bracketing
# |x|, chosen with the right probability -> unbiased SR. The `.to(tl.float8e4nv)` is exact in the
# normal range (the value already has <=3 mantissa bits); in the subnormal range (|x| < 2^-6) it
# rounds the kept value onto the coarser 2^-9 grid, which stays two-neighbor and unbiased in practice.
@triton.jit
def _sr_fp8_software_kernel(x_ptr, y_ptr, seed_ptr, n_elements, BLOCK: tl.constexpr):
    seed = tl.load(seed_ptr)  # on-device, no host sync
    pid = tl.program_id(0)
    grp = pid * BLOCK + tl.arange(0, BLOCK)  # (BLOCK,) group index = global flat index >> 2
    r0, r1, r2, r3 = tl.randint4x(seed, grp)  # 4 streams; the group's 4 elements each take one
    # contiguous (r0,r1,r2,r3) lane order so element 4c+lane == tl.randint4x(seed,c)[lane] ==
    # prng.bits(key,..,uint32)[4c+lane], letting api.py's eager reference bit-match with no permute.
    rand = tl.interleave(tl.interleave(r0, r2), tl.interleave(r1, r3))  # (4*BLOCK,) (r0,r1,r2,r3)
    offs = pid * (4 * BLOCK) + tl.arange(0, 4 * BLOCK)  # contiguous global flat indices
    mask = offs < n_elements
    xi = tl.load(x_ptr + offs, mask=mask).to(tl.int32, bitcast=True)
    rand20 = (rand & 0xFFFFF).to(tl.int32)  # uniform 20-bit dither; randint4x is uint32
    xi = (xi + rand20) & -1048576  # add dither, then truncate the low 20 mantissa bits (0xFFF00000)
    y = xi.to(tl.float32, bitcast=True).to(tl.float8e4nv)
    tl.store(y_ptr + offs, y, mask=mask)


def sr_fp8_software_triton(x, seed, block=1024):
    """Tiling-invariant software fp32 -> float8_e4m3fn stochastic rounding. `seed` is an on-device
    int32 tensor; `block` is exposed so callers can prove tile-invariance across launch sizes."""
    assert x.dtype == torch.float32, f"SR fp8 expects fp32 input, got {x.dtype}"
    assert x.is_contiguous()
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    n = x.numel()

    def grid(meta):
        return (triton.cdiv(n, 4 * meta["BLOCK"]),)

    _sr_fp8_software_kernel[grid](x, out, seed, n, BLOCK=block)
    return out


# --- software SR for float4_e2m1fn_x2 (packed nvfp4, two e2m1 codes per byte) --------------------
# fp4 e2m1 keeps a single mantissa bit, so fp32 -> fp4 drops the low 22 mantissa bits (drop =
# 23 - 1); the same software add-dither-then-truncate trick as bf16/fp8, just wider. Two differences
# from the fp8 kernel: (1) fp4 saturates at +-6 with no `.to` cast, so we clamp first, then encode the
# already-SR'd value to packed e2m1 with the Blackwell `cvt.rn.satfinite.e2m1x2.f32` intrinsic (via
# `_convert_fp32_to_fp4_packed`, the same hardware pack the repo's `_nvfp4_kernel` uses -- the cvt is
# plain round-to-nearest, so it is a no-op on the post-truncation grid points and bit-matches gold's
# software f32_to_f4_unpacked; the stochastic rounding is still done in software by the dither);
# (2) the output is PACKED two codes per byte (even element -> low nibble, odd -> high, matching gold
# `pack_uint4`), so the kernel stores n/2 bytes. The cvt.rn.satfinite.e2m1x2.f32 is Blackwell-only
# PTX, so unlike the bf16/fp8 software kernels this one requires cuda capability (10, 0).
@triton.jit
def _convert_fp32_to_fp4_packed(x_pairs):
    # copy of quant_cast_triton/recipes.py `_convert_fp32_to_fp4_packed` (itself verbatim from MSLK):
    # hardware fp32 -> packed fp4 e2m1 (RNE, saturating), two values per byte (first arg -> low nibble,
    # second -> high nibble). Blackwell-only (cvt.rn.satfinite.e2m1x2.f32).
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;
        cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints=("=r,r,r,r,r,r,r,r,r"),
        args=x_pairs,
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _sr_fp4_software_kernel(x_ptr, y_ptr, seed_ptr, n_elements, BLOCK: tl.constexpr):
    seed = tl.load(seed_ptr)  # on-device, no host sync
    pid = tl.program_id(0)
    grp = pid * BLOCK + tl.arange(0, BLOCK)  # (BLOCK,) group index = global flat index >> 2
    r0, r1, r2, r3 = tl.randint4x(seed, grp)  # 4 streams; the group's 4 elements each take one
    # contiguous (r0,r1,r2,r3) lane order so element 4c+lane == tl.randint4x(seed,c)[lane] ==
    # prng.bits(key,..,uint32)[4c+lane], letting api.py's eager reference bit-match with no permute.
    rand = tl.interleave(tl.interleave(r0, r2), tl.interleave(r1, r3))  # (4*BLOCK,) (r0,r1,r2,r3)
    offs = pid * (4 * BLOCK) + tl.arange(0, 4 * BLOCK)  # contiguous global flat indices
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    x = tl.minimum(tl.maximum(x, -6.0), 6.0)  # fp4 saturates at +-6 (before the dither, as gold does)
    xi = x.to(tl.int32, bitcast=True)
    rand22 = (rand & 0x3FFFFF).to(tl.int32)  # uniform 22-bit dither; randint4x is uint32
    xi = (xi + rand22) & -4194304  # add dither, then truncate the low 22 mantissa bits (0xFFC00000)
    x_sr = xi.to(tl.float32, bitcast=True)  # sits on the fp4 grid for normals -> the RNE cvt is a no-op
    # hardware e2m1 pack: reshape to pairs and split into (even, odd); even -> low nibble, odd -> high
    # (matches pack_uint4). Blackwell cvt.rn.satfinite.e2m1x2.f32.
    q = _convert_fp32_to_fp4_packed(tl.reshape(x_sr, (2 * BLOCK, 2)).split())  # (2*BLOCK,) packed bytes
    b = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)  # output byte indices
    tl.store(y_ptr + b, q, mask=b < (n_elements // 2))  # n even -> byte b holds elements 2b, 2b+1


def sr_fp4_software_triton(x, seed, block=1024):
    """Tiling-invariant fp32 -> float4_e2m1fn_x2 (packed nvfp4) stochastic rounding. The SR is software
    (dither + truncate); the final e2m1 pack uses the Blackwell `cvt.rn.satfinite.e2m1x2.f32`, so this
    requires cuda capability (10, 0). `seed` is an on-device int32 tensor; `block` is exposed so callers
    can prove tile-invariance across launch sizes. Output is packed two codes per byte (last dim halves;
    it must be even)."""
    assert x.dtype == torch.float32, f"SR fp4 expects fp32 input, got {x.dtype}"
    assert x.is_contiguous()
    assert x.shape[-1] % 2 == 0, "fp4 packs 2 codes per byte; the last dim must be even"
    n = x.numel()
    out = torch.empty(n // 2, dtype=torch.uint8, device=x.device)

    def grid(meta):
        return (triton.cdiv(n, 4 * meta["BLOCK"]),)

    _sr_fp4_software_kernel[grid](x, out, seed, n, BLOCK=block)
    return out.view(torch.float4_e2m1fn_x2).reshape(*x.shape[:-1], x.shape[-1] // 2)


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


# --- hardware SR for fp8_e4m3fn: copy of nvidia_rs_demo/api.py `_rs_quad32_kernel` for e4m3 ------
# The `cvt.rs.satfinite.e4m3x4.f32` intrinsic converts 4 f32 lanes -> one b32 (four packed e4m3),
# consuming one 32-bit `rbits` word (the hardware slices out 8 dither bits per element; fp32->fp8
# discards 20 mantissa bits, so unlike bf16 the hardware sees fewer bits than are dropped -- still
# valid SR, just coarser dither). `.satfinite` is mandatory for the x4 forms. Blackwell-only.
@triton.jit
def _sr_fp8_hardware_kernel(x_ptr, y_ptr, seed_ptr, num_groups, BLOCK: tl.constexpr):
    # BLOCK = input f32 elements per program. Each group is 4 elements needing one 32-bit rbits
    # word, and randint4x yields 4 words per counter, so this program needs:
    #     groups   = BLOCK // 4
    #     counters = groups // 4 = BLOCK // 16  (one randint4x call each, all 4 words used)
    seed = tl.load(seed_ptr)  # Philox seed, on-device (no host sync)
    pid = tl.program_id(0)
    ctr = pid * (BLOCK // 16) + tl.arange(0, BLOCK // 16)  # unique Philox counters for this program
    r0, r1, r2, r3 = tl.randint4x(seed, ctr)               # 4 words per counter, (BLOCK//16,) each
    # interleave lays the words down so 4 consecutive groups take one counter's 4 words: no word is
    # recomputed and none discarded. Element f -> group f>>2 -> counter f>>4 (independent of BLOCK).
    rbits = tl.interleave(tl.interleave(r0, r1), tl.interleave(r2, r3))  # (BLOCK//4,) one word/group
    g = pid * (BLOCK // 4) + tl.arange(0, BLOCK // 4)       # group index (each group = 4 elements)
    mask = g < num_groups
    base = g * 4
    # e4m3x4 packs {$1,$2,$3,$4} with $1 in the HIGH byte, so a little-endian view of the b32 output
    # is [$4,$3,$2,$1]. Load the 4 lanes reversed so the stored quad comes out in element order
    # [base+0, base+1, base+2, base+3] rather than reversed within the group.
    a = tl.load(x_ptr + base + 3, mask=mask)
    b = tl.load(x_ptr + base + 2, mask=mask)
    c = tl.load(x_ptr + base + 1, mask=mask)
    d = tl.load(x_ptr + base + 0, mask=mask)
    q = tl.inline_asm_elementwise(
        asm="cvt.rs.satfinite.e4m3x4.f32 $0, {$1, $2, $3, $4}, $5;",
        constraints="=r,f,f,f,f,r", args=[a, b, c, d, rbits],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    tl.store(y_ptr + g, q, mask=mask)


def sr_fp8_hardware_triton(x, seed, block=1024):
    """Tiling-invariant hardware fp32 -> float8_e4m3fn stochastic rounding via
    `cvt.rs.satfinite.e4m3x4.f32`. Blackwell-only (the caller must gate on cuda capability).
    `seed` is an on-device int32 tensor."""
    assert x.dtype == torch.float32 and x.is_contiguous()
    n = x.numel()
    assert n % 4 == 0, "e4m3x4 needs an element count divisible by 4"
    x = x.reshape(-1)
    num_groups = n // 4
    out_i32 = torch.empty(num_groups, dtype=torch.int32, device=x.device)
    grid = (triton.cdiv(n, block),)  # block = input elements per program
    _sr_fp8_hardware_kernel[grid](x, out_i32, seed, num_groups, BLOCK=block)
    return out_i32.view(torch.float8_e4m3fn).reshape(x.shape)
