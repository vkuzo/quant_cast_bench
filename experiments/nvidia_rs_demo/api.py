"""Demonstrate NVIDIA's hardware stochastic-rounding conversion intrinsic (PTX `cvt.rs`) by
inlining each variant into a Triton kernel, one kernel per intrinsic.

`.rs` is a rounding mode on the PTX `cvt` instruction: it rounds toward or away from zero based
on the carry-out of adding a supplied 32-bit random operand (`rbits`) to the discarded mantissa
bits of an f32 input. See PTX ISA sec 9.7.9.22. `.rs` requires sm_100a/sm_103a (Blackwell / B200).

Source is always .f32. The variants demonstrated here (fp6 skipped):

    cvt.rs.f16x2.f32              2 f32 -> 2 fp16       (rbits)
    cvt.rs.bf16x2.f32            2 f32 -> 2 bf16       (rbits)
    cvt.rs.satfinite.e4m3x4.f32   4 f32 -> 4 fp8 e4m3   (rbits, .satfinite mandatory)
    cvt.rs.satfinite.e5m2x4.f32   4 f32 -> 4 fp8 e5m2   (rbits, .satfinite mandatory)
    cvt.rs.satfinite.e2m1x4.f32   4 f32 -> 4 fp4 e2m1   (rbits, .satfinite mandatory)

Randomness comes from Triton's `tl.randint4x`, which returns four independent uint32 Philox words
per counter. We use one word as the `rbits` operand for each group; the hardware slices out the
per-element dither bits (13 for fp16, 16 for bf16, 8 each for the x4 narrow-float lanes).

Design: each intrinsic packs G consecutive f32 (G=2 or 4) into one small integer register. We use
`pack=1` and load the G lanes as separate tensors, so one asm invocation converts exactly one
group into one packed integer output (b32 for f16x2/bf16x2/fp8x4, b16 for fp4x4). The host then
`.view()`s that integer buffer as the real narrow dtype.
"""

import torch
import triton
import triton.language as tl

# Input f32 elements processed per program. Must be divisible by 16 so the derived group and
# Philox-counter arange lengths (BLOCK//2, BLOCK//4, BLOCK//8, BLOCK//16) stay whole powers of two.
BLOCK = 1024


# --- G=2 (f16x2 / bf16x2): 2 f32 lanes -> one b32 (two packed 16-bit values) --------------------
@triton.jit
def _rs_pair_kernel(
    x_ptr, y_ptr, seed_ptr, num_groups,
    ASM: tl.constexpr, CONSTR: tl.constexpr, BLOCK: tl.constexpr,
):
    # BLOCK = input f32 elements per program. Each group is 2 elements needing one 32-bit rbits
    # word, and randint4x yields 4 words per counter, so this program needs:
    #     groups   = BLOCK // 2
    #     counters = groups // 4 = BLOCK // 8   (one randint4x call each, all 4 words used)
    seed = tl.load(seed_ptr)  # Philox seed, on-device (no host sync)
    pid = tl.program_id(0)
    ctr = pid * (BLOCK // 8) + tl.arange(0, BLOCK // 8)  # unique Philox counters for this program
    r0, r1, r2, r3 = tl.randint4x(seed, ctr)             # 4 words per counter, (BLOCK//8,) each
    # interleave lays the words down so 4 consecutive groups take one counter's 4 words: no word is
    # recomputed and none discarded.
    rbits = tl.interleave(tl.interleave(r0, r1), tl.interleave(r2, r3))  # (BLOCK//2,) one word/group
    g = pid * (BLOCK // 2) + tl.arange(0, BLOCK // 2)     # group index (each group = 2 elements)
    mask = g < num_groups
    base = g * 2
    # a -> upper half of the packed result, b -> lower half (little-endian view yields [b, a]);
    # load a from the odd lane and b from the even lane so the stored pair is [even, odd].
    a = tl.load(x_ptr + base + 1, mask=mask)
    b = tl.load(x_ptr + base + 0, mask=mask)
    q = tl.inline_asm_elementwise(
        asm=ASM, constraints=CONSTR, args=[a, b, rbits],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    tl.store(y_ptr + g, q, mask=mask)


# --- G=4 (e4m3x4 / e5m2x4 / e2m1x4): 4 f32 lanes -> one packed integer ---------------------------
# Two variants differing only in the output register width: b32 (4 fp8) vs b16 (4 fp4). The
# `dtype` argument to `inline_asm_elementwise` must be a literal tl dtype, so it can't be threaded
# through a constexpr kernel parameter -- hence the two near-identical kernels.
@triton.jit
def _rs_quad32_kernel(
    x_ptr, y_ptr, seed_ptr, num_groups,
    ASM: tl.constexpr, CONSTR: tl.constexpr, BLOCK: tl.constexpr,
):
    # BLOCK = input f32 elements per program. Each group is 4 elements needing one 32-bit rbits
    # word, and randint4x yields 4 words per counter, so this program needs:
    #     groups   = BLOCK // 4
    #     counters = groups // 4 = BLOCK // 16  (one randint4x call each, all 4 words used)
    seed = tl.load(seed_ptr)
    pid = tl.program_id(0)
    ctr = pid * (BLOCK // 16) + tl.arange(0, BLOCK // 16)  # unique Philox counters for this program
    r0, r1, r2, r3 = tl.randint4x(seed, ctr)               # 4 words per counter, (BLOCK//16,) each
    rbits = tl.interleave(tl.interleave(r0, r1), tl.interleave(r2, r3))  # (BLOCK//4,) one word/group
    g = pid * (BLOCK // 4) + tl.arange(0, BLOCK // 4)       # group index (each group = 4 elements)
    mask = g < num_groups
    base = g * 4
    a = tl.load(x_ptr + base + 0, mask=mask)
    b = tl.load(x_ptr + base + 1, mask=mask)
    c = tl.load(x_ptr + base + 2, mask=mask)
    d = tl.load(x_ptr + base + 3, mask=mask)
    q = tl.inline_asm_elementwise(
        asm=ASM, constraints=CONSTR, args=[a, b, c, d, rbits],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    tl.store(y_ptr + g, q, mask=mask)


@triton.jit
def _rs_quad16_kernel(
    x_ptr, y_ptr, seed_ptr, num_groups,
    ASM: tl.constexpr, CONSTR: tl.constexpr, BLOCK: tl.constexpr,
):
    # BLOCK = input f32 elements per program (see _rs_quad32_kernel): groups = BLOCK // 4,
    # counters = BLOCK // 16.
    seed = tl.load(seed_ptr)
    pid = tl.program_id(0)
    ctr = pid * (BLOCK // 16) + tl.arange(0, BLOCK // 16)  # unique Philox counters for this program
    r0, r1, r2, r3 = tl.randint4x(seed, ctr)               # 4 words per counter, (BLOCK//16,) each
    rbits = tl.interleave(tl.interleave(r0, r1), tl.interleave(r2, r3))  # (BLOCK//4,) one word/group
    g = pid * (BLOCK // 4) + tl.arange(0, BLOCK // 4)       # group index (each group = 4 elements)
    mask = g < num_groups
    base = g * 4
    a = tl.load(x_ptr + base + 0, mask=mask)
    b = tl.load(x_ptr + base + 1, mask=mask)
    c = tl.load(x_ptr + base + 2, mask=mask)
    d = tl.load(x_ptr + base + 3, mask=mask)
    q = tl.inline_asm_elementwise(
        asm=ASM, constraints=CONSTR, args=[a, b, c, d, rbits],
        dtype=tl.int16, is_pure=True, pack=1,
    )
    tl.store(y_ptr + g, q, mask=mask)


def _launch_pair(x, seed, asm, constraints):
    assert x.dtype == torch.float32 and x.is_contiguous()
    n = x.numel()
    assert n % 2 == 0, "f16x2/bf16x2 need an even element count"
    x = x.reshape(-1)
    num_groups = n // 2
    out_i32 = torch.empty(num_groups, dtype=torch.int32, device=x.device)
    grid = (triton.cdiv(n, BLOCK),)  # BLOCK = input elements per program
    _rs_pair_kernel[grid](
        x, out_i32, seed, num_groups,
        ASM=asm, CONSTR=constraints, BLOCK=BLOCK,
    )
    return out_i32


def _launch_quad(x, seed, asm, constraints, out_dt):
    assert x.dtype == torch.float32 and x.is_contiguous()
    n = x.numel()
    assert n % 4 == 0, "the x4 intrinsics need an element count divisible by 4"
    x = x.reshape(-1)
    num_groups = n // 4
    out = torch.empty(num_groups, dtype=out_dt, device=x.device)
    kernel = _rs_quad16_kernel if out_dt == torch.int16 else _rs_quad32_kernel
    grid = (triton.cdiv(n, BLOCK),)  # BLOCK = input elements per program
    kernel[grid](
        x, out, seed, num_groups,
        ASM=asm, CONSTR=constraints, BLOCK=BLOCK,
    )
    return out


# --- public API: one function per intrinsic -----------------------------------------------------
def rs_f16(x, seed):
    """f32 -> fp16 with stochastic rounding via `cvt.rs.f16x2.f32`."""
    out_i32 = _launch_pair(
        x, seed,
        asm="cvt.rs.f16x2.f32 $0, $1, $2, $3;",
        constraints="=r,f,f,r",
    )
    return out_i32.view(torch.float16).reshape(x.shape)


def rs_bf16(x, seed):
    """f32 -> bf16 with stochastic rounding via `cvt.rs.bf16x2.f32`."""
    out_i32 = _launch_pair(
        x, seed,
        asm="cvt.rs.bf16x2.f32 $0, $1, $2, $3;",
        constraints="=r,f,f,r",
    )
    return out_i32.view(torch.bfloat16).reshape(x.shape)


def rs_e4m3(x, seed):
    """f32 -> fp8 e4m3 with stochastic rounding via `cvt.rs.satfinite.e4m3x4.f32`."""
    out_i32 = _launch_quad(
        x, seed,
        asm="cvt.rs.satfinite.e4m3x4.f32 $0, {$1, $2, $3, $4}, $5;",
        constraints="=r,f,f,f,f,r",
        out_dt=torch.int32,
    )
    return out_i32.view(torch.float8_e4m3fn).reshape(x.shape)


def rs_e5m2(x, seed):
    """f32 -> fp8 e5m2 with stochastic rounding via `cvt.rs.satfinite.e5m2x4.f32`."""
    out_i32 = _launch_quad(
        x, seed,
        asm="cvt.rs.satfinite.e5m2x4.f32 $0, {$1, $2, $3, $4}, $5;",
        constraints="=r,f,f,f,f,r",
        out_dt=torch.int32,
    )
    return out_i32.view(torch.float8_e5m2).reshape(x.shape)


def rs_e2m1(x, seed):
    """f32 -> fp4 e2m1 with stochastic rounding via `cvt.rs.satfinite.e2m1x4.f32`.

    Returns a packed `float4_e2m1fn_x2` tensor of shape (..., N//2) (two fp4 per byte). The b16
    result (4 fp4) is stored as int16, reviewed as uint8, then as the packed fp4 dtype.
    """
    out_i16 = _launch_quad(
        x, seed,
        asm="cvt.rs.satfinite.e2m1x4.f32 $0, {$1, $2, $3, $4}, $5;",
        constraints="=h,f,f,f,f,r",
        out_dt=torch.int16,
    )
    packed = out_i16.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    shape = x.shape[:-1] + (x.shape[-1] // 2,)
    return packed.reshape(shape)
