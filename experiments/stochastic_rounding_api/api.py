from enum import StrEnum

import torch
import torch.func._random as prng

from kernels import (
    sr_bf16_hardware_triton,
    sr_bf16_software_triton,
    sr_fp4_hardware_triton,
    sr_fp4_software_triton,
    sr_fp8_hardware_triton,
    sr_fp8_software_triton,
)
# fp4 has no direct `.to()` cast, so the eager reference reuses the gold's tested pure-PyTorch e2m1
# encode + pack (bit-matched vs the kernel's _f32_to_f4_code_tl in the main package).
from quant_cast_bench.quant_cast_gold.utils import f32_to_f4_unpacked, pack_uint4


class Rounding(StrEnum):
    RTNE = "rtne"
    STOCHASTIC = "stochastic"
    STOCHASTIC_NVIDIA_SM100 = "stochastic-nvidia-sm100"


def _rs_satfinite_x4_round(flat, R, *, maxval, mbits, exp_bias):
    """Shared eager model of the `.rs` carry-round used by both satfinite x4 cvt intrinsics
    (`cvt.rs.satfinite.e4m3x4.f32` for fp8_e4m3, `cvt.rs.satfinite.e2m1x4.f32` for fp4 e2m1). Given
    the fp32 tensor `flat` and its per-element 16-bit random slice `R` (already laid out by the
    caller from the shared 32-bit Philox word), returns the stochastically-rounded value ON THE fp32
    GRID of the target format -- the caller does the final dtype encode (fp8 casts, fp4 packs e2m1).

    The formats differ only in three e-format constants, so the whole numeric core is parametrized:
      * maxval    -- satfinite saturation magnitude (fp8: 448, fp4: 6).
      * mbits     -- kept mantissa bits -> normals discard drop = 23 - mbits fp32 mantissa bits
                     (fp8: 3 -> 20, fp4: 1 -> 22).
      * exp_bias  -- format exponent bias -> min normal unbiased exponent E_min = 1 - exp_bias
                     (fp8: 7 -> -6, fp4: 1 -> 0) and subnormal ulp = 2^(E_min - mbits)
                     (fp8: 2^-9, fp4: 2^-1).

    The .rs rule is add-dither-then-truncate with a carry-out round-up, but with a discarded-field
    width D chosen so truncation lands exactly on the target grid: normals discard `drop` bits, and
    each binade below E_min needs one more -> D(E) = drop + max(0, E_min - E). R aligns to the TOP of
    the D-bit field (<< D-16), so round-up is the carry out of bit D. This is exact while D <= 23
    (|x| >= subnormal_ulp). The bottom bin |x| < subnormal_ulp rounds between 0 and subnormal_ulp,
    which is not a mantissa truncation (0 needs a zero exponent), so it's a direct frac compare.
    Out-of-range magnitudes saturate to +-maxval (satfinite)."""
    e_min = 1 - exp_bias
    drop = 23 - mbits
    sub_ulp = 2.0 ** (e_min - mbits)

    # the intrinsic is satfinite, so we clamp to the max representable magnitude first
    flat_c = flat.clamp(-maxval, maxval)

    # fp32 exponent of |x|: decides whether the value is normal or subnormal in the target encoding
    E = ((flat_c.abs().view(torch.int32).to(torch.int64) >> 23) & 0xFF) - 127

    # exponent-dependent drop width freezes the truncation grid at the subnormal ulp for the
    # subnormal binades (spacing = 2^(E-23+D)); cap to a valid shift/mask width (D>23 only for the
    # bottom bin, which is overridden below)
    D = drop + (e_min - E).clamp(min=0)
    Dc = D.clamp(max=23)

    # add-dither-truncate in int64 (avoids overflow in the negative mask arithmetic and the shifts)
    xbits = flat_c.view(torch.int32).to(torch.int64)
    xi = (xbits + (R.to(torch.int64) << (Dc - 16))) & -(torch.ones_like(Dc) << Dc)
    out = xi.to(torch.int32).view(torch.float32)  # on the target grid for |x| >= subnormal_ulp

    # bottom bin |x| < subnormal_ulp: neighbors are 0 and subnormal_ulp, frac = |x| / subnormal_ulp;
    # round away from zero on frac + R/2^16 >= 1
    ax = flat.abs().double()
    promote = ax / sub_ulp + R / (1 << 16) >= 1.0
    bottom = torch.sign(flat).double() * torch.where(
        promote, torch.full_like(ax, sub_ulp), torch.zeros_like(ax)
    )
    return torch.where(ax < sub_ulp, bottom.to(torch.float32), out)


def to(
    x: torch.Tensor, 
    dtype: torch.dtype, 
    *, 
    rounding: Rounding=Rounding.RTNE, 
    key=None, 
    # debug only
    _reference_impl=False):
    """Cast `x` (fp32) to `dtype` with the requested rounding mode. 

    As this is a POC, input must be fp32 and output is bfloat16, float8_e4m3fn, or
    float4_e2m1fn_x2 (packed nvfp4, two e2m1 codes per byte -- the last dim halves, and it must be
    even). fp4 is stochastic-only (there is no `.to()` fp4 cast for RTNE), and both fp4 SR paths use a
    Blackwell e2m1 cvt intrinsic (software SR packs with cvt.rn.satfinite.e2m1x2.f32, hardware SR
    rounds with cvt.rs.satfinite.e2m1x4.f32), so fp4 needs cuda capability (10, 0) -- unlike the
    portable bf16/fp8 software-SR kernels.

    Examples::

        to(x, torch.bfloat16)                                                       # RTNE (default)
        to(x, torch.bfloat16, rounding=Rounding.STOCHASTIC, key=k)                  # software SR
        to(x, torch.float8_e4m3fn, rounding=Rounding.STOCHASTIC, key=k)             # software SR, fp8
        to(x, torch.float4_e2m1fn_x2, rounding=Rounding.STOCHASTIC, key=k)          # software SR, fp4
        to(x, torch.bfloat16, rounding=Rounding.STOCHASTIC_NVIDIA_SM100, key=k)     # hardware SR
        to(x, torch.float8_e4m3fn, rounding=Rounding.STOCHASTIC_NVIDIA_SM100, key=k) # hardware SR, fp8
        to(x, torch.float4_e2m1fn_x2, rounding=Rounding.STOCHASTIC_NVIDIA_SM100, key=k) # hardware SR, fp4

    Rounding modes:
      * RTNE -- round-to-nearest-even
      * STOCHASTIC -- unbiased stochastic rounding in software, tile-invariant.
                    requires a torch build with torch.func._random.bits. Backed by
                    a triton kernel.
      * STOCHASTIC_NVIDIA_SM100  -- stochastic rounding in hardware via the
                    NVIDIA Blackwell PTX intrinsics `cvt.rs.bf16x2.f32` (bf16),
                    `cvt.rs.satfinite.e4m3x4.f32` (fp8) and `cvt.rs.satfinite.e2m1x4.f32`
                    (fp4). Backed by a triton kernel.

    Randomness source (required for the two stochastic modes):
      * key= -- a `torch.func._random` Philox key tensor.

    `_reference_impl` (debug only) computes the result with an eager-PyTorch reference, matches the triton kernels bitwise.
    """
    assert x.dtype == torch.float32, f"only fp32 input supported for now, got {x.dtype}"
    assert dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float4_e2m1fn_x2), (
        f"only bfloat16 / float8_e4m3fn / float4_e2m1fn_x2 output supported for now, got {dtype}"
    )
    # float4_e2m1fn_x2 (packed nvfp4) is stochastic-only: there is no direct `.to()` fp4 cast for RTNE
    # (both SR modes encode e2m1 explicitly instead). Software SR packs with cvt.rn.satfinite.e2m1x2.f32
    # and hardware SR rounds with cvt.rs.satfinite.e2m1x4.f32 -- both need cuda capability (10, 0).
    if dtype == torch.float4_e2m1fn_x2 and rounding == Rounding.RTNE:
        raise NotImplementedError(
            f"float4_e2m1fn_x2 output has no RTNE cast; use rounding='{Rounding.STOCHASTIC}' or "
            f"'{Rounding.STOCHASTIC_NVIDIA_SM100}'"
        )

    if rounding == Rounding.RTNE:
        if key is not None:
            raise ValueError("rtne rounding is deterministic; do not pass key=")
        return x.to(dtype)

    if key is None:
        raise ValueError("stochastic rounding needs a key= (torch.func._random Philox key)")

    if rounding == Rounding.STOCHASTIC:
        if not hasattr(prng, "bits"):
            raise RuntimeError(
                "rounding='stochastic' requires the raw-uint32 Philox API torch.func._random.bits, "
                "which is what lets the eager reference bit-match the kernel; build/install a torch "
                "that includes https://github.com/pytorch/pytorch/pull/190253."
            )
        if _reference_impl:
            # number of mantissa bits to drop, by target dtype: bf16 keeps 7, fp8_e4m3 keeps 3, fp4
            # (e2m1) keeps 1, so fp32's 23 mantissa bits drop 16 / 20 / 22 respectively.
            drop = {torch.bfloat16: 16, torch.float8_e4m3fn: 20, torch.float4_e2m1fn_x2: 22}[dtype]

            # `prng.bits` output matches an in-kernel `tl.randint4x` output with a predefined
            # permutation as they call philox the same way, which is why we can bitwise match
            # the randomness of eager vs triton
            b = prng.bits(key, x.numel(), dtype=torch.uint32).view(torch.int32).reshape(x.shape)

            # fp4 saturates at +-6 (no `.to()` fp4 cast exists), so clamp before the dither exactly as
            # the gold _f32_to_packed_fp4_sr / the fp4 kernel do; bf16/fp8 rely on the final cast to
            # saturate, so they pass x through unclamped.
            xf = x.contiguous()
            if dtype == torch.float4_e2m1fn_x2:
                xf = xf.clamp(-6.0, 6.0).contiguous()

            # eager mode reference of software stochastic rounding, with 32 bits of
            # randomness per source element
            rand = (b & ((1 << drop) - 1)).to(torch.int32)
            xi = (xf.view(torch.int32) + rand) & -(1 << drop)
            x_sr = xi.view(torch.float32)  # sits on the target grid for normals (subnormals double-round)
            if dtype == torch.float4_e2m1fn_x2:
                # no `.to(fp4)`: RNE-encode the truncated value to e2m1 codes and pack two per byte
                # (the RNE is a no-op for normals). Bit-matches the kernel's _f32_to_f4_code_tl.
                return pack_uint4(f32_to_f4_unpacked(x_sr)).view(torch.float4_e2m1fn_x2)
            return x_sr.to(dtype)

        else:
            seed = key.reshape(-1)[:1].view(torch.int32)
            if dtype == torch.bfloat16:
                return sr_bf16_software_triton(x, seed)
            elif dtype == torch.float4_e2m1fn_x2:
                return sr_fp4_software_triton(x, seed)
            else:
                assert dtype == torch.float8_e4m3fn, "unsupported"
                return sr_fp8_software_triton(x, seed)

    if rounding == Rounding.STOCHASTIC_NVIDIA_SM100:
        if _reference_impl:
            # Bit-exact eager model of the cvt.rs.* intrinsic (matches the hardware kernel). Each
            # kernel feeds one 32-bit Philox word W per group (2 elements for cvt.rs.bf16x2.f32, 4 for
            # the x4 forms cvt.rs.satfinite.e4m3x4.f32 / .e2m1x4.f32) and the intrinsic hands each
            # element a 16-bit slice of W, reverse-engineered in experiments/nvidia_rs_bit_probe:
            #   bf16x2:  element base+0 <- W[0:15]            element base+1 <- W[16:31]
            #   e4m3x4:  element base+0 <- W[0:15]            element base+2 <- W[16:31]
            #            element base+1 <- reverse(W[0:15])   element base+3 <- reverse(W[16:31])
            #   e2m1x4:  same two-lanes-share-a-slice pattern, but the slices are byte-interleaved
            #            instead of contiguous 16-bit halves (fp4's probe finding). With bytes
            #            b0=W[0:7], b1=W[8:15], b2=W[16:23], b3=W[24:31] and rev() a per-byte
            #            bit-reverse, the four elements read:
            #              base+0 <- (b2<<8)|b0            base+2 <- (b3<<8)|b1
            #              base+1 <- (rev b0<<8)|rev b2    base+3 <- (rev b1<<8)|rev b3
            # (bf16 reads its two halves in natural weight order; the x4 forms pack 4 elements into the
            # same word by reusing each slice twice, once bit-reversed.) All dtypes then use the SAME
            # .rs carry rule -- add the 16-bit dither R to the discarded mantissa field, round up on
            # the carry-out -- but with a discarded-field width D that lands truncation exactly on the
            # target grid: bf16 has a fixed D=16 (a plain add-and-truncate), while fp8/fp4 vary D per
            # element so they stay grid-correct for subnormals too (see the fp8/fp4 blocks). All
            # kernels lay the 4 words of Philox counter c across groups
            # 4c,4c+2,4c+1,4c+3 (perm [0,2,1,3] within each block of 4 groups, from their shared
            # interleave(interleave(r0,r1),interleave(r2,r3))), so we index the words the same way.

            # get the correct number of 32-bit words of randomness
            flat = x.contiguous().reshape(-1)
            per_group = 2 if dtype == torch.bfloat16 else 4  # fp8/fp4 x4 both group 4 elements
            assert flat.numel() % per_group == 0, f"{dtype} SR needs numel divisible by {per_group}"
            groups = flat.numel() // per_group
            n_words = ((groups + 3) // 4) * 4  # round up to a whole Philox counter (4 words each)
            bits = prng.bits(key, n_words, dtype=torch.uint32).to(torch.int64)

            # rearrange the random 32-bit words to match the kernel's interleave order
            g = torch.arange(groups, device=flat.device)
            perm = torch.tensor([0, 2, 1, 3], device=flat.device)  # kernel's interleave order
            W = bits[4 * (g // 4) + perm[g % 4]]  # (groups,) one random word per group

            # reverse engineer the NVIDIA implementation of mapping the 32 bits 
            # of randomness to the inputs of `cvt.rs.bf16x2.f32` and `cvt.rs.satfinite.e4m3x4.f32`
            if dtype == torch.float4_e2m1fn_x2:
                # fp4 slices are byte-interleaved (see the layout comment). Split W into bytes, and
                # reverse the bits of a byte for the two lanes that read a slice bit-reversed.
                b0, b1 = W & 0xFF, (W >> 8) & 0xFF
                b2, b3 = (W >> 16) & 0xFF, (W >> 24) & 0xFF
                rev = lambda bb: sum(((bb >> i) & 1) << (7 - i) for i in range(8))  # per-byte reverse
                r_e0 = (b2 << 8) | b0                    # element base+0: bytes 2&0, natural
                r_e1 = (rev(b0) << 8) | rev(b2)          # element base+1: bytes 0&2, bit-reversed
                r_e2 = (b3 << 8) | b1                    # element base+2: bytes 3&1, natural
                r_e3 = (rev(b1) << 8) | rev(b3)          # element base+3: bytes 1&3, bit-reversed
                R = torch.stack([r_e0, r_e1, r_e2, r_e3], dim=1).reshape(-1).double()
            else:
                low, high = W & 0xFFFF, (W >> 16) & 0xFFFF
                if dtype == torch.bfloat16:
                    R = torch.stack([low, high], dim=1).reshape(-1).double()  # per element, natural
                else:
                    low_rev = sum(((low >> i) & 1) << (15 - i) for i in range(16))    # reverse low 16
                    high_rev = sum(((high >> i) & 1) << (15 - i) for i in range(16))  # reverse high 16
                    R = torch.stack([low, low_rev, high, high_rev], dim=1).reshape(-1).double()

            # rounding logic that consumes the per-element 16-bit random slice R
            if dtype == torch.bfloat16:
                # bf16: R fills the full 16 discarded mantissa bits, so the cvt.rs additive-carry rule
                # is exactly add-dither-then-truncate (same trick as the software branch, just fed the
                # hardware's shared-word bit layout). Round-up = carry out of the low 16 bits into the
                # kept mantissa. Exact for bf16 normals; randn-scale data never produces bf16
                # subnormals (~2^-133) where a fixed 16-bit drop would diverge.
                xi = (flat.view(torch.int32) + R.to(torch.int32)) & -(1 << 16)
                return xi.view(torch.float32).to(torch.bfloat16).reshape(x.shape)

            elif dtype == torch.float8_e4m3fn:
                # fp8_e4m3: mantissa=3, exp bias 7 (E_min=-6), saturates at +-448. The exponent-
                # dependent .rs carry round is the shared model in _rs_satfinite_x4_round; the final
                # `.to(fp8)` cast is exact on the returned grid points (|x| >= 2^-9 already has <=3
                # mantissa bits; the bottom bin returns 0 or 2^-9 exactly).
                out = _rs_satfinite_x4_round(flat, R, maxval=448.0, mbits=3, exp_bias=7)
                return out.to(dtype).reshape(x.shape)

            else:
                assert dtype == torch.float4_e2m1fn_x2
                # fp4 e2m1: mantissa=1, exp bias 1 (E_min=0), saturates at +-6 -- same shared carry
                # round as fp8, just those three format constants. fp4 has no `.to()` cast, so RNE-encode
                # the on-grid result to e2m1 codes and pack two per byte (the RNE is a no-op on grid
                # points), bit-matching the software reference / gold f32_to_f4_unpacked.
                out = _rs_satfinite_x4_round(flat, R, maxval=6.0, mbits=1, exp_bias=1)
                packed = pack_uint4(f32_to_f4_unpacked(out)).view(torch.float4_e2m1fn_x2)
                return packed.reshape(*x.shape[:-1], x.shape[-1] // 2)
        else:
            cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
            if cap != (10, 0):
                raise RuntimeError(
                    f"rounding='{Rounding.STOCHASTIC_NVIDIA_SM100}' needs the cvt.rs intrinsic (NVIDIA "
                    f"Blackwell sm_100a, cuda capability (10, 0)); this device is {cap}. "
                    f"Use rounding='{Rounding.STOCHASTIC}'."
                )
            seed = key.reshape(-1)[:1].view(torch.int32)  # on-device int32 seed for the kernel, no sync
            if dtype == torch.bfloat16:
                return sr_bf16_hardware_triton(x, seed)  # cvt.rs.bf16x2.f32
            elif dtype == torch.float4_e2m1fn_x2:
                return sr_fp4_hardware_triton(x, seed)  # cvt.rs.satfinite.e2m1x4.f32
            return sr_fp8_hardware_triton(x, seed)  # cvt.rs.satfinite.e4m3x4.f32

    raise ValueError(f"unknown rounding {rounding!r}; expected one of {[r.value for r in Rounding]}")
