from enum import StrEnum

import torch
import torch.func._random as prng

from kernels import (
    sr_bf16_hardware_triton,
    sr_bf16_software_triton,
    sr_fp8_hardware_triton,
    sr_fp8_software_triton,
)


class Rounding(StrEnum):
    RTNE = "rtne"
    STOCHASTIC = "stochastic"
    STOCHASTIC_NVIDIA_SM100 = "stochastic-nvidia-sm100"


def to(
    x: torch.Tensor, 
    dtype: torch.dtype, 
    *, 
    rounding: Rounding=Rounding.RTNE, 
    key=None, 
    # debug only
    _reference_impl=False):
    """Cast `x` (fp32) to `dtype` with the requested rounding mode. 

    As this is a POC, input must be fp32 and output is
    bfloat16 or float8_e4m3fn.

    Examples::

        to(x, torch.bfloat16)                                                       # RTNE (default)
        to(x, torch.bfloat16, rounding=Rounding.STOCHASTIC, key=k)                  # software SR
        to(x, torch.float8_e4m3fn, rounding=Rounding.STOCHASTIC, key=k)             # software SR, fp8
        to(x, torch.bfloat16, rounding=Rounding.STOCHASTIC_NVIDIA_SM100, key=k)     # hardware SR
        to(x, torch.float8_e4m3fn, rounding=Rounding.STOCHASTIC_NVIDIA_SM100, key=k) # hardware SR, fp8

    Rounding modes:
      * RTNE -- round-to-nearest-even
      * STOCHASTIC -- unbiased stochastic rounding in software, tile-invariant.
                    requires a torch build with torch.func._random.bits. Backed by
                    a triton kernel.
      * STOCHASTIC_NVIDIA_SM100  -- stochastic rounding in hardware via the
                    NVIDIA Blackwell PTX intrinsics `cvt.rs.bf16x2.f32` (bf16) and
                    `cvt.rs.satfinite.e4m3x4.f32` (fp8). Backed by a triton kernel.

    Randomness source (required for the two stochastic modes):
      * key= -- a `torch.func._random` Philox key tensor.

    `_reference_impl` (debug only) computes the result with an eager-PyTorch reference, matches the triton kernels bitwise.
    """
    assert x.dtype == torch.float32, f"only fp32 input supported for now, got {x.dtype}"
    assert dtype in (torch.bfloat16, torch.float8_e4m3fn), (
        f"only bfloat16 / float8_e4m3fn output supported for now, got {dtype}"
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
            # number of mantissa bits to drop, by target dtype
            drop = 16 if dtype == torch.bfloat16 else 20

            # `prng.bits` output matches an in-kernel `tl.randint4x` output with a predefined
            # permutation as they call philox the same way, which is why we can bitwise match 
            # the randomness of eager vs triton
            b = prng.bits(key, x.numel(), dtype=torch.uint32).view(torch.int32).reshape(x.shape)

            # eager mode reference of software stochastic rounding, with 32 bits of 
            # randomness per source element
            rand = (b & ((1 << drop) - 1)).to(torch.int32)
            xi = (x.contiguous().view(torch.int32) + rand) & -(1 << drop)
            return xi.view(torch.float32).to(dtype)

        else:
            seed = key.reshape(-1)[:1].view(torch.int32)
            if dtype == torch.bfloat16:
                return sr_bf16_software_triton(x, seed)
            else:
                assert dtype == torch.float8_e4m3fn, "unsupported"
                return sr_fp8_software_triton(x, seed)

    if rounding == Rounding.STOCHASTIC_NVIDIA_SM100:
        if _reference_impl:
            # Bit-exact eager model of the cvt.rs.* intrinsic (matches the hardware kernel). Each
            # kernel feeds one 32-bit Philox word W per group (2 elements for cvt.rs.bf16x2.f32, 4 for
            # cvt.rs.satfinite.e4m3x4.f32) and the intrinsic hands each element a 16-bit slice of W,
            # reverse-engineered in experiments/nvidia_rs_bit_probe:
            #   bf16x2:  element base+0 <- W[0:15]            element base+1 <- W[16:31]
            #   e4m3x4:  element base+0 <- W[0:15]            element base+2 <- W[16:31]
            #            element base+1 <- reverse(W[0:15])   element base+3 <- reverse(W[16:31])
            # (bf16 reads its two halves in natural weight order; fp8 packs 4 elements into the same
            # word by reusing each half twice, once bit-reversed.) Both dtypes then use the SAME .rs
            # carry rule -- add the 16-bit dither R to the discarded mantissa field, round up on the
            # carry-out -- but with a discarded-field width D that lands truncation exactly on the
            # target grid: bf16 has a fixed D=16 (a plain add-and-truncate), while fp8 varies D per
            # element so it stays grid-correct for subnormals too (see the fp8 block). Both
            # kernels lay the 4 words of Philox counter c across groups
            # 4c,4c+2,4c+1,4c+3 (perm [0,2,1,3] within each block of 4 groups, from their shared
            # interleave(interleave(r0,r1),interleave(r2,r3))), so we index the words the same way.

            # get the correct number of 32-bit words of randomness
            flat = x.contiguous().reshape(-1)
            per_group = 2 if dtype == torch.bfloat16 else 4
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
            low, high = W & 0xFFFF, (W >> 16) & 0xFFFF
            if dtype == torch.bfloat16:
                R = torch.stack([low, high], dim=1).reshape(-1).double()  # per element, natural order
            else:
                low_rev = sum(((low >> i) & 1) << (15 - i) for i in range(16))    # bit-reverse low 16
                high_rev = sum(((high >> i) & 1) << (15 - i) for i in range(16))  # bit-reverse high 16
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

            else:
                assert dtype == torch.float8_e4m3fn

                # fp8_e4m3: same add-dither-truncate carry rule as bf16, but a FIXED 20-bit drop is wrong
                # for subnormals (it truncates onto a too-fine grid, then the cast double-rounds). Make the
                # drop width exponent-dependent so truncation lands exactly on the true fp8 grid: normals
                # discard D=20 fp32 mantissa bits, and each binade below the min normal exponent
                # (E_min=-6) needs one more -> D(E) = 20 + max(0, E_min - E). R aligns to the TOP of the
                # D-bit field (<< D-16), so round-up is the carry out of bit D. This is exact for
                # |x| >= 2^-9 (E >= -9 -> D <= 23). The bottom bin |x| < 2^-9 rounds between 0 and 2^-9,
                # which is not a mantissa truncation (0 needs a zero exponent), so it's a direct frac
                # compare masked in below. Out-of-range magnitudes saturate to +-448 (satfinite).

                # the intrinsic is satfinite, so we clamp to min/max repr value
                flat_c = flat.clamp(-448.0, 448.0)

                # get the unbiased fp32 exponent, E decides whether the value will
                # be normal or subnormal in the float8_e4m3fn encoding
                E = ((flat_c.abs().view(torch.int32).to(torch.int64) >> 23) & 0xFF) - 127  # fp32 exp of |x|

                # exponent-dependent drop width: incrementing D one bit per binade below E_min
                # freezes the truncation grid at the fixed 2^-9 subnormal ulp (spacing = 2^(E-23+D)),
                # which is what makes the dither work correctly for subnormals
                D = 20 + (-6 - E).clamp(min=0)          # per-element discarded-bit width

                # cap to a valid shift/mask width; D>23 only for |x|<2^-9, which is overridden below
                Dc = D.clamp(max=23)

                # add-dither-truncate, use int64 to prevent overflow in the negative mask
                # arithmetic and the shifts
                xbits = flat_c.view(torch.int32).to(torch.int64)
                xi = (xbits + (R.to(torch.int64) << (Dc - 16))) & -(torch.ones_like(Dc) << Dc)
                out = xi.to(torch.int32).view(torch.float32)  # sits exactly on the fp8 grid for |x| >= 2^-9

                # special case numbers next to the zero bin
                # bottom bin |x| < 2^-9: neighbors are 0 and 2^-9, frac = |x| / 2^-9; round away from zero
                ax = flat.abs().double()
                promote = ax * (1 << 9) + R / (1 << 16) >= 1.0  # frac + R/2^16 >= 1
                bottom = torch.sign(flat).double() * torch.where(promote, torch.full_like(ax, 2.0**-9), torch.zeros_like(ax))

                # combine the truncation path (`out`, |x| >= 2^-9) and the bottom-bin path
                # (`bottom`, |x| < 2^-9) with a mask
                out = torch.where(ax < 2.0**-9, bottom.to(torch.float32), out)

                # last round to target dtype, and we're done!
                return out.to(dtype).reshape(x.shape)
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
            return sr_fp8_hardware_triton(x, seed)  # cvt.rs.satfinite.e4m3x4.f32

    raise ValueError(f"unknown rounding {rounding!r}; expected one of {[r.value for r in Rounding]}")
