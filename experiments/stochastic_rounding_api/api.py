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
    _reference_impl=False):
    """Cast `x` (fp32) to `dtype` with the requested rounding mode. Input must be fp32; output is
    bfloat16 or float8_e4m3fn (every mode supports both).

    Examples::

        to(x, torch.bfloat16)                                                       # RTNE (default)
        to(x, torch.bfloat16, rounding=Rounding.STOCHASTIC, key=k)                  # software SR
        to(x, torch.float8_e4m3fn, rounding=Rounding.STOCHASTIC, key=k)             # software SR, fp8
        to(x, torch.bfloat16, rounding=Rounding.STOCHASTIC_NVIDIA_SM100, key=k)     # hardware SR
        to(x, torch.float8_e4m3fn, rounding=Rounding.STOCHASTIC_NVIDIA_SM100, key=k) # hardware SR, fp8

    Rounding modes (a `Rounding` enum member; the plain string value also works):
      * RTNE ("rtne", default)   -- round-to-nearest-even. This IS what `x.to(torch.bfloat16)` does,
                                    so we just call it; no randomness needed.
      * STOCHASTIC ("stochastic") -- unbiased stochastic rounding in software, tile-invariant (output
                                    independent of the kernel's block size). Portable to any GPU, but
                                    requires a torch build with torch.func._random.bits (see below).
      * STOCHASTIC_NVIDIA_SM100 ("stochastic-nvidia-sm100") -- stochastic rounding in hardware via the
                                    NVIDIA Blackwell PTX intrinsics `cvt.rs.bf16x2.f32` (bf16) and
                                    `cvt.rs.satfinite.e4m3x4.f32` (fp8). Faster, but gated to cuda
                                    capability (10, 0) and NOT reproducible from eager PyTorch (see
                                    `_reference_impl` below). The hardware slices a fixed per-element
                                    dither out of one 32-bit word (16 bits for bf16, 8 for each fp8
                                    lane), so fp8 gets fewer bits than the 20 it discards -- still
                                    valid SR, just coarser dither than the software path.

    Randomness source (required for the two stochastic modes):
      * key= -- a `torch.func._random` Philox key tensor (the functional, layout-independent,
                reproducible precedent).

    `_reference_impl` (debug only) computes the result with an eager-PyTorch reference instead of the
    Triton kernel, to cross-check it. For STOCHASTIC the reference is BIT-IDENTICAL to the kernel: it
    reads Philox's raw uint32 words via torch.func._random.bits -- the same words the kernel's
    randint4x lays across elements -- and dithers with their low 16 bits (see experiments/prng_match/).
    For STOCHASTIC_NVIDIA_SM100 the float8_e4m3fn reference is ALSO bit-identical: it reproduces the
    cvt.rs.satfinite.e4m3x4.f32 intrinsic in eager PyTorch from its reverse-engineered random-bit
    layout (each element's 16-bit slice of the shared word, plus the additive round-up rule; see
    experiments/nvidia_rs_bit_probe). The bf16 (cvt.rs.bf16x2.f32) reference is not implemented and
    raises NotImplementedError.
    """
    assert x.dtype == torch.float32, f"only fp32 input supported for now, got {x.dtype}"
    assert dtype in (torch.bfloat16, torch.float8_e4m3fn), (
        f"only bfloat16 / float8_e4m3fn output supported for now, got {dtype}"
    )

    if rounding == Rounding.RTNE:
        if key is not None:
            raise ValueError("rtne rounding is deterministic; do not pass key=")
        return x.to(dtype)  # native cast already rounds to nearest-even (both impls agree)

    if key is None:
        raise ValueError("stochastic rounding needs a key= (torch.func._random Philox key)")

    if rounding == Rounding.STOCHASTIC:
        if not hasattr(prng, "bits"):
            raise RuntimeError(
                "rounding='stochastic' requires the raw-uint32 Philox API torch.func._random.bits, "
                "which is what lets the eager reference bit-match the kernel; build/install a torch "
                "that includes https://github.com/pytorch/pytorch/pull/190253."
            )
        # fp32 keeps its top mantissa bits and dithers+truncates the rest: bf16 keeps 7 of the 23
        # mantissa bits (drops 16), float8_e4m3fn keeps 3 (drops 20). Same trick, wider dither.
        drop = 16 if dtype == torch.bfloat16 else 20
        if _reference_impl:
            # eager reference, BIT-IDENTICAL to the kernel (see experiments/prng_match). `bits`
            # exposes Philox's raw uint32 words in the same contiguous layout the kernel lays across
            # elements (word 4c+lane == tl.randint4x(seed,c)[lane]), so element f's dither is just
            # bits[f]. Take its low `drop` bits, add to the fp32 bits, truncate the low `drop` bits.
            # The cast is exact for bf16 and fp8 normals; for fp8 subnormals it rounds the kept value
            # onto the coarser grid but stays two-neighbor/unbiased (matches the kernel bitwise).
            b = prng.bits(key, x.numel(), dtype=torch.uint32).view(torch.int32).reshape(x.shape)
            rand = (b & ((1 << drop) - 1)).to(torch.int32)
            xi = (x.contiguous().view(torch.int32) + rand) & -(1 << drop)
            return xi.view(torch.float32).to(dtype)
        seed = key.reshape(-1)[:1].view(torch.int32)  # on-device int32 seed for the kernel, no sync
        if dtype == torch.bfloat16:
            return sr_bf16_software_triton(x, seed)
        return sr_fp8_software_triton(x, seed)

    if rounding == Rounding.STOCHASTIC_NVIDIA_SM100:
        if _reference_impl:
            if dtype != torch.float8_e4m3fn:
                raise NotImplementedError(
                    "the eager reference for rounding='stochastic-nvidia-sm100' is only implemented "
                    "for float8_e4m3fn (from the reverse-engineered cvt.rs.satfinite.e4m3x4.f32 "
                    "random-bit layout, see experiments/nvidia_rs_bit_probe); the bf16 "
                    "cvt.rs.bf16x2.f32 layout has no eager equivalent here."
                )
            # Bit-exact eager model of cvt.rs.satfinite.e4m3x4.f32 (matches sr_fp8_hardware_triton).
            # The kernel feeds one 32-bit Philox word W per group of 4 elements; the intrinsic hands
            # each element a 16-bit slice of W (reverse-engineered in experiments/nvidia_rs_bit_probe):
            #   element base+0 <- W[0:15]               element base+2 <- W[16:31]
            #   element base+1 <- reverse(W[0:15])      element base+3 <- reverse(W[16:31])
            # and rounds AWAY from zero iff  frac + R/2^16 >= 1, where frac is |x|'s fractional position
            # between its two bracketing fp8 grid points (the additive .rs model; holds for normals and
            # subnormals alike -- only the grid spacing differs). The kernel lays the 4 words of Philox
            # counter c across groups 4c,4c+2,4c+1,4c+3 (perm [0,2,1,3] within each block of 4 groups,
            # from its interleave(interleave(r0,r1),interleave(r2,r3))), so we index bits the same way.
            flat = x.contiguous().reshape(-1)
            n = flat.numel()
            assert n % 4 == 0, "e4m3x4 needs an element count divisible by 4"
            groups = n // 4
            n_words = ((groups + 3) // 4) * 4  # round up to a whole Philox counter (4 words each)
            bits = prng.bits(key, n_words, dtype=torch.uint32).to(torch.int64)
            g = torch.arange(groups, device=flat.device)
            perm = torch.tensor([0, 2, 1, 3], device=flat.device)  # kernel's interleave order
            W = bits[4 * (g // 4) + perm[g % 4]]  # (groups,) one random word per group
            low, high = W & 0xFFFF, (W >> 16) & 0xFFFF
            low_rev = sum(((low >> i) & 1) << (15 - i) for i in range(16))   # bit-reverse low 16 bits
            high_rev = sum(((high >> i) & 1) << (15 - i) for i in range(16))  # bit-reverse high 16 bits
            R = torch.stack([low, low_rev, high, high_rev], dim=1).reshape(-1).double()  # per element
            # |x|'s two bracketing points on the finite non-negative e4m3fn grid (0 .. 448)
            codes = torch.arange(256, device=flat.device, dtype=torch.uint8).view(dtype).float()
            grid = torch.unique(codes[torch.isfinite(codes) & (codes >= 0)]).sort().values.double()
            ax = flat.abs().double()
            i = (torch.searchsorted(grid, ax, right=True) - 1).clamp(0, grid.numel() - 2)
            frac = (ax - grid[i]) / (grid[i + 1] - grid[i])
            mag = torch.where(frac + R / (1 << 16) >= 1.0, grid[i + 1], grid[i])  # away from zero on carry
            return (torch.sign(flat) * mag).to(torch.float32).to(dtype).reshape(x.shape)
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
