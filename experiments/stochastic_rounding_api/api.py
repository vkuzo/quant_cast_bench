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
    For STOCHASTIC_NVIDIA_SM100 there is no eager equivalent -- the rounding happens inside the PTX
    intrinsic and cannot be reproduced without inline_asm -- so it raises NotImplementedError.
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
            raise NotImplementedError(
                "hardware cvt.rs stochastic rounding rounds inside the PTX instruction and is not "
                "reproducible in eager PyTorch without inline_asm; there is no reference impl for "
                "rounding='stochastic-nvidia-sm100'."
            )
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
