from enum import StrEnum

import torch
import torch.func._random as prng

from kernels import sr_bf16_hardware_triton, sr_bf16_software_triton


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
    """Cast `x` (fp32) to `dtype` (bfloat16) with the requested rounding mode. fp32 -> bf16 only.

    Examples::

        to(x, torch.bfloat16)                                                   # RTNE (default)
        to(x, torch.bfloat16, rounding=Rounding.STOCHASTIC, key=k)              # software SR
        to(x, torch.bfloat16, rounding=Rounding.STOCHASTIC_NVIDIA_SM100, key=k) # hardware SR (sm_100)

    Rounding modes (a `Rounding` enum member; the plain string value also works):
      * RTNE ("rtne", default)   -- round-to-nearest-even. This IS what `x.to(torch.bfloat16)` does,
                                    so we just call it; no randomness needed.
      * STOCHASTIC ("stochastic") -- unbiased stochastic rounding in software, tile-invariant (output
                                    independent of the kernel's block size). Portable to any GPU.
      * STOCHASTIC_NVIDIA_SM100 ("stochastic-nvidia-sm100") -- stochastic rounding in hardware via the
                                    NVIDIA Blackwell PTX intrinsic `cvt.rs.bf16x2.f32`. Faster, but
                                    gated to cuda capability (10, 0) and NOT reproducible from eager
                                    PyTorch (see `_reference_impl` below). The hardware reuses a fixed
                                    random-bit budget, so for the narrow formats (fp8/fp4, added later)
                                    it gets fewer bits than there are discarded mantissa bits; for bf16
                                    specifically it is exact -- in fact bit-identical to STOCHASTIC
                                    given the same bits.

    Randomness source (required for the two stochastic modes):
      * key= -- a `torch.func._random` Philox key tensor (the functional, layout-independent,
                reproducible precedent).

    `_reference_impl` (debug only) computes the result with an eager-PyTorch reference instead of the
    Triton kernel, to cross-check the kernel's logic. For STOCHASTIC it uses a different RNG (a float
    uniform's top 16 bits, vs the kernel's raw randint4x low 16 bits with a different counter->element
    layout), so it is statistically equal but NOT bit-identical (see experiments/prng_match/). For
    STOCHASTIC_NVIDIA_SM100 there is no eager equivalent -- the rounding happens inside the PTX
    intrinsic and cannot be reproduced without inline_asm -- so it raises NotImplementedError.
    """
    assert x.dtype == torch.float32, f"only fp32 input supported for now, got {x.dtype}"
    assert dtype == torch.bfloat16, f"only bfloat16 output supported for now, got {dtype}"

    if rounding == Rounding.RTNE:
        if key is not None:
            raise ValueError("rtne rounding is deterministic; do not pass key=")
        return x.to(torch.bfloat16)  # native cast already rounds to nearest-even (both impls agree)

    if key is None:
        raise ValueError("stochastic rounding needs a key= (torch.func._random Philox key)")

    if rounding == Rounding.STOCHASTIC:
        if _reference_impl:
            # eager reference: draw a FLOAT uniform per element keyed on its GLOBAL flat index
            # (tile-invariant, reproducible), take its top 16 fractional bits as the dither, add to
            # the fp32 bits and truncate the low 16 (mask -65536 == 0xFFFF0000; the cast is then
            # exact). Statistically equal to the kernel but not bit-identical -- the kernel uses the
            # LOW 16 bits of a raw randint4x uint32 with a different layout.
            seed = key.reshape(-1)[:1].to(torch.int64)
            gidx = torch.arange(x.numel(), device=x.device, dtype=torch.int64)
            keys = torch.stack([seed.expand(gidx.numel()), gidx], dim=-1).to(torch.uint64)
            u = prng.uniform(keys, (gidx.numel(),)).reshape(x.shape)  # float uniform in [0, 1)
            rand16 = (u * (1 << 16)).to(torch.int32)
            xi = (x.contiguous().view(torch.int32) + rand16) & -65536
            return xi.view(torch.float32).to(torch.bfloat16)
        seed = key.reshape(-1)[:1].view(torch.int32)  # on-device int32 seed for the kernel, no sync
        return sr_bf16_software_triton(x, seed)

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
        return sr_bf16_hardware_triton(x, seed)

    raise ValueError(f"unknown rounding {rounding!r}; expected one of {[r.value for r in Rounding]}")
