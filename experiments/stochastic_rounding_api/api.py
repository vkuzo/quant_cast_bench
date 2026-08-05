"""A `to()` cast API prototype with three rounding modes, fp32 -> bf16 only.

    to(x, torch.bfloat16)                                  # RTNE (default)
    to(x, torch.bfloat16, rounding="stochastic", key=k)    # software SR, tile-invariant
    to(x, torch.bfloat16, rounding="stochastic-approx", key=k)  # hardware SR (Blackwell cvt.rs)

Rounding modes:
  * "rtne" (default)          -- round-to-nearest-even. This IS what `x.to(torch.bfloat16)` does, so
                                 we just call it; no randomness needed.
  * "stochastic"              -- unbiased stochastic rounding in software, tile-invariant (output
                                 independent of the kernel's block size). Portable to any GPU.
  * "stochastic-approx"       -- stochastic rounding in hardware via the Blackwell PTX intrinsic
                                 `cvt.rs.bf16x2.f32`. Faster, but gated to cuda capability (10, 0)
                                 and NOT reproducible from eager PyTorch (see reference.py). Named
                                 "approx" because the hardware reuses a fixed random-bit budget, so
                                 for the narrow formats (fp8/fp4, added later) it gets fewer bits
                                 than there are discarded mantissa bits. For bf16 specifically it is
                                 exact -- in fact bit-identical to "stochastic" given the same bits.

Randomness source (required for the two stochastic modes, exactly one of):
  * key=       -- a `torch.func._random` Philox key tensor (the functional, layout-independent,
                  reproducible precedent). Preferred for determinism.
  * generator= -- a `torch.Generator`. Convenience path: we draw a fresh seed from it, which ADVANCES
                  the generator, so repeated calls differ (like other stateful torch RNG ops).
"""

import torch

from kernels import sr_bf16_hardware_triton, sr_bf16_software_triton

RTNE = "rtne"
STOCHASTIC = "stochastic"
STOCHASTIC_APPROX = "stochastic-approx"

# cvt.rs emits Blackwell-only PTX; the hardware ("approx") mode is gated to this capability.
_APPROX_CAPABILITY = (10, 0)


def _seed_from_source(key, generator, device):
    """Resolve a randomness source to an on-device int32 seed tensor (no host sync).

    Exactly one of `key` / `generator` must be given. A `key` (torch.func._random Philox key) yields
    a deterministic seed via its first 32 bits; a `generator` yields a fresh seed each call (advancing
    its state)."""
    if (key is None) == (generator is None):
        raise ValueError("stochastic rounding needs exactly one of key= or generator=")
    if key is not None:
        # first 32 bits of the key, kept on-device (matches the repo idiom).
        return key.reshape(-1)[:1].view(torch.int32)
    # generator path: draw one int32 seed; this advances the generator so successive calls differ.
    return torch.randint(
        0, 2**31, (1,), generator=generator, dtype=torch.int32, device=device
    )


def to(x, dtype, *, rounding=RTNE, key=None, generator=None):
    """Cast `x` (fp32) to `dtype` (bfloat16) with the requested rounding mode. See module docstring."""
    assert x.dtype == torch.float32, f"only fp32 input supported for now, got {x.dtype}"
    assert dtype == torch.bfloat16, f"only bfloat16 output supported for now, got {dtype}"

    if rounding == RTNE:
        if key is not None or generator is not None:
            raise ValueError("rtne rounding is deterministic; do not pass key= or generator=")
        return x.to(torch.bfloat16)  # native cast already rounds to nearest-even

    seed = _seed_from_source(key, generator, x.device)

    if rounding == STOCHASTIC:
        return sr_bf16_software_triton(x, seed)

    if rounding == STOCHASTIC_APPROX:
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        if cap != _APPROX_CAPABILITY:
            raise RuntimeError(
                f"rounding='{STOCHASTIC_APPROX}' needs the cvt.rs intrinsic (Blackwell sm_100a, cuda "
                f"capability {_APPROX_CAPABILITY}); this device is {cap}. Use rounding='{STOCHASTIC}'."
            )
        return sr_bf16_hardware_triton(x, seed)

    raise ValueError(
        f"unknown rounding {rounding!r}; expected one of "
        f"{RTNE!r}, {STOCHASTIC!r}, {STOCHASTIC_APPROX!r}"
    )
