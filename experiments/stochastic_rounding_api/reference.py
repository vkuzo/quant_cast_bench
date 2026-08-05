"""Eager PyTorch reference for `api.to`, matching the *logic* of the kernels (not their bits).

`to_reference` mirrors the two non-hardware modes of `api.to`:

  * "rtne"       -- native `x.to(torch.bfloat16)`, bit-identical to the API.
  * "stochastic" -- the SAME software algorithm as the Triton kernel: key a uniform 16-bit dither on
                    each element's GLOBAL flat index, add it to the fp32 bits, then truncate the low
                    16 mantissa bits. Tile-invariant and reproducible from a key.

    IMPORTANT -- this is NOT bit-identical to the Triton "stochastic" kernel, only statistically
    equal. PyTorch exposes no raw integer randomness, so the eager path draws a FLOAT uniform in
    [0, 1) with `torch.func._random.uniform` and scales it (taking the TOP 16 fractional bits),
    whereas the kernel takes the LOW 16 bits of a raw `tl.randint4x` uint32 with a different
    counter->element layout. Both are correct unbiased SR; they just don't agree bit-for-bit. (See
    experiments/prng_match/ for the detailed why.)

  * "stochastic-approx" -- has NO eager reference: the hardware `cvt.rs` intrinsic rounds inside the
                         PTX instruction, which cannot be reproduced in eager PyTorch without
                         inline_asm. Raises NotImplementedError. (For bf16 the hardware result is
                         actually bit-identical to the software "stochastic" mode given the same
                         random bits -- but eager PyTorch cannot produce those integer bits.)
"""

import torch
import torch.func._random as prng


def _sr_bf16_dither(x, rand16):
    """Apply a uniform 16-bit dither `rand16` to `x` (fp32) then truncate to bf16.

    dither the 16 mantissa bits fp32->bf16 drops, then truncate them (mask off the low 16
    bits). -65536 == 0xFFFF0000 as int32; .to(bfloat16) is exact since the low bits are zero.
    """
    xi = x.contiguous().view(torch.int32) + rand16
    xi = xi & -65536
    return xi.view(torch.float32).to(torch.bfloat16)


def _seed_int64(key, generator, device):
    """Resolve key/generator to a scalar int64 seed tensor (shape (1,)) for building Philox keys."""
    if (key is None) == (generator is None):
        raise ValueError("stochastic rounding needs exactly one of key= or generator=")
    if key is not None:
        # first key element as an int64 scalar (a slice, not .item(), so it stays on-device).
        return key.reshape(-1)[:1].to(torch.int64)
    return torch.randint(0, 2**31, (1,), generator=generator, dtype=torch.int64, device=device)


def to_reference(x, dtype, *, rounding="rtne", key=None, generator=None):
    """Eager reference matching `api.to`'s logic. fp32 -> bf16 only."""
    assert x.dtype == torch.float32, f"only fp32 input supported, got {x.dtype}"
    assert dtype == torch.bfloat16, f"only bfloat16 output supported, got {dtype}"

    if rounding == "rtne":
        return x.to(torch.bfloat16)

    if rounding == "stochastic":
        seed = _seed_int64(key, generator, x.device)
        # per-element GLOBAL flat index -> tile-invariant, reproducible.
        gidx = torch.arange(x.numel(), device=x.device, dtype=torch.int64)
        seed = seed.expand(gidx.numel())
        keys = torch.stack([seed, gidx], dim=-1).to(torch.uint64)  # [seed, global_index] per element
        u = prng.uniform(keys, (gidx.numel(),)).reshape(x.shape)  # float uniform in [0, 1)
        rand16 = (u * (1 << 16)).to(torch.int32)  # uniform int in [0, 2**16): top 16 fractional bits
        return _sr_bf16_dither(x, rand16)

    if rounding == "stochastic-approx":
        raise NotImplementedError(
            "hardware cvt.rs stochastic rounding rounds inside the PTX instruction and is not "
            "reproducible in eager PyTorch without inline_asm; there is no eager reference for "
            "rounding='stochastic-approx'."
        )

    raise ValueError(f"unknown rounding {rounding!r}")
