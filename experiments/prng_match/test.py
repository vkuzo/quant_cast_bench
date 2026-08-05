"""Pytest: verify the integer dither produced in eager PyTorch matches Triton's `tl.randint4x`
(the recipe kernel `_sr_bf16_global_kernel`), fed identical inputs.

Two eager generators are compared to the Triton kernel:

  * `rand16_bits`  -- built on `torch.func._random.bits` (PR pytorch#190253), which exposes
    Philox's RAW uint32 words. This MATCHES the Triton kernel bit-for-bit: same Philox4x32-10
    with the same seed->key / offset->counter mapping, low 16 bits, and the kernel's contiguous
    (r0,r1,r2,r3) lane order -- so `bits` lines up flat, no permutation needed.

  * `rand16_uniform` -- the OLD path built on `prng.uniform`, a FLOAT uniform. It does NOT
    match: `int(u * 2**16)` keeps the float's TOP fractional bits, while the kernel uses the
    LOW 16 bits of a raw uint32. Kept here as a contrast to show what the raw-bits API fixed.

Run: pytest experiments/prng_match/test.py   (requires a CUDA GPU + a build with pytorch#190253)
"""

import os
import sys

import pytest
import torch
import torch.func._random as prng

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import rand16_bits, rand16_triton, rand16_uniform  # noqa: E402

def _key(seed=42):
    return prng.key(seed, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(not hasattr(prng, "bits"), reason="needs torch.func._random.bits (pytorch#190253)")
@pytest.mark.parametrize("n", [128, 1024, 1000, 4096])
def test_bits_matches_triton(n):
    """The raw-bits eager path reproduces the Triton kernel's dither bit-for-bit, including
    a non-multiple-of-4 size (1000) that exercises the tail-masking path."""
    key = _key()
    assert torch.equal(rand16_bits(n, key), rand16_triton(n, key))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(not hasattr(prng, "bits"), reason="needs torch.func._random.bits (pytorch#190253)")
@pytest.mark.parametrize("seed", [0, 1, 42, 123456])
def test_bits_matches_triton_across_seeds(seed):
    key = _key(seed)
    assert torch.equal(rand16_bits(256, key), rand16_triton(256, key))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_uniform_does_not_match_triton():
    """The old float-uniform path keeps the float's TOP fractional bits, not the raw uint32's
    LOW 16 bits, so it cannot bit-match the kernel -- documented contrast."""
    M, N = 8, 16
    key = _key()
    tri = rand16_triton(M * N, key)
    uni = rand16_uniform(M, N, key, global_row=0, global_col=0, num_col=N).reshape(-1)
    assert not torch.equal(tri, uni)
