"""Compare, bit-for-bit, the integer dither produced by PyTorch's stateless Philox PRNG
(the gold recipe `sr_bf16_global_f`) against Triton's `tl.randint4x` (the recipe kernel
`_sr_bf16_global_kernel`), fed identical inputs.

They do NOT match today, and this script demonstrates why. See the explanation printed at
the end, and the "why" summary in this module docstring:

  1. Float vs int. PyTorch's stateless PRNG only exposes a FLOAT uniform in [0, 1). The
     baseline turns it into the dither via `int(u * 2**16)`, i.e. the TOP fractional bits
     of the float. `tl.randint4x` returns RAW uint32 words and the recipe uses the LOW 16
     bits (`rand & 0xFFFF`). Top-16-of-a-float != low-16-of-a-raw-int.

  2. One offset -> four values. The baseline keys one draw per element on counter = f (the
     global flat index). `tl.randint4x(seed, offset)` returns FOUR words per single offset,
     so the recipe keys on counter = f >> 2 and interleaves the 4 streams across the 4
     elements -- a different counter->value mapping with no 1:1 correspondence.

  (Underlying both: PyTorch packs (seed, offset) into philox4x32's key/counter and consumes
  one of the 4 philox lanes per element, while Triton maps seed->key, offset->counter and
  exposes all 4 lanes as r0..r3.)

Usage: python experiments/prng_match/test.py  (requires a CUDA GPU)
"""

import torch
import torch.func._random as prng

from api import baseline_rand16, rand16_triton


def main():
    M, N, key = 8, 16, prng.key(42, device="cuda")
    n = M * N

    # Same inputs to both. Origins 0 and num_col=N make the baseline's global flat index
    # equal the plain row-major flat position f, matching the standalone Triton kernel.
    rand16_ref = baseline_rand16(M, N, key, global_row=0, global_col=0, num_col=N).reshape(-1)
    rand16_tri = rand16_triton(n, key)

    print(f"shape: ({M}, {N}) -> {n} elements, key = {key.tolist()}")
    print(f"baseline (pytorch float PRNG): dtype={rand16_ref.dtype}, shape={tuple(rand16_ref.shape)}")
    print(f"triton   (tl.randint4x)      : dtype={rand16_tri.dtype}, shape={tuple(rand16_tri.shape)}")

    k = 8
    print(f"\nfirst {k} dither values (flat index f):")
    print(f"  {'f':>3} {'baseline':>10} {'triton':>10}")
    for f in range(k):
        print(f"  {f:>3} {rand16_ref[f].item():>10} {rand16_tri[f].item():>10}")

    match = torch.equal(rand16_ref, rand16_tri)
    print(f"\nbitwise equal: {match}  ->  {'[MATCH]' if match else '[MISMATCH]'}")

    print(
        "\nWhy they do not match today:\n"
        "  1. float vs int: the baseline draws a FLOAT uniform and takes int(u * 2**16)\n"
        "     (top fractional bits); the recipe takes the LOW 16 bits of a raw uint32\n"
        "     (rand & 0xFFFF). Different bits, even from the same generator.\n"
        "  2. one offset -> four values: the baseline keys one draw per element on\n"
        "     counter = f; tl.randint4x keys on counter = f >> 2 and returns 4 words that\n"
        "     are interleaved across the 4 elements -- a different counter->value mapping.\n"
        "  (underlying: PyTorch packs (seed, offset) into philox4x32's key/counter and\n"
        "   consumes one of the 4 philox lanes per element, while Triton maps seed->key,\n"
        "   offset->counter and exposes all 4 lanes as r0..r3.)"
    )

    # This demo documents a KNOWN non-match; "pass" means we still observe the mismatch.
    if match:
        print("\n[FAIL] expected the documented MISMATCH, but the outputs matched.")
        return 1
    print("\n[PASS] observed the documented MISMATCH.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
