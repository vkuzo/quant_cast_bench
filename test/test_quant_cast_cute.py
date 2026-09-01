"""Correctness tests for the CuTeDSL quant-cast recipes: each `cute_fn` must reproduce its gold
`pt_ref_fn`'s outputs. Mirrors test_quant_cast_triton.py: every recipe's outputs must be a valid
quantization (gold correctness_fn), and every non-SR recipe must match the gold bit-for-bit.
"""

import importlib.metadata
import os
import sys

import pytest
import torch
import torch.func._random as prng

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdata_utils import mismatch_fraction, qdata_and_scale_equal

# The CuTeDSL kernels import `_maybe_recast_from_f4_f6` (the fp4/fp6 register-packing helper) from
# cutlass.cute.testing. That is the nvidia-cutlass-dsl >= 4.5.2 name -- older releases spelled it
# `_maybe_recast_from_f4` (f4-only). Gate the whole module on the installed version so an older (or
# absent) install skips cleanly instead of erroring at collection with an ImportError, and guard the
# recipes import (it's used in a parametrize decorator, which runs at collection time regardless of
# the skip mark).
_MIN_CUTEDSL = (4, 5, 2)
try:
    _cutedsl_version = tuple(
        int(x) for x in importlib.metadata.version("nvidia-cutlass-dsl").split(".")[:3]
    )
except (ImportError, importlib.metadata.PackageNotFoundError):
    _cutedsl_version = None

HAS_CUTEDSL = _cutedsl_version is not None and _cutedsl_version >= _MIN_CUTEDSL

if HAS_CUTEDSL:
    from quant_cast_bench.quant_cast_cute.recipes import ALL_RECIPES, SR_F32_TO_BF16_GLOBAL
else:
    ALL_RECIPES = []
    SR_F32_TO_BF16_GLOBAL = None

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_CUTEDSL,
    reason=(
        f"requires CUDA and nvidia-cutlass-dsl >= {'.'.join(map(str, _MIN_CUTEDSL))} "
        f"(found {'.'.join(map(str, _cutedsl_version)) if _cutedsl_version else 'none'})"
    ),
)

# Recipes whose CuTeDSL kernels emit Blackwell-only PTX cvt instructions -- the fp4 E2M1 cvt
# (`cvt.e2m1x2.f32`) for the nvfp4 casts, and the MX E8M0 scale cvt (`cvt...ue8m0x2`) for the mxfp8
# casts. ptxas rejects these below sm_100, so gate them to cuda capability 10.0.
_REQUIRES_SM100 = frozenset({
    "nvfp4_swizzle",
    "nvfp4_blocked_outer",
    "mxfp8",
    "mxfp8_swizzle",
    "mxfp8_dim_m",
    "mxfp8_dim_m_swizzle",
    "mxfp8_dim_km",
    "mxfp8_dim_km_swizzle",
    "mxfp8_32x32",
})


@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_cute_matches_reference(name, recipe):
    # the CuTeDSL kernel should reproduce the gold reference bit-for-bit (identical fp32 math + RNE
    # cast). example_input_fn builds the full positional inputs (x, *aux).
    if name in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{name} emits Blackwell-only PTX; requires cuda capability 10.0")
    torch.manual_seed(0)
    inputs = recipe.example_input_fn(512, 512)

    # flex_tile_map framework kwargs naming the tile's global origin + parent row stride. The test
    # runs the whole tensor as one tile, so origin = (0, 0) and num_col = full width. These are needed
    # by the global-offsets SR *reference* (`sr_bf16_global_f`) to reconstruct each element's global
    # index from a sub-tile; every recipe fn takes **kwargs, and all the cute kernels ignore them
    # (they own their own tiling), so passing them is harmless.
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    cute_outs = recipe.cute_fn(*inputs, **tile_kwargs)

    assert len(cute_outs) == len(ref_outs), f"{name}: output count {len(cute_outs)} != {len(ref_outs)}"
    for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )

    # Every recipe's outputs must be a valid quantization (the gold correctness_fn).
    recipe.correctness_fn(inputs, cute_outs)

    # Stochastic rounding (the *_sr recipes) is generally the one case that can't bit-match: the cute
    # kernel draws its dither from a hand-written in-kernel Philox, and the tile-LOCAL gold keys on the
    # reference's torch RNG in tile order, so only the SR *property* (unbiased, lands on the two
    # bracketing bf16 grid points) is well-defined -- correctness_fn above checks that, and we stop
    # (~2p(1-p) of elements differ between any two independent draws, so a per-element bound is
    # meaningless here).
    #
    # The exception is fp32_to_bf16_sr_global_offsets: the cute kernel's hand-written Philox is
    # bit-identical to the gold's (same counter = gidx>>2, same straight lane order, same low-16-bit
    # dither), so it IS bit-exact and must fall through to the equality assertion below.
    if "_sr" in name and name != "fp32_to_bf16_sr_global_offsets":
        return

    # Every other recipe must reproduce the gold bit-for-bit: identical fp32 math + RNE cast (the
    # kernels mirror torch's per-op rounding, incl. reciprocal-multiply vs div.rn, so even the fp8/fp4
    # RNE ties resolve the same way). Any divergence is a real bug, not tolerable RNE noise.
    for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
        assert qdata_and_scale_equal(t, r), (
            f"{name} output {i}: {mismatch_fraction(t, r):.3%} of elements differ from the gold "
            f"reference -- expected bit-for-bit equality"
        )


def test_bf16_sr_global_full_key():
    """The bit-exact bf16 global-offsets SR kernel must reproduce its gold (`sr_bf16_global_f`) for
    ANY Philox key, including an advanced one -- SR-specific, so it lives here rather than in the
    parametrized sweep (which only uses the default key). The gold draws via `prng.bits(key, n)`,
    which honors the FULL key: `key[0]` is the 64-bit seed and `key[1]` a base counter offset (in
    Philox-block units). A kernel that reads only the low 32 bits of `key[0]` and starts counters at
    `f>>2` (ignoring `key[1]`) still matches the DEFAULT key -- `prng.key(int)` yields
    `[small_seed, 0]` -- but diverges once `fold_in`/`split` set BOTH words to full 64-bit values.
    This guards that the kernel consumes the whole key (seed = key[0] 64-bit, counter = key[1] +
    f>>2)."""
    recipe = SR_F32_TO_BF16_GLOBAL
    M, N = 512, 512
    torch.manual_seed(0)
    x, key = recipe.example_input_fn(M, N)  # key = prng.key(0) -> [0, 0]
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": N}

    # default key: kernel and gold draw the same Philox stream -> bit-for-bit equal.
    (ref0,) = recipe.pt_ref_fn(x, key, **tile_kwargs)
    (cute0,) = recipe.cute_fn(x, key, **tile_kwargs)
    assert qdata_and_scale_equal(cute0, ref0), (
        f"default key: {mismatch_fraction(cute0, ref0):.3%} of elements differ -- expected "
        f"bit-for-bit equality"
    )

    # advance the key: fold_in sets both words to full 64-bit values. gold's prng.bits honors them, so
    # a kernel that drops key[1] or the high 32 bits of key[0] would diverge here.
    key2 = prng.fold_in(key, 1)
    assert int(key2[1]) != 0, "expected fold_in to set a nonzero base-offset word (key[1])"
    (ref1,) = recipe.pt_ref_fn(x, key2, **tile_kwargs)
    (cute1,) = recipe.cute_fn(x, key2, **tile_kwargs)
    assert qdata_and_scale_equal(cute1, ref1), (
        f"advanced key: {mismatch_fraction(cute1, ref1):.3%} of elements differ -- kernel must "
        f"consume key[1] and the high 32 bits of key[0]; expected bit-for-bit equality"
    )
