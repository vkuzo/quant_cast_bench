"""Correctness tests for the Triton quant-cast recipes: each `triton_fn` must reproduce its
gold `pt_ref_fn`'s outputs. Inputs come from the recipe's (inherited) `example_input_fn`.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdata_utils import mismatch_fraction, qdata_and_scale_equal
from quant_cast_bench.quant_cast_triton.recipes import ALL_RECIPES

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


# Recipes whose Triton kernels emit Blackwell-only PTX cvt instructions -- fp4 E2M1
# (`cvt.e2m1x2.f32`) for the nvfp4 casts, and the MX E8M0 scale cvt (`cvt...ue8m0x2`) for the
# dim_m mxfp8 casts. ptxas rejects these below sm_100, so gate them to cuda capability 10.0.
_REQUIRES_SM100 = frozenset({
    "nvfp4",
    "nvfp4_swizzle",
    "nvfp4_sr_swizzle",
    "nvfp4_nvidia_sr_swizzle",  # cvt.rs.satfinite.e2m1x4.f32 is Blackwell-only PTX
    "mxfp8_dim_m",
    "mxfp8_dim_m_swizzle",
})


# Shapes each recipe is run at. (512, 512) is the aligned baseline every recipe supports. The two
# ragged shapes force the swizzle kernels' scale-grid padding (M%128!=0 and/or N%128!=0), which the
# aligned shapes never exercise -- the kernels allocate that grid with torch.empty and must write 0
# into every padded slot themselves. Both 96x160 and 128x160 pad M and/or N for every mxfp8 recipe
# that allows M%32/N%32 alignment (mxfp8_32x32_swizzle, mxfp8_swizzle, mxfp8_dim_m, mxfp8_dim_m_swizzle,
# mxfp8_dim_km, mxfp8_dim_km_swizzle); 128x160 additionally keeps M%128==0 while padding only N.
_SHAPES = [(512, 512), (96, 160), (128, 160)]

# Recipes whose input construction / gold / kernel needs stricter alignment than a given ragged
# shape provides, so they're skipped for that shape. deepseek needs N%128==0; nvfp4 needs N%64==0;
# nvfp4_blocked_outer needs M%128==0 and N%128==0. (The mxfp8 dim-M/dim-KM kernels only need M%32==0
# and N%32==0, so they run at both ragged shapes.) (512, 512) supports every recipe -- no blocklist.
_SHAPE_UNSUPPORTED = {
    (96, 160): frozenset({
        "fp8_deepseek_1x128", "fp8_deepseek_1x128_dim_m", "fp8_deepseek_1x128_dim_km",
        "fp8_deepseek_128x128", "nvfp4", "nvfp4_swizzle", "nvfp4_sr_swizzle",
        "nvfp4_nvidia_sr_swizzle", "nvfp4_blocked_outer",
    }),
    (128, 160): frozenset({
        "fp8_deepseek_1x128", "fp8_deepseek_1x128_dim_km", "fp8_deepseek_128x128",
        "nvfp4", "nvfp4_swizzle", "nvfp4_sr_swizzle", "nvfp4_nvidia_sr_swizzle", "nvfp4_blocked_outer",
    }),
}


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{m}x{n}" for m, n in _SHAPES])
@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_triton_matches_reference(name, recipe, shape):
    # the Triton kernel should reproduce the gold reference bit-for-bit (identical fp32 math + RNE
    # cast). example_input_fn builds the full positional inputs (x, *aux).
    if name in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{name} emits Blackwell-only PTX; requires cuda capability 10.0")
    if name in _SHAPE_UNSUPPORTED.get(shape, ()):
        pytest.skip(f"{name} needs stricter alignment than shape {shape[0]}x{shape[1]} provides")
    M, N = shape
    torch.manual_seed(0)
    inputs = recipe.example_input_fn(M, N)

    # flex_tile_map framework kwargs naming the tile's global origin + parent row stride. The test
    # runs the whole tensor as one tile, so origin = (0, 0) and num_col = full width. These are needed
    # by the global-offsets SR *reference* (`sr_bf16_global_f`) to reconstruct each element's global
    # index from a sub-tile; every recipe fn takes **kwargs, and all the Triton kernels ignore them
    # (they own their own tiling), so passing them is harmless.
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    tri_outs = recipe.triton_fn(*inputs, **tile_kwargs)

    assert len(tri_outs) == len(ref_outs), f"{name}: output count {len(tri_outs)} != {len(ref_outs)}"
    for i, (t, r) in enumerate(zip(tri_outs, ref_outs)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )

    # Every recipe's outputs must be a valid quantization (the gold correctness_fn).
    recipe.correctness_fn(inputs, tri_outs)

    # Stochastic rounding (the *_sr recipes) is generally the one case that can't bit-match: the
    # Triton kernel draws from its own counter-based Philox (tl.randint4x), not the reference's torch
    # RNG, so only the SR *property* (unbiased, lands on the two bracketing bf16 grid points) is
    # well-defined -- correctness_fn above checks that, and we stop (~2p(1-p) of elements differ
    # between any two independent draws, so a per-element bound is meaningless here).
    #
    # Two exceptions bit-match because kernel and gold draw from the SAME single-seed Philox counter
    # stream keyed on each element's GLOBAL flat index (gold gathers prng.bits(key, n)[gidx]; kernel
    # does tl.randint4x(seed, gidx>>2)[gidx&3]):
    #   * fp32_to_bf16_sr_global_offsets -- 16-bit dither, add-then-truncate to bf16.
    #   * nvfp4_sr_swizzle -- 22-bit dither into the scaled fp32 data, truncate, then the hardware fp4
    #     cvt. Because the truncation lands normals exactly on the e2m1 grid and the scale uses div.rn,
    #     the cvt matches gold's software f32_to_f4_unpacked (incl. the subnormal ties truncation makes).
    #   * nvfp4_nvidia_sr_swizzle -- the Blackwell cvt.rs.satfinite.e2m1x4.f32 intrinsic, fed one Philox
    #     word per group of 4 elements in the gold's perm order; the gold reproduces the intrinsic's
    #     exact carry-round + byte-interleaved bit layout, so kernel and gold agree bit-for-bit.
    # All fall through to the equality assertion below. Other _sr recipes (e.g. nvfp4_dim_m_rht_sr_swizzle)
    # have no Triton kernel yet and are only property-checked, so they still stop here.
    if "_sr" in name and name not in (
        "fp32_to_bf16_sr_global_offsets", "nvfp4_sr_swizzle", "nvfp4_nvidia_sr_swizzle"
    ):
        return

    # Every other recipe must reproduce the gold bit-for-bit: identical fp32 math + RNE cast (the
    # kernels mirror torch's per-op rounding, incl. reciprocal-multiply vs div.rn, so even the fp8/fp4
    # RNE ties resolve the same way). Any divergence is a real bug, not tolerable RNE noise.
    for i, (t, r) in enumerate(zip(tri_outs, ref_outs)):
        assert qdata_and_scale_equal(t, r), (
            f"{name} output {i}: {mismatch_fraction(t, r):.3%} of elements differ from the gold "
            f"reference -- expected bit-for-bit equality"
        )
