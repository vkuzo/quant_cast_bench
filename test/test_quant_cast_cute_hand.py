"""Correctness tests for the handwritten CuTeDSL quant-cast recipes (quant_cast_cute_hand): each
`cute_fn` must reproduce its gold `pt_ref_fn` bit-for-bit. Mirrors test_quant_cast_cute.py; this is
the playground module we iterate on.
"""

import importlib.metadata
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdata_utils import mismatch_fraction, qdata_and_scale_equal
from quant_cast_bench.quant_cast_gold.recipes import _from_blocked_4d

# The CuTeDSL kernels import `_maybe_recast_from_f4_f6` (the fp4/fp6 register-packing helper) from
# cutlass.cute.testing. That is the nvidia-cutlass-dsl >= 4.5.2 name; gate the whole module on the
# installed version so an older (or absent) install skips cleanly instead of erroring at collection.
_MIN_CUTEDSL = (4, 5, 2)
try:
    _cutedsl_version = tuple(
        int(x) for x in importlib.metadata.version("nvidia-cutlass-dsl").split(".")[:3]
    )
except (ImportError, importlib.metadata.PackageNotFoundError):
    _cutedsl_version = None

HAS_CUTEDSL = _cutedsl_version is not None and _cutedsl_version >= _MIN_CUTEDSL

if HAS_CUTEDSL:
    from quant_cast_bench.quant_cast_cute_hand.recipes import (
        ALL_RECIPES, add_v0, add_v1, add_v2, transpose_v0, transpose_v1
    )
else:
    ALL_RECIPES = []

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_CUTEDSL,
    reason=(
        f"requires CUDA and nvidia-cutlass-dsl >= {'.'.join(map(str, _MIN_CUTEDSL))} "
        f"(found {'.'.join(map(str, _cutedsl_version)) if _cutedsl_version else 'none'})"
    ),
)

torch.manual_seed(0)

# The mxfp8 cast emits the Blackwell-only `cvt.rp.ue8m0x2.f32` (the MX E8M0 scale cvt); ptxas
# rejects it below sm_100, so gate those recipes to cuda capability 10.0. Mirrors the
# _REQUIRES_SM100 set in test_quant_cast_cute.py.
_REQUIRES_SM100 = frozenset({
    "mxfp8_swizzle",
    "mxfp8_swizzle_v2",
    "mxfp8_swizzle_v3",
    "mxfp8_swizzle_v4",
    "mxfp8_swizzle_v5",
})

def _get_recipe(recipe_name):
    _recipe_name, recipe = [x for x in ALL_RECIPES if x[0] == recipe_name][0]
    return recipe

def test_add_v0():
    M, K = 2, 64
    inputs = torch.arange(M * K, device="cuda", dtype=torch.float32).view(M, K)
    print(inputs.shape)
    print(inputs)

    num = 1.0
    outputs = add_v0(inputs, num)
    print(outputs)
    assert torch.equal(outputs, inputs + num)

def test_add_v1():
    M, K = 2, 64
    inputs = torch.arange(M * K, device="cuda", dtype=torch.float32).view(M, K)
    print(inputs.shape)
    print(inputs)

    num = 1.0
    outputs = add_v1(inputs, num)
    print(outputs)
    assert torch.equal(outputs, inputs + num)

def test_add_v2():
    M, K = 2, 64
    # M, K = 4, 1024
    inputs = torch.arange(M * K, device="cuda", dtype=torch.float32).view(M, K)
    # print(inputs.shape)
    # print(inputs)

    num = 1.0
    outputs = add_v2(inputs, num)
    # print(outputs)
    assert torch.equal(outputs, inputs + num)

def test_deepseek_1x128():
    recipe = _get_recipe("deepseek_1x128")
    inputs = recipe.example_input_fn(2, 2048)
    print(inputs[0].shape)
    print(inputs)

    outputs = recipe.cute_fn(*inputs)
    print(outputs)
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    print(ref_outputs)
    recipe.correctness_fn(inputs, outputs)

def test_mxfp8_swizzle():
    # M % 128 == 0 and (N // 32) % 4 == 0: whole 128x4 swizzle atoms (ngc=16, ncb=4).
    if "mxfp8_swizzle" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle")
    inputs = recipe.example_input_fn(128, 512)
    print(inputs[0].shape)
    # print(inputs)

    outputs = recipe.cute_fn(*inputs)
    # print(outputs)
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    # print(ref_outputs)
    recipe.correctness_fn(inputs, outputs)

def test_mxfp8_swizzle_v2():
    # TMA (bulk-tensor) load/store variant. 128x128 tile: M%128==0 and N%128==0. Use 256x256 so the
    # grid is 2x2 (exercises multiple m/n tiles), still all full tiles.
    if "mxfp8_swizzle_v2" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v2 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v2")
    inputs = recipe.example_input_fn(256, 256)
    print(inputs[0].shape)

    outputs = recipe.cute_fn(*inputs)
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    recipe.correctness_fn(inputs, outputs)

def test_mxfp8_swizzle_v3():
    # 2-D 32x128-tile variant of v1 with a 16-elem/thread aligned uint32-word load (2x LDG.128).
    # M % 128 == 0 and N % 128 == 0: whole 128x4 swizzle atoms, tile fits (ngc=16, ncb=4).
    if "mxfp8_swizzle_v3" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v3 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v3")
    inputs = recipe.example_input_fn(128, 512)
    print(inputs[0].shape)

    outputs = recipe.cute_fn(*inputs)
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    recipe.correctness_fn(inputs, outputs)

def test_mxfp8_swizzle_v4():
    # Same 2-D 32x128 tile + 16 bf16/thread as v3, but the load is two plain 8-elem fragment .load()s
    # (each an LDG.128) concat'd in registers -- no uint32-word path.
    if "mxfp8_swizzle_v4" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v4 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v4")
    inputs = recipe.example_input_fn(128, 512)
    print(inputs[0].shape)

    outputs = recipe.cute_fn(*inputs)
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    recipe.correctness_fn(inputs, outputs)


@pytest.mark.parametrize("M,K", [(1, 32), (31, 96), (33, 160), (127, 64), (129, 160)])
def test_mxfp8_swizzle_v4_padding(M, K):
    if "mxfp8_swizzle_v4" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v4 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v4")
    inputs = recipe.example_input_fn(M, K)

    outputs = recipe.cute_fn(*inputs)
    ref_outputs = recipe.pt_ref_fn(*inputs)
    assert qdata_and_scale_equal(outputs[0], ref_outputs[0])
    assert qdata_and_scale_equal(outputs[1], ref_outputs[1])
    assert outputs[0].shape == (M, K)

    ngc = K // 32
    nrb, ncb = (M + 127) // 128, (ngc + 3) // 4
    assert outputs[1].shape == (nrb, ncb, 32, 16)
    padded = _from_blocked_4d(outputs[1], nrb * 128, ncb * 4).view(torch.uint8)
    assert torch.count_nonzero(padded[M:, :]) == 0
    assert torch.count_nonzero(padded[:M, ngc:]) == 0


@pytest.mark.parametrize("M,K", [(0, 32), (1, 0), (1, 31), (1, 33)])
def test_mxfp8_swizzle_v4_rejects_invalid_shapes(M, K):
    if "mxfp8_swizzle_v4" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v4 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v4")
    inputs = recipe.example_input_fn(M, K)
    with pytest.raises(AssertionError):
        recipe.cute_fn(*inputs)


def test_mxfp8_swizzle_v5():
    # Best of v1 and v4: v1's flat 1-D grid + v4's 16-elem/thread load (two LDG.128 concat'd in
    # registers) and single STG.128 store.
    if "mxfp8_swizzle_v5" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v5 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v5")
    inputs = recipe.example_input_fn(128, 512)
    print(inputs[0].shape)

    outputs = recipe.cute_fn(*inputs)
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    recipe.correctness_fn(inputs, outputs)


@pytest.mark.parametrize("M,K", [(1, 32), (31, 96), (33, 160), (127, 64), (129, 160)])
def test_mxfp8_swizzle_v5_padding(M, K):
    if "mxfp8_swizzle_v5" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v5 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v5")
    inputs = recipe.example_input_fn(M, K)

    outputs = recipe.cute_fn(*inputs)
    ref_outputs = recipe.pt_ref_fn(*inputs)
    assert qdata_and_scale_equal(outputs[0], ref_outputs[0])
    assert qdata_and_scale_equal(outputs[1], ref_outputs[1])
    assert outputs[0].shape == (M, K)

    ngc = K // 32
    nrb, ncb = (M + 127) // 128, (ngc + 3) // 4
    assert outputs[1].shape == (nrb, ncb, 32, 16)
    padded = _from_blocked_4d(outputs[1], nrb * 128, ncb * 4).view(torch.uint8)
    assert torch.count_nonzero(padded[M:, :]) == 0
    assert torch.count_nonzero(padded[:M, ngc:]) == 0


@pytest.mark.parametrize("M,K", [(0, 32), (1, 0), (1, 31), (1, 33)])
def test_mxfp8_swizzle_v5_rejects_invalid_shapes(M, K):
    if "mxfp8_swizzle_v5" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v5 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v5")
    inputs = recipe.example_input_fn(M, K)
    with pytest.raises(AssertionError):
        recipe.cute_fn(*inputs)


def test_transpose_v0():
    # 128x16 tile: M must be a multiple of 128, K a multiple of 16.
    M, K = 128, 256
    inputs = torch.arange(M * K, device="cuda", dtype=torch.bfloat16).view(M, K)
    # print(inputs.shape)
    print('\n', inputs.shape, inputs)

    outputs = transpose_v0(inputs)
    print(outputs.shape, outputs)
    assert torch.equal(outputs, inputs.t().contiguous())

def test_transpose_v1():
    # 128x16 tile: M must be a multiple of 128, K a multiple of 16.
    M, K = 128, 256
    inputs = torch.arange(M * K, device="cuda", dtype=torch.bfloat16).view(M, K)
    # print(inputs.shape)
    print('\n', inputs.shape, inputs)

    outputs = transpose_v1(inputs)
    print(outputs.shape, outputs)
    assert torch.equal(outputs, inputs.t().contiguous())

def test_deepseek_1x128_dim_m():
    recipe = _get_recipe("deepseek_1x128_dim_m")
    inputs = recipe.example_input_fn(256, 512)
    print(inputs[0].shape)
    print(inputs)

    outputs = recipe.cute_fn(*inputs)
    print(outputs)

    # return
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    print(ref_outputs)
    recipe.correctness_fn(inputs, outputs)

def test_deepseek_1x128_dim_m_v2():
    recipe = _get_recipe("deepseek_1x128_dim_m_v2")
    inputs = recipe.example_input_fn(256, 512)
    print(inputs[0].shape)
    print(inputs)

    outputs = recipe.cute_fn(*inputs)
    print(outputs)

    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    print(ref_outputs)
    recipe.correctness_fn(inputs, outputs)

@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_cute_hand_matches_reference(name, recipe):
    # the CuTeDSL kernel should reproduce the gold reference bit-for-bit (identical fp32 math + RNE
    # cast). example_input_fn builds the full positional inputs (x, *aux).
    if name in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{name} emits Blackwell-only PTX; requires cuda capability 10.0")
    torch.manual_seed(0)
    inputs = recipe.example_input_fn(512, 512)

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

    # And must reproduce the gold bit-for-bit: identical fp32 math + RNE cast.
    for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
        assert qdata_and_scale_equal(t, r), (
            f"{name} output {i}: {mismatch_fraction(t, r):.3%} of elements differ from the gold "
            f"reference -- expected bit-for-bit equality"
        )
