"""Correctness tests for the handwritten CuTeDSL quant-cast recipes (quant_cast_cute_hand): each
`cute_fn` must reproduce its gold `pt_ref_fn` bit-for-bit. Mirrors test_quant_cast_cute.py; this is
the playground module we iterate on.
"""

import importlib.metadata
import math
import os
import sys

import pytest
import torch
import torch.func._random as prng

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdata_utils import mismatch_fraction, qdata_and_scale_equal
from quant_cast_bench.quant_cast_gold.recipes import (
    Nvfp4GsSwizzleDimMRHTGold,
    _from_blocked_4d,
    hadamard_rht_fp32_f,
    hadamard_rht_matrix,
    mxfp8_swizzle_sr_f,
    nvfp4_gs_scale,
)

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
    from cutedsl_test_utils import run_i32_ceil_div
    from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma.blockscaled_tma_impl import (
        mxfp4,
        mxfp4_swizzle_v2,
        mxfp8,
        mxfp8_swizzle_stateful_sr_v2,
        mxfp8_swizzle_v2,
        nvfp4,
        nvfp4_swizzle_16x16_tma,
        nvfp4_swizzle_tma,
    )
    from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma.blockscaled_tma_kernels import (
        _compile_blockscaled_tma,
    )
    from quant_cast_bench.quant_cast_cute_hand.nvfp4_pipelined.nvfp4_pipelined_impl import (
        nvfp4_swizzle_dim_k_dim_m_rht_pipelined,
    )
    from quant_cast_bench.quant_cast_cute_hand.nvfp4_pipelined.nvfp4_pipelined_kernels import (
        _compile_nvfp4_rht_pipelined,
    )
    from quant_cast_bench.quant_cast_cute_hand.recipes import (
        ALL_RECIPES,
        add_v0,
        add_v1,
        add_v2,
        mxfp8_swizzle_v4,
        transpose_v0,
        transpose_v1,
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
    "mxfp8",
    "mxfp8_swizzle",
    "mxfp8_swizzle_v2",
    "mxfp8_swizzle_sr_v2",
    "mxfp8_swizzle_stateful_sr_v2",
    "mxfp8_32x32_swizzle_v2",
    "mxfp8_swizzle_v3",
    "mxfp8_swizzle_v4",
    "mxfp8_swizzle_sr_v4",
    "mxfp8_swizzle_v5",
    "mxfp8_dim_m_swizzle_v2",
    "mxfp8_dim_m_swizzle_sr_v2",
    "mxfp8_dim_km_swizzle_v2",
    "mxfp8_dim_km_swizzle_sr_v2",
    "mxfp4_swizzle_v2",
    "mxfp4_dim_m_swizzle_v2",
    "mxfp4_dim_km_swizzle_v2",
    "mxfp4",
    "nvfp4",
    "nvfp4_swizzle_direct",
    "nvfp4_swizzle_tma",
    "nvfp4_swizzle_16x16_tma",
    "nvfp4_dim_m_swizzle_tma",
    "nvfp4_dim_km_swizzle_tma",
    "nvfp4_dim_m_rht_swizzle_pipelined",
    "nvfp4_dim_m_swizzle_rht_sr_pipelined",
    "nvfp4_swizzle_dim_k_dim_m_rht_pipelined",
    "nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined",
})

_MX_V2_RECIPES = frozenset({
    "mxfp8",
    "mxfp8_swizzle_v2",
    "mxfp8_swizzle_sr_v2",
    "mxfp8_swizzle_stateful_sr_v2",
    "mxfp8_32x32_swizzle_v2",
    "mxfp8_dim_m_swizzle_v2",
    "mxfp8_dim_m_swizzle_sr_v2",
    "mxfp8_dim_km_swizzle_v2",
    "mxfp8_dim_km_swizzle_sr_v2",
    "mxfp4_swizzle_v2",
    "mxfp4_dim_m_swizzle_v2",
    "mxfp4_dim_km_swizzle_v2",
    "mxfp4",
})
_BLOCKSCALED_TMA_RECIPES = _MX_V2_RECIPES | {
    "nvfp4",
    "nvfp4_swizzle_tma",
    "nvfp4_swizzle_16x16_tma",
    "nvfp4_dim_m_swizzle_tma",
    "nvfp4_dim_km_swizzle_tma",
}
_SUPPORTS_NON_BFLOAT16 = _BLOCKSCALED_TMA_RECIPES

def _get_recipe(recipe_name):
    _recipe_name, recipe = [x for x in ALL_RECIPES if x[0] == recipe_name][0]
    return recipe


def _nvfp4_dim_m_rht_test_inputs(M, K, *, stochastic=False):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rht_sign = torch.tensor([1, -1] * 8, device=x.device, dtype=x.dtype)
    rht = hadamard_rht_matrix(rht_sign, x.device, x.dtype)
    (x_t_rht,) = hadamard_rht_fp32_f(x.t().contiguous(), rht)
    outer_scale = nvfp4_gs_scale(x_t_rht).reciprocal()
    if stochastic:
        key = prng.key(0, device=x.device)
        return (x, outer_scale, rht_sign, key), (x, outer_scale, rht, key)
    return (x, outer_scale, rht_sign), (x, outer_scale, rht)


def _nvfp4_dim_km_rht_test_inputs(M, K, *, stochastic=False):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rht_sign = torch.tensor([1, -1] * 8, device=x.device, dtype=x.dtype)
    rht = hadamard_rht_matrix(rht_sign, x.device, x.dtype)
    (x_t_rht,) = hadamard_rht_fp32_f(x.t().contiguous(), rht)
    outer_scale_k = nvfp4_gs_scale(x).reciprocal()
    outer_scale_m = nvfp4_gs_scale(x_t_rht).reciprocal()
    if stochastic:
        key = prng.fold_in(prng.key(7, device=x.device), 12345)
        return (
            x, outer_scale_k, outer_scale_m, rht_sign, key
        ), (
            x, outer_scale_k, outer_scale_m, rht, key
        )
    return (
        x, outer_scale_k, outer_scale_m, rht_sign
    ), (
        x, outer_scale_k, outer_scale_m, rht
    )

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
    inputs = recipe.example_input_fn(2, 2048, torch.bfloat16)
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
    inputs = recipe.example_input_fn(128, 512, torch.bfloat16)
    print(inputs[0].shape)
    # print(inputs)

    outputs = recipe.cute_fn(*inputs)
    # print(outputs)
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    # print(ref_outputs)
    recipe.correctness_fn(inputs, outputs)

@pytest.mark.parametrize(
    "M,K", [(256, 256), (256, 512), (512, 256), (1024, 1152), (2048, 4224)]
)
def test_mxfp8_swizzle_v2(M, K):
    # TMA (bulk-tensor) load/store variant. Rectangles catch swapped M/N grid coordinates.
    if "mxfp8_swizzle_v2" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v2 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v2")
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)
    print(inputs[0].shape)

    outputs = recipe.cute_fn(*inputs)
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    recipe.correctness_fn(inputs, outputs)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_mxfp8_swizzle_v2_guards_input_device():
    input_device = 1
    if torch.cuda.get_device_capability(input_device) != (10, 0):
        pytest.skip("mxfp8_swizzle_v2 emits Blackwell-only PTX; input device must be SM100")

    original_device = torch.cuda.current_device()
    try:
        with torch.cuda.device(input_device):
            x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
            reference = _get_recipe("mxfp8_swizzle_v2").pt_ref_fn(x)

        torch.cuda.set_device(0)
        outputs = mxfp8_swizzle_v2(x)

        assert torch.cuda.current_device() == 0
        assert all(output.device == x.device for output in outputs)
        for output, expected in zip(outputs, reference):
            assert qdata_and_scale_equal(output, expected)
    finally:
        torch.cuda.set_device(original_device)


def test_mxfp8_v2_rejects_unsupported_input_dtype():
    x = torch.randn(128, 256, dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="supports only bf16, fp16, and fp32"):
        mxfp8_swizzle_v2(x)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_mxfp8_v2_rejects_misaligned_contiguous_input(dtype):
    storage = torch.empty(128 * 256 + 1, dtype=dtype, device="cuda")
    x = storage[1:].view(128, 256)
    assert x.is_contiguous()
    assert x.data_ptr() % 16 != 0

    with pytest.raises(ValueError, match="requires a 16-byte-aligned input"):
        mxfp8_swizzle_v2(x)


@pytest.mark.parametrize(
    "input_transform,kwargs,error",
    [
        (lambda x: x.unsqueeze(0), {}, "requires a 2D input"),
        (lambda x: x.t(), {}, "requires a contiguous input"),
        (lambda x: x, {"quant_orientation": "rows"}, "unsupported quant_orientation"),
        (lambda x: x, {"rounding_mode": "toward_zero"}, "unsupported rounding_mode"),
        (
            lambda x: x,
            {"rounding_mode": "stochastic"},
            "stochastic rounding requires a Philox key",
        ),
    ],
)
def test_mxfp8_v2_rejects_invalid_arguments(input_transform, kwargs, error):
    x = input_transform(torch.randn(128, 256, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match=error):
        mxfp8_swizzle_v2(x, **kwargs)


def test_mxfp8_v2_rejects_key_for_rtne():
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    key = torch.tensor([0, 0], dtype=torch.uint64, device=x.device)
    with pytest.raises(ValueError, match="RTNE rounding does not use a Philox key"):
        mxfp8_swizzle_v2(x, key=key)


def test_mxfp8_v2_rejects_non_cuda_input():
    with pytest.raises(ValueError, match="requires a CUDA input"):
        mxfp8_swizzle_v2(torch.randn(128, 256, dtype=torch.bfloat16))


def test_mxfp8_v2_rejects_unexpected_keyword_arguments():
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        ValueError,
        match="unexpected keyword arguments: global_row, roundng_mode",
    ):
        mxfp8_swizzle_v2(x, global_row=0, roundng_mode="stochastic")


def test_mxfp8_v2_dynamic_shapes_share_compile_cache():
    x0 = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    x1 = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")

    mxfp8_swizzle_v2(x0)
    after_first = _compile_blockscaled_tma.cache_info()
    mxfp8_swizzle_v2(x1)
    after_second = _compile_blockscaled_tma.cache_info()

    assert after_second.currsize == after_first.currsize
    assert after_second.hits == after_first.hits + 1


def test_nvfp4_tma_dynamic_shapes_share_compile_cache():
    x0 = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    x1 = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")
    outer_scale = torch.ones(1, dtype=torch.float32, device="cuda")

    nvfp4_swizzle_tma(x0, outer_scale)
    after_first = _compile_blockscaled_tma.cache_info()
    nvfp4_swizzle_tma(x1, outer_scale)
    after_second = _compile_blockscaled_tma.cache_info()

    assert after_second.currsize == after_first.currsize
    assert after_second.hits == after_first.hits + 1


def test_nvfp4_pipelined_dynamic_shapes_share_compile_cache():
    x0 = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
    x1 = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda")
    outer_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    rht_sign = torch.ones(16, dtype=torch.bfloat16, device="cuda")

    nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
        x0,
        outer_scale,
        outer_scale,
        rht_sign,
    )
    after_first = (
        _compile_nvfp4_rht_pipelined.cache_info()
    )
    nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
        x1,
        outer_scale,
        outer_scale,
        rht_sign,
    )
    after_second = (
        _compile_nvfp4_rht_pipelined.cache_info()
    )

    assert after_second.currsize == after_first.currsize
    assert after_second.hits == after_first.hits + 1


@pytest.mark.parametrize("M,K", [(32, 32), (96, 160), (128, 256), (1024, 1152)])
def test_mxfp8_32x32_swizzle_v2(M, K):
    recipe = _get_recipe("mxfp8_32x32_swizzle_v2")
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)
    outputs = recipe.cute_fn(*inputs)
    ref_outputs = recipe.pt_ref_fn(*inputs)
    assert qdata_and_scale_equal(outputs[0], ref_outputs[0])
    assert qdata_and_scale_equal(outputs[1], ref_outputs[1])


@pytest.mark.parametrize("M,K", [(31, 32), (32, 31)])
def test_mxfp8_32x32_swizzle_v2_rejects_invalid_shapes(M, K):
    recipe = _get_recipe("mxfp8_32x32_swizzle_v2")
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)
    with pytest.raises(ValueError):
        recipe.cute_fn(*inputs)


@pytest.mark.parametrize(
    "M,K,expected_shapes",
    [
        (0, 32, ((0, 32), (0, 1, 32, 16))),
        (32, 0, ((32, 0), (1, 0, 32, 16))),
    ],
)
def test_mxfp8_32x32_swizzle_v2_empty(M, K, expected_shapes):
    x = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
    outputs = _get_recipe("mxfp8_32x32_swizzle_v2").cute_fn(x)

    assert tuple(output.shape for output in outputs) == expected_shapes
    assert all(output.numel() == 0 for output in outputs)
    assert tuple(output.dtype for output in outputs) == (
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
    )


@pytest.mark.parametrize("M,K", [(16, 32), (32, 128), (96, 160), (128, 256)])
def test_nvfp4_swizzle_16x16_tma(M, K):
    recipe = _get_recipe("nvfp4_swizzle_16x16_tma")
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)

    outputs = nvfp4_swizzle_16x16_tma(*inputs)
    reference = recipe.pt_ref_fn(*inputs)

    for output, expected in zip(outputs, reference):
        assert qdata_and_scale_equal(output, expected)


@pytest.mark.parametrize("M,K", [(15, 32), (16, 16)])
def test_nvfp4_swizzle_16x16_tma_rejects_invalid_shapes(M, K):
    recipe = _get_recipe("nvfp4_swizzle_16x16_tma")
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)
    with pytest.raises(ValueError):
        nvfp4_swizzle_16x16_tma(*inputs)


@pytest.mark.parametrize(
    "M,K,expected_shapes",
    [
        (0, 32, ((0, 16), (0, 1, 32, 16))),
        (16, 0, ((16, 0), (1, 0, 32, 16))),
    ],
)
def test_nvfp4_swizzle_16x16_tma_empty(M, K, expected_shapes):
    x = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
    outer_scale = torch.ones(1, dtype=torch.float32, device=x.device)

    outputs = nvfp4_swizzle_16x16_tma(x, outer_scale)

    assert tuple(output.shape for output in outputs) == expected_shapes
    assert all(output.numel() == 0 for output in outputs)
    assert tuple(output.dtype for output in outputs) == (
        torch.float4_e2m1fn_x2,
        torch.float8_e4m3fn,
    )


def test_mxfp8_swizzle_sr_v2_folded_key_and_padding():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("v2 stochastic rounding emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_sr_v2")
    x, _ = recipe.example_input_fn(129, 160, torch.bfloat16)
    x[0, :8] = torch.tensor(
        [448.0, -448.0, 0.0, -0.0, 2.0**-9, -(2.0**-9), 2.0**-10, -(2.0**-10)],
        dtype=x.dtype,
        device=x.device,
    )
    key = prng.fold_in(prng.key(7, device=x.device), 12345)

    outputs = mxfp8_swizzle_v2(x, key=key, rounding_mode="STOCHASTIC")
    ref_outputs = recipe.pt_ref_fn(x, key)
    assert qdata_and_scale_equal(outputs[0], ref_outputs[0])
    assert qdata_and_scale_equal(outputs[1], ref_outputs[1])


def test_mxfp8_swizzle_stateful_sr_v2_generator_mapping_and_padding():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("v2 stochastic rounding emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_stateful_sr_v2")
    x, _ = recipe.example_input_fn(129, 160, torch.bfloat16)
    generator_ref = torch.Generator(device=x.device).manual_seed((1 << 64) - 3)
    generator_ref.set_offset(20)
    generator_actual = generator_ref.clone_state()

    ref_outputs = recipe.pt_ref_fn(x, generator_ref)
    outputs = mxfp8_swizzle_stateful_sr_v2(x, generator_actual)

    assert generator_ref.get_offset() == 24
    assert generator_actual.get_offset() == 24
    assert qdata_and_scale_equal(outputs[0], ref_outputs[0])
    assert qdata_and_scale_equal(outputs[1], ref_outputs[1])


def test_mxfp8_swizzle_stateful_sr_v2_cuda_graph():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("v2 stochastic rounding emits Blackwell-only PTX; requires cuda capability 10.0")
    x = torch.randn(129, 160, device="cuda", dtype=torch.bfloat16)
    seed = (1 << 64) - 3

    eager_generator = torch.Generator(device=x.device).manual_seed(seed)
    eager_outputs_0 = mxfp8_swizzle_stateful_sr_v2(x, eager_generator)
    eager_outputs_1 = mxfp8_swizzle_stateful_sr_v2(x, eager_generator)

    graph_generator = torch.Generator(device=x.device).manual_seed(seed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs_0 = mxfp8_swizzle_stateful_sr_v2(x, graph_generator)
        graph_outputs_1 = mxfp8_swizzle_stateful_sr_v2(x, graph_generator)
    graph.replay()
    torch.cuda.synchronize()

    for actual_outputs, expected_outputs in (
        (graph_outputs_0, eager_outputs_0),
        (graph_outputs_1, eager_outputs_1),
    ):
        for actual, expected in zip(actual_outputs, expected_outputs):
            assert qdata_and_scale_equal(actual, expected)


def test_mxfp8_swizzle_sr_v2_mixed_width_indexing():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("v2 stochastic rounding emits Blackwell-only PTX; requires cuda capability 10.0")

    # This is the first square shape divisible by 32 whose flattened element count exceeds the
    # signed-int32 range. Logical coordinates stay int32, but the flattening multiply must widen
    # before computing the Philox counter. Pick a 1x32 group starting exactly at flat index 2**31.
    M = K = 46368
    flat_start = 2**31
    row, col = divmod(flat_start, K)
    assert col % 32 == 0

    x = torch.empty((M, K), dtype=torch.bfloat16, device="cuda")
    values = torch.linspace(-1.3, 1.7, 32, dtype=torch.float32, device=x.device).to(
        torch.bfloat16
    )
    x[row, col : col + 32] = values
    key = prng.key(7, device=x.device)

    qdata, scale = mxfp8_swizzle_v2(
        x,
        key=key,
        rounding_mode="stochastic",
    )
    shifted_key = torch.tensor(
        [7, flat_start // 16], dtype=torch.uint64, device=x.device
    )
    qdata_ref, _ = mxfp8_swizzle_sr_f(values.reshape(1, 32), shifted_key)

    assert torch.equal(
        qdata[row, col : col + 32].view(torch.uint8),
        qdata_ref[0].view(torch.uint8),
    )

    del x, qdata, scale
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "quant_orientation,recipe_name,M,N",
    [
        ("dim_m", "mxfp8_dim_m_swizzle_sr_v2", 96, 144),
        ("dim_km", "mxfp8_dim_km_swizzle_sr_v2", 96, 160),
    ],
)
def test_mxfp8_swizzle_sr_v2_dim_m_orientations(
    quant_orientation, recipe_name, M, N
):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("v2 stochastic rounding emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe(recipe_name)
    x, _ = recipe.example_input_fn(M, N, torch.bfloat16)
    key = prng.fold_in(prng.key(7, device=x.device), 12345)

    outputs = mxfp8_swizzle_v2(
        x,
        quant_orientation=quant_orientation,
        key=key,
        rounding_mode="stochastic",
    )
    ref_outputs = recipe.pt_ref_fn(x, key)
    assert len(outputs) == len(ref_outputs)
    for output, ref_output in zip(outputs, ref_outputs):
        assert qdata_and_scale_equal(output, ref_output)


@pytest.mark.parametrize(
    "quant_orientation,reference",
    [
        ("dim_m", "mxfp8_dim_m_swizzle_v2"),
        ("dim_km", "mxfp8_dim_km_swizzle_v2"),
    ],
)
def test_mxfp8_swizzle_v2_quant_orientation(quant_orientation, reference):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v2 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v2")
    reference_recipe = _get_recipe(reference)
    inputs = recipe.example_input_fn(96, 160, torch.bfloat16)

    outputs = recipe.cute_fn(*inputs, quant_orientation=quant_orientation)
    ref_outputs = reference_recipe.pt_ref_fn(*inputs)
    assert len(outputs) == len(ref_outputs)
    for output, ref_output in zip(outputs, ref_outputs):
        assert qdata_and_scale_equal(output, ref_output)


@pytest.mark.parametrize(
    "quant_orientation,reference,M,K",
    [
        ("dim_k", "mxfp4_swizzle_v2", 129, 160),
        ("dim_m", "mxfp4_dim_m_swizzle_v2", 160, 144),
        ("dim_km", "mxfp4_dim_km_swizzle_v2", 160, 160),
    ],
)
def test_mxfp4_swizzle_v2_quant_orientation(
    quant_orientation, reference, M, K
):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp4 v2 emits Blackwell-only PTX; requires cuda capability 10.0")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    outputs = mxfp4_swizzle_v2(x, quant_orientation=quant_orientation)
    ref_outputs = _get_recipe(reference).pt_ref_fn(x)

    assert len(outputs) == len(ref_outputs)
    for output, ref_output in zip(outputs, ref_outputs):
        assert qdata_and_scale_equal(output, ref_output)


@pytest.mark.parametrize(
    "name,kernel,M,K,scale_shape",
    [
        ("mxfp8", mxfp8, 129, 160, (129, 5)),
        ("mxfp4", mxfp4, 129, 160, (129, 5)),
        ("nvfp4", nvfp4, 129, 160, (129, 10)),
    ],
)
def test_blockscaled_tma_compact_scale(name, kernel, M, K, scale_shape):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{name} emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe(name)
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)

    outputs = kernel(*inputs)
    reference = recipe.pt_ref_fn(*inputs)

    assert outputs[1].shape == scale_shape
    for output, expected in zip(outputs, reference):
        assert qdata_and_scale_equal(output, expected)


@pytest.mark.parametrize(
    "M,K",
    [
        (1, 32),
        (31, 96),
        (33, 160),
        (127, 64),
        (128, 160),
        (129, 128),
        (129, 160),
        (1025, 1056),
        (2049, 4096),
        (4096, 4064),
    ],
)
def test_mxfp8_swizzle_v2_padding(M, K):
    if "mxfp8_swizzle_v2" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v2 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v2")
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)

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


@pytest.mark.parametrize("M,K", [(1, 31), (1, 33)])
def test_mxfp8_swizzle_v2_rejects_invalid_shapes(M, K):
    if "mxfp8_swizzle_v2" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v2 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v2")
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)
    with pytest.raises(ValueError):
        recipe.cute_fn(*inputs)


@pytest.mark.parametrize(
    "quant_orientation,M,K,expected_shapes",
    [
        ("dim_k", 0, 32, ((0, 32), (0, 1, 32, 16))),
        ("dim_k", 1, 0, ((1, 0), (1, 0, 32, 16))),
        ("dim_m", 0, 16, ((16, 0), (1, 0, 32, 16))),
        ("dim_m", 32, 0, ((0, 32), (0, 1, 32, 16))),
        (
            "dim_km",
            0,
            32,
            ((0, 32), (0, 1, 32, 16), (32, 0), (1, 0, 32, 16)),
        ),
        (
            "dim_km",
            32,
            0,
            ((32, 0), (1, 0, 32, 16), (0, 32), (0, 1, 32, 16)),
        ),
        ("dim_km", 0, 0, ((0, 0), (0, 0, 32, 16)) * 2),
    ],
)
@pytest.mark.parametrize("rounding_mode", ["rtne", "stochastic"])
def test_mxfp8_swizzle_v2_empty(
    quant_orientation, M, K, expected_shapes, rounding_mode
):
    x = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
    key = prng.key(0, device=x.device) if rounding_mode == "stochastic" else None
    outputs = mxfp8_swizzle_v2(
        x,
        quant_orientation=quant_orientation,
        key=key,
        rounding_mode=rounding_mode,
    )

    assert tuple(output.shape for output in outputs) == expected_shapes
    assert all(output.numel() == 0 for output in outputs)
    assert tuple(output.dtype for output in outputs) == (
        (torch.float8_e4m3fn, torch.float8_e8m0fnu)
        if quant_orientation != "dim_km"
        else (
            torch.float8_e4m3fn,
            torch.float8_e8m0fnu,
            torch.float8_e4m3fn,
            torch.float8_e8m0fnu,
        )
    )


def test_mxfp8_swizzle_v2_rejects_grid_y_overflow():
    # The largest legal CUDA grid.y is 65,535. Dim-M pads M to a complete 128-row
    # scale-layout block, making this 65,538 CTAs with its selected 64-row tile.
    x = torch.empty((4_194_368, 16), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="launch grid exceeds CUDA limits"):
        mxfp8_swizzle_v2(x, quant_orientation="dim_m")


def test_cute_i32_ceil_div_does_not_overflow():
    numerator = 2_147_483_616  # Largest multiple of 32 below INT32_MAX.
    denominator = 128
    assert run_i32_ceil_div(numerator, denominator) == (
        numerator + denominator - 1
    ) // denominator


def test_mxfp8_swizzle_v3():
    # 2-D 32x128-tile variant of v1 with a 16-elem/thread aligned uint32-word load (2x LDG.128).
    # M % 128 == 0 and N % 128 == 0: whole 128x4 swizzle atoms, tile fits (ngc=16, ncb=4).
    if "mxfp8_swizzle_v3" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v3 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v3")
    inputs = recipe.example_input_fn(128, 512, torch.bfloat16)
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
    inputs = recipe.example_input_fn(128, 512, torch.bfloat16)
    print(inputs[0].shape)

    outputs = recipe.cute_fn(*inputs)
    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    recipe.correctness_fn(inputs, outputs)


def test_mxfp8_swizzle_sr_v4_folded_key_and_padding():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("v4 stochastic rounding emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_sr_v4")
    x, _ = recipe.example_input_fn(129, 160, torch.bfloat16)
    x[0, :8] = torch.tensor(
        [448.0, -448.0, 0.0, -0.0, 2.0**-9, -(2.0**-9), 2.0**-10, -(2.0**-10)],
        dtype=x.dtype,
        device=x.device,
    )
    key = prng.fold_in(prng.key(7, device=x.device), 12345)

    outputs = mxfp8_swizzle_v4(x, key, rounding_mode="STOCHASTIC")
    ref_outputs = recipe.pt_ref_fn(x, key)
    assert qdata_and_scale_equal(outputs[0], ref_outputs[0])
    assert qdata_and_scale_equal(outputs[1], ref_outputs[1])


def test_mxfp8_swizzle_v4_rounding_mode_validation():
    x = torch.randn(1, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="requires a Philox key"):
        mxfp8_swizzle_v4(x, rounding_mode="stochastic")
    with pytest.raises(AssertionError, match="unsupported rounding_mode"):
        mxfp8_swizzle_v4(x, rounding_mode="toward_zero")


@pytest.mark.parametrize("kernel", ["mxfp8_dim_m_swizzle_v2"])
def test_mxfp8_dim_m_swizzle(kernel):
    if kernel in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{kernel} emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe(kernel)
    inputs = recipe.example_input_fn(128, 512, torch.bfloat16)
    outputs = recipe.cute_fn(*inputs)
    ref_outputs = recipe.pt_ref_fn(*inputs)
    assert qdata_and_scale_equal(outputs[0], ref_outputs[0])
    assert qdata_and_scale_equal(outputs[1], ref_outputs[1])


@pytest.mark.parametrize("kernel", ["mxfp8_dim_m_swizzle_v2"])
def test_mxfp8_dim_m_swizzle_padding(kernel):
    if kernel in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{kernel} emits Blackwell-only PTX; requires cuda capability 10.0")
    M, N = 96, 144
    recipe = _get_recipe(kernel)
    inputs = recipe.example_input_fn(M, N, torch.bfloat16)
    outputs = recipe.cute_fn(*inputs)
    ref_outputs = recipe.pt_ref_fn(*inputs)
    assert qdata_and_scale_equal(outputs[0], ref_outputs[0])
    assert qdata_and_scale_equal(outputs[1], ref_outputs[1])
    assert outputs[0].shape == (N, M)

    nrb, ncb = (N + 127) // 128, ((M // 32) + 3) // 4
    assert outputs[1].shape == (nrb, ncb, 32, 16)
    padded = _from_blocked_4d(outputs[1], nrb * 128, ncb * 4).view(torch.uint8)
    assert torch.count_nonzero(padded[N:, :]) == 0
    assert torch.count_nonzero(padded[:N, M // 32:]) == 0


@pytest.mark.parametrize("kernel", ["mxfp8_dim_km_swizzle_v2"])
@pytest.mark.parametrize("M,N", [(128, 512), (96, 160)])
def test_mxfp8_dim_km_swizzle(kernel, M, N):
    if kernel in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{kernel} emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe(kernel)
    inputs = recipe.example_input_fn(M, N, torch.bfloat16)
    outputs = recipe.cute_fn(*inputs)
    ref_outputs = recipe.pt_ref_fn(*inputs)
    assert len(outputs) == 4
    for output, ref_output in zip(outputs, ref_outputs):
        assert qdata_and_scale_equal(output, ref_output)

    qk, sk, qm, sm = outputs
    assert qk.shape == (M, N)
    assert qm.shape == (N, M)
    nrb_k, ncb_k = (M + 127) // 128, ((N // 32) + 3) // 4
    nrb_m, ncb_m = (N + 127) // 128, ((M // 32) + 3) // 4
    assert sk.shape == (nrb_k, ncb_k, 32, 16)
    assert sm.shape == (nrb_m, ncb_m, 32, 16)
    sk_unblocked = _from_blocked_4d(sk, nrb_k * 128, ncb_k * 4).view(torch.uint8)
    sm_unblocked = _from_blocked_4d(sm, nrb_m * 128, ncb_m * 4).view(torch.uint8)
    assert torch.count_nonzero(sk_unblocked[M:, :]) == 0
    assert torch.count_nonzero(sk_unblocked[:M, N // 32:]) == 0
    assert torch.count_nonzero(sm_unblocked[N:, :]) == 0
    assert torch.count_nonzero(sm_unblocked[:N, M // 32:]) == 0


@pytest.mark.parametrize("kernel", ["mxfp8_dim_km_swizzle_v2"])
def test_mxfp8_dim_km_swizzle_rejects_partial_group(kernel):
    if kernel in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{kernel} emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe(kernel)
    inputs = recipe.example_input_fn(96, 144, torch.bfloat16)
    with pytest.raises(ValueError, match="K % 32"):
        recipe.cute_fn(*inputs)


@pytest.mark.parametrize(
    "kernel,M,K",
    [
        ("nvfp4_dim_m_rht_swizzle_pipelined", 256, 384),
        ("nvfp4_dim_m_rht_swizzle_pipelined", 384, 256),
        ("nvfp4_dim_m_swizzle_rht_sr_pipelined", 256, 384),
        ("nvfp4_dim_m_swizzle_rht_sr_pipelined", 384, 256),
    ],
)
def test_nvfp4_dim_m_rht_pipelined(kernel, M, K):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{kernel} emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe(kernel)
    cute_inputs, gold_inputs = _nvfp4_dim_m_rht_test_inputs(
        M,
        K,
        stochastic="swizzle_rht_sr" in kernel,
    )
    outputs = recipe.cute_fn(*cute_inputs)
    assert tuple(output.shape for output in outputs) == (
        (K, M // 2),
        (K // 128, M // 64, 32, 16),
    )
    recipe.correctness_fn(gold_inputs, outputs)


@pytest.mark.parametrize(
    "kernel,M,K",
    [
        ("nvfp4_swizzle_direct", 1, 16),
        ("nvfp4_swizzle_direct", 128, 64),
        ("nvfp4_swizzle_direct", 33, 80),
        ("nvfp4_swizzle_direct", 129, 144),
        ("nvfp4_swizzle_tma", 1, 32),
        ("nvfp4_swizzle_tma", 33, 96),
        ("nvfp4_swizzle_tma", 129, 160),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_nvfp4_swizzle_cute_hand(kernel, M, K, dtype):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{kernel} emits Blackwell-only PTX; requires cuda capability 10.0")
    if kernel == "nvfp4_swizzle_direct" and dtype != torch.bfloat16:
        pytest.skip("nvfp4_swizzle_direct is bf16-only")
    recipe = _get_recipe(kernel)
    inputs = recipe.example_input_fn(M, K, dtype)
    outputs = recipe.cute_fn(*inputs)
    ref_outputs = recipe.pt_ref_fn(*inputs)
    assert qdata_and_scale_equal(outputs[0], ref_outputs[0])
    assert qdata_and_scale_equal(outputs[1], ref_outputs[1])
    assert outputs[0].shape == (M, K // 2)

    nrb, ncb = (M + 127) // 128, ((K // 16) + 3) // 4
    assert outputs[1].shape == (nrb, ncb, 32, 16)
    padded = _from_blocked_4d(outputs[1], nrb * 128, ncb * 4).view(torch.uint8)
    assert torch.count_nonzero(padded[M:, :]) == 0
    assert torch.count_nonzero(padded[:M, K // 16:]) == 0


def test_nvfp4_swizzle_tma_rejects_unaligned_packed_stride():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("nvfp4_swizzle_tma emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("nvfp4_swizzle_tma")
    inputs = recipe.example_input_fn(32, 16, torch.bfloat16)
    with pytest.raises(ValueError, match="K % 32"):
        recipe.cute_fn(*inputs)


def test_nvfp4_swizzle_tma_rejects_rht_argument():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("nvfp4_swizzle_tma emits Blackwell-only PTX; requires cuda capability 10.0")
    x = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
    outer_scale = torch.ones(1, dtype=torch.float32, device=x.device)
    rht_sign = torch.ones(16, dtype=torch.bfloat16, device=x.device)
    with pytest.raises(ValueError, match="unexpected keyword arguments: rht_sign"):
        nvfp4_swizzle_tma(x, outer_scale, rht_sign=rht_sign)


@pytest.mark.parametrize(
    "kwargs,error,exception_type",
    [
        (
            {"quant_orientation": "rows"},
            "unsupported quant_orientation",
            ValueError,
        ),
        ({"quant_orientation": "dim_m"}, "M % 32", ValueError),
        ({"mode": "dim_m"}, "unexpected keyword arguments: mode", ValueError),
        ({"outer_scale": None}, "dim-K outer scale is required", ValueError),
        (
            {"outer_scale": 1.0},
            "dim-K outer scale must be a torch.Tensor",
            TypeError,
        ),
    ],
)
def test_nvfp4_swizzle_tma_rejects_invalid_arguments(
    kwargs, error, exception_type
):
    x = torch.randn(31, 32, dtype=torch.bfloat16, device="cuda")
    outer_scale = torch.ones(1, dtype=torch.float32, device=x.device)
    kwargs = {"outer_scale": outer_scale, **kwargs}
    with pytest.raises(exception_type, match=error):
        nvfp4_swizzle_tma(x, **kwargs)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "quant_orientation,M,K,expected_shapes",
    [
        ("dim_k", 0, 32, ((0, 16), (0, 1, 32, 16))),
        ("dim_k", 1, 0, ((1, 0), (1, 0, 32, 16))),
        ("dim_m", 0, 16, ((16, 0), (1, 0, 32, 16))),
        ("dim_m", 32, 0, ((0, 16), (0, 1, 32, 16))),
        (
            "dim_km",
            0,
            32,
            ((0, 16), (0, 1, 32, 16), (32, 0), (1, 0, 32, 16)),
        ),
        (
            "dim_km",
            32,
            0,
            ((32, 0), (1, 0, 32, 16), (0, 16), (0, 1, 32, 16)),
        ),
        ("dim_km", 0, 0, ((0, 0), (0, 0, 32, 16)) * 2),
    ],
)
def test_nvfp4_swizzle_tma_empty(dtype, quant_orientation, M, K, expected_shapes):
    x = torch.empty(M, K, dtype=dtype, device="cuda")
    outer_scale = torch.ones(1, dtype=torch.float32, device=x.device)
    outputs = nvfp4_swizzle_tma(
        x,
        outer_scale,
        outer_scale_m=outer_scale if quant_orientation == "dim_km" else None,
        quant_orientation=quant_orientation,
    )

    assert tuple(output.shape for output in outputs) == expected_shapes
    assert all(output.numel() == 0 for output in outputs)
    assert tuple(output.dtype for output in outputs) == (
        (torch.float4_e2m1fn_x2, torch.float8_e4m3fn)
        if quant_orientation != "dim_km"
        else (
            torch.float4_e2m1fn_x2,
            torch.float8_e4m3fn,
            torch.float4_e2m1fn_x2,
            torch.float8_e4m3fn,
        )
    )


@pytest.mark.parametrize(
    "kernel,M,K",
    [
        ("nvfp4_dim_m_swizzle_tma", 32, 16),
        ("nvfp4_dim_m_swizzle_tma", 96, 48),
        ("nvfp4_dim_m_swizzle_tma", 160, 144),
        ("nvfp4_dim_km_swizzle_tma", 32, 32),
        ("nvfp4_dim_km_swizzle_tma", 96, 160),
        ("nvfp4_dim_km_swizzle_tma", 160, 96),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_nvfp4_dim_m_tma_padding(kernel, M, K, dtype):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{kernel} emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe(kernel)
    cute_inputs = gold_inputs = recipe.example_input_fn(M, K, dtype)
    outputs = recipe.cute_fn(*cute_inputs)
    references = recipe.pt_ref_fn(*gold_inputs)
    for output, reference in zip(outputs, references):
        assert qdata_and_scale_equal(output, reference)

    qm, sm = outputs[-2:]
    assert qm.shape == (K, M // 2)
    nrb_m, ncb_m = (K + 127) // 128, ((M // 16) + 3) // 4
    assert sm.shape == (nrb_m, ncb_m, 32, 16)
    sm_padded = _from_blocked_4d(sm, nrb_m * 128, ncb_m * 4).view(torch.uint8)
    assert torch.count_nonzero(sm_padded[K:, :]) == 0
    assert torch.count_nonzero(sm_padded[:K, M // 16:]) == 0

    if kernel == "nvfp4_dim_km_swizzle_tma":
        qk, sk = outputs[:2]
        assert qk.shape == (M, K // 2)
        nrb_k, ncb_k = (M + 127) // 128, ((K // 16) + 3) // 4
        assert sk.shape == (nrb_k, ncb_k, 32, 16)
        sk_padded = _from_blocked_4d(sk, nrb_k * 128, ncb_k * 4).view(
            torch.uint8
        )
        assert torch.count_nonzero(sk_padded[M:, :]) == 0
        assert torch.count_nonzero(sk_padded[:M, K // 16:]) == 0


@pytest.mark.parametrize("M,K", [(32, 32), (2080, 2080)])
def test_nvfp4_dim_km_tma_scale_stores_stay_within_allocations(
    monkeypatch, M, K
):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(
            "nvfp4 dim-KM TMA emits Blackwell-only PTX; requires cuda capability 10.0"
        )

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    outer_scale = torch.ones(1, dtype=torch.float32, device=x.device)

    # Compile before replacing torch.empty so compiler-internal allocations are unaffected.
    nvfp4_swizzle_tma(
        x,
        outer_scale,
        outer_scale_m=outer_scale,
        quant_orientation="dim_km",
    )
    torch.cuda.synchronize()

    real_empty = torch.empty
    sentinel = 0xA5
    guard_bytes = 1024
    allocation_alignment = 1024
    qdata_bytes = M * K
    scale_k_bytes = ((M + 127) // 128) * ((K + 63) // 64) * 32 * 16
    scale_m_bytes = ((K + 127) // 128) * ((M + 63) // 64) * 32 * 16
    arena_bytes = qdata_bytes + scale_k_bytes + scale_m_bytes + 8 * guard_bytes
    arena = real_empty(arena_bytes, dtype=torch.uint8, device=x.device)
    arena.fill_(sentinel)
    guarded_ranges = []
    next_offset = 0

    def guarded_empty(*size, **kwargs):
        nonlocal next_offset
        if kwargs.get("dtype") == torch.uint8 and kwargs.get("device") == x.device:
            shape = size[0] if len(size) == 1 and isinstance(size[0], tuple) else size
            numel = math.prod(shape)
            start = (
                (next_offset + allocation_alignment - 1) // allocation_alignment
            ) * allocation_alignment
            end = start + numel
            guard_end = end + guard_bytes
            assert guard_end <= arena.numel()
            guarded_ranges.append((end, guard_end))
            next_offset = guard_end
            return arena[start:end].view(shape)
        return real_empty(*size, **kwargs)

    monkeypatch.setattr(torch, "empty", guarded_empty)
    nvfp4_swizzle_tma(
        x,
        outer_scale,
        outer_scale_m=outer_scale,
        quant_orientation="dim_km",
    )
    torch.cuda.synchronize()

    assert len(guarded_ranges) == 4
    for start, end in guarded_ranges:
        assert torch.all(arena[start:end] == sentinel)


@pytest.mark.parametrize(
    "kernel,M,K",
    [
        ("nvfp4_swizzle_dim_k_dim_m_rht_pipelined", 96, 128),
        ("nvfp4_swizzle_dim_k_dim_m_rht_pipelined", 128, 160),
        ("nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined", 96, 128),
        ("nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined", 128, 160),
        ("nvfp4_dim_m_rht_swizzle_pipelined", 96, 128),
        ("nvfp4_dim_m_rht_swizzle_pipelined", 128, 160),
        ("nvfp4_dim_m_swizzle_rht_sr_pipelined", 96, 128),
        ("nvfp4_dim_m_swizzle_rht_sr_pipelined", 128, 160),
    ],
)
def test_nvfp4_rht_pipelined_requires_full_tiles(kernel, M, K):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{kernel} emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe(kernel)
    if "dim_k" in kernel:
        cute_inputs, _ = _nvfp4_dim_km_rht_test_inputs(
            M,
            K,
            stochastic="_sr_dim_m_" in kernel,
        )
    else:
        cute_inputs, _ = _nvfp4_dim_m_rht_test_inputs(
            M,
            K,
            stochastic="_sr_pipelined" in kernel,
        )
    with pytest.raises(AssertionError, match="M % 128.*K % 128"):
        recipe.cute_fn(*cute_inputs)


@pytest.mark.parametrize("M,K", [(1, 32), (31, 96), (33, 160), (127, 64), (129, 160)])
def test_mxfp8_swizzle_v4_padding(M, K):
    if "mxfp8_swizzle_v4" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v4 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v4")
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)

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
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)
    with pytest.raises(AssertionError):
        recipe.cute_fn(*inputs)


def test_mxfp8_swizzle_v5():
    # Best of v1 and v4: v1's flat 1-D grid + v4's 16-elem/thread load (two LDG.128 concat'd in
    # registers) and single STG.128 store.
    if "mxfp8_swizzle_v5" in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("mxfp8_swizzle_v5 emits Blackwell-only PTX; requires cuda capability 10.0")
    recipe = _get_recipe("mxfp8_swizzle_v5")
    inputs = recipe.example_input_fn(128, 512, torch.bfloat16)
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
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)

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
    inputs = recipe.example_input_fn(M, K, torch.bfloat16)
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
    inputs = recipe.example_input_fn(256, 512, torch.bfloat16)
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
    inputs = recipe.example_input_fn(256, 512, torch.bfloat16)
    print(inputs[0].shape)
    print(inputs)

    outputs = recipe.cute_fn(*inputs)
    print(outputs)

    tile_kwargs = {"global_row": 0, "global_col": 0, "num_col": inputs[0].shape[-1]}
    ref_outputs = recipe.pt_ref_fn(*inputs, **tile_kwargs)
    print(ref_outputs)
    recipe.correctness_fn(inputs, outputs)

@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=["bf16", "fp16", "fp32"],
)
@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_cute_hand_matches_reference(name, recipe, dtype):
    # the CuTeDSL kernel should reproduce the gold reference bit-for-bit (identical fp32 math + RNE
    # cast). example_input_fn builds the full positional inputs (x, *aux).
    if name in _REQUIRES_SM100 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip(f"{name} emits Blackwell-only PTX; requires cuda capability 10.0")
    if dtype != torch.bfloat16 and name not in _SUPPORTS_NON_BFLOAT16:
        pytest.skip(f"{name} does not support {dtype} input")
    torch.manual_seed(0)
    if name in (
        "nvfp4_swizzle_dim_k_dim_m_rht_pipelined",
        "nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined",
    ):
        cute_inputs, gold_inputs = _nvfp4_dim_km_rht_test_inputs(
            512,
            512,
            stochastic="_sr_dim_m_" in name,
        )
    elif name in (
        "nvfp4_dim_m_rht_swizzle_pipelined",
        "nvfp4_dim_m_swizzle_rht_sr_pipelined",
    ):
        cute_inputs, gold_inputs = _nvfp4_dim_m_rht_test_inputs(
            512,
            512,
            stochastic="swizzle_rht_sr" in name,
        )
    else:
        gold_inputs = recipe.example_input_fn(512, 512, dtype)
        cute_inputs = tuple(
            value.clone_state()
            if isinstance(value, torch.Generator)
            else value
            for value in gold_inputs
        )

    tile_kwargs = {
        "global_row": 0,
        "global_col": 0,
        "num_col": gold_inputs[0].shape[-1],
    }
    ref_outs = recipe.pt_ref_fn(*gold_inputs, **tile_kwargs)
    cute_kwargs = {} if name in _BLOCKSCALED_TMA_RECIPES else tile_kwargs
    cute_outs = recipe.cute_fn(*cute_inputs, **cute_kwargs)

    assert len(cute_outs) == len(ref_outs), f"{name}: output count {len(cute_outs)} != {len(ref_outs)}"
    for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )

    # Every recipe's outputs must be a valid quantization (the gold correctness_fn).
    recipe.correctness_fn(gold_inputs, cute_outs)

    # The TE-style pipelined variants deliberately use BF16 UMMA and approximate reciprocal math;
    # validate their quantization quality above, but do not impose the other kernels' bitwise rule.
    if not name.endswith("_pipelined"):
        for i, (t, r) in enumerate(zip(cute_outs, ref_outs)):
            assert qdata_and_scale_equal(t, r), (
                f"{name} output {i}: {mismatch_fraction(t, r):.3%} of elements differ from the gold "
                f"reference -- expected bit-for-bit equality"
            )
