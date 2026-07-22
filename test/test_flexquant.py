import os
import sys

import pytest
import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_bench.flexquant.api import _HopMode, flex_cast_quant_dense
from quant_cast_bench.flexquant.api_triton_for_debugging import flex_cast_quant_dense_triton
from quant_cast_bench.flexquant.recipe_debug_triton import (
    RecipeTriton,
    deepseek_fp8_128_128_triton,
    deepseek_fp8_1_128_dim_m_triton,
)
from quant_cast_bench.flexquant.recipes import (
    Recipe,
    deepseek_fp8_1_128,
    deepseek_fp8_1_128_dim_m,
    deepseek_fp8_128_128,
    mxfp8_floor,
    mxfp8_floor_dim_m,
    mxfp8_floor_swizzle,
    nvfp4_no_gs,
    nvfp4_no_gs_lut,
    nvfp4_no_gs_swizzle,
    nvfp4_with_gs,
)

# (label, recipe, hop_mode). Recipes that support both HOP and non-HOP routes
# are listed twice with the two modes.
RECIPES_PT: list[tuple[str, Recipe, _HopMode]] = [
    ("deepseek_fp8_1_128", deepseek_fp8_1_128, _HopMode.NO_HOP),
    ("deepseek_fp8_1_128_dim_m", deepseek_fp8_1_128_dim_m, _HopMode.NO_HOP),
    ("deepseek_fp8_1_128_dim_m_hop", deepseek_fp8_1_128_dim_m, _HopMode.HOP),
    ("deepseek_fp8_128_128", deepseek_fp8_128_128, _HopMode.NO_HOP),
    ("deepseek_fp8_128_128_hop", deepseek_fp8_128_128, _HopMode.HOP),
    ("nvfp4_no_gs", nvfp4_no_gs, _HopMode.NO_HOP),
    ("nvfp4_no_gs_lut", nvfp4_no_gs_lut, _HopMode.NO_HOP),
    ("nvfp4_no_gs_swizzle", nvfp4_no_gs_swizzle, _HopMode.NO_HOP),
    ("nvfp4_with_gs", nvfp4_with_gs, _HopMode.NO_HOP),
    ("mxfp8_floor", mxfp8_floor, _HopMode.NO_HOP),
    ("mxfp8_floor_dim_m", mxfp8_floor_dim_m, _HopMode.NO_HOP),
    ("mxfp8_floor_swizzle", mxfp8_floor_swizzle, _HopMode.NO_HOP),
]
RECIPES_TRITON: list[tuple[str, RecipeTriton]] = [
    ("deepseek_fp8_1_128_dim_m_triton", deepseek_fp8_1_128_dim_m_triton),
    ("deepseek_fp8_128_128_triton", deepseek_fp8_128_128_triton),
]


@pytest.fixture(autouse=True)
def _reset_dynamo():
    # The Dynamo cache is keyed on flex_cast_quant_dense's code object, which
    # is shared across every parametrized test in this file. Without a reset,
    # cache entries from earlier tests accumulate and trip recompile_limit (8).
    torch._dynamo.reset()


def _call_pt(recipe: Recipe, hop_mode: _HopMode, x: torch.Tensor, fn=None):
    pt_fn = fn if fn is not None else flex_cast_quant_dense
    return pt_fn(
        x,
        block_size=recipe.block_size,
        dim=recipe.dim,
        qdata_dtype=recipe.qdata_dtype,
        scale_dtype=recipe.scale_dtype,
        amax_to_scale_fn=recipe.amax_to_scale_fn,
        cast_to_dtype_fn=recipe.cast_to_dtype_fn,
        scale_swizzle=recipe.scale_swizzle,
        _hop_mode=hop_mode,
    )


def _call_triton(recipe: RecipeTriton, x: torch.Tensor):
    return flex_cast_quant_dense_triton(
        x,
        block_size=recipe.block_size,
        dim=recipe.dim,
        qdata_dtype=recipe.qdata_dtype,
        scale_dtype=recipe.scale_dtype,
        amax_to_scale_fn_triton=recipe.amax_to_scale_fn,
        cast_to_dtype_fn_triton=recipe.cast_to_dtype_fn,
    )


@pytest.mark.parametrize(
    "label,recipe,hop_mode", RECIPES_PT, ids=[label for label, _, _ in RECIPES_PT]
)
def test_pt_eager_vs_reference(label: str, recipe: Recipe, hop_mode: _HopMode):
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata, scale = _call_pt(recipe, hop_mode, x)
    qdata_ref, scale_ref = recipe._reference_fn(x)

    if qdata.dtype == torch.float4_e2m1fn_x2:
        # `.to(torch.float32)` isn't implemented for float4_e2m1fn_x2; compare
        # the underlying packed bytes instead.
        assert torch.equal(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
    else:
        assert torch.equal(qdata.to(torch.float32), qdata_ref.to(torch.float32))

    # Two-level scaling returns a list [outer_scale, inner_scale].
    if isinstance(scale, list):
        assert isinstance(scale_ref, list) and len(scale) == len(scale_ref)
        for s, s_ref in zip(scale, scale_ref):
            assert torch.equal(s, s_ref)
    else:
        assert torch.equal(scale, scale_ref)


@pytest.mark.parametrize(
    "label,recipe", RECIPES_TRITON, ids=[label for label, _ in RECIPES_TRITON]
)
def test_triton_eager_vs_reference(label: str, recipe: RecipeTriton):
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata, scale = _call_triton(recipe, x)
    qdata_ref, scale_ref = recipe._reference_fn(x)

    assert torch.equal(qdata.to(torch.float32), qdata_ref.to(torch.float32))
    assert torch.equal(scale, scale_ref)


@pytest.mark.parametrize(
    "label,recipe,hop_mode", RECIPES_PT, ids=[label for label, _, _ in RECIPES_PT]
)
def test_eager_vs_compile(label: str, recipe: Recipe, hop_mode: _HopMode):
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata_eager, scale_eager = _call_pt(recipe, hop_mode, x)

    compiled = torch.compile(flex_cast_quant_dense, fullgraph=True)
    qdata_compiled, scale_compiled = _call_pt(recipe, hop_mode, x, fn=compiled)

    if qdata_eager.dtype == torch.float4_e2m1fn_x2:
        # `.to(torch.float32)` isn't implemented for float4_e2m1fn_x2; compare
        # the underlying packed bytes instead.
        assert torch.equal(
            qdata_eager.view(torch.uint8), qdata_compiled.view(torch.uint8)
        )
    else:
        assert torch.equal(
            qdata_eager.to(torch.float32), qdata_compiled.to(torch.float32)
        )

    scales_eager = scale_eager if isinstance(scale_eager, list) else [scale_eager]
    scales_compiled = (
        scale_compiled if isinstance(scale_compiled, list) else [scale_compiled]
    )
    assert len(scales_eager) == len(scales_compiled)
    for s_eager, s_compiled in zip(scales_eager, scales_compiled):
        assert torch.equal(s_eager, s_compiled)


@pytest.mark.parametrize(
    "label,recipe,hop_mode", RECIPES_PT, ids=[label for label, _, _ in RECIPES_PT]
)
def test_eager_vs_compile_cuda_graph(label: str, recipe: Recipe, hop_mode: _HopMode):
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata_eager, scale_eager = _call_pt(recipe, hop_mode, x)

    compiled = torch.compile(
        flex_cast_quant_dense, fullgraph=True, mode="reduce-overhead"
    )
    # cudagraph-trees needs a warmup invocation before the captured replay;
    # the second call returns cudagraph-managed outputs.
    _ = _call_pt(recipe, hop_mode, x, fn=compiled)
    qdata_cg, scale_cg = _call_pt(recipe, hop_mode, x, fn=compiled)

    # Outputs from cudagraph-trees are freed on the next captured replay;
    # clone so later assertions are stable if someone extends the test.
    qdata_cg = qdata_cg.clone()
    scale_cg = (
        [s.clone() for s in scale_cg]
        if isinstance(scale_cg, list)
        else scale_cg.clone()
    )

    if qdata_eager.dtype == torch.float4_e2m1fn_x2:
        assert torch.equal(qdata_eager.view(torch.uint8), qdata_cg.view(torch.uint8))
    else:
        assert torch.equal(qdata_eager.to(torch.float32), qdata_cg.to(torch.float32))

    scales_eager = scale_eager if isinstance(scale_eager, list) else [scale_eager]
    scales_cg = scale_cg if isinstance(scale_cg, list) else [scale_cg]
    assert len(scales_eager) == len(scales_cg)
    for s_eager, s_cg in zip(scales_eager, scales_cg):
        assert torch.equal(s_eager, s_cg)


def test_no_hop_fuses_with_preceding_pointwise():
    recipe = deepseek_fp8_1_128

    def fn(x):
        flex_cast_quant_dense_c = torch.compile(flex_cast_quant_dense)
        return flex_cast_quant_dense_c(
            F.relu(x),
            block_size=recipe.block_size,
            dim=recipe.dim,
            qdata_dtype=recipe.qdata_dtype,
            scale_dtype=recipe.scale_dtype,
            amax_to_scale_fn=recipe.amax_to_scale_fn,
            cast_to_dtype_fn=recipe.cast_to_dtype_fn,
        )

    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    compiled = torch.compile(fn, fullgraph=True)
    _, code = run_and_get_code(compiled, x)
    triton_code = "\n".join(code)

    # If relu fuses into the quant kernel, there is exactly one @triton.jit
    # kernel emitted. A second kernel would mean Inductor failed to fuse.
    FileCheck().check_count("@triton.jit", 1, exactly=True).run(triton_code)

def test_hop_works_with_preceding_pointwise():
    recipe = deepseek_fp8_128_128

    def fn(x):
        flex_cast_quant_dense_c = torch.compile(flex_cast_quant_dense)
        return flex_cast_quant_dense_c(
            F.relu(x),
            block_size=recipe.block_size,
            dim=recipe.dim,
            qdata_dtype=recipe.qdata_dtype,
            scale_dtype=recipe.scale_dtype,
            amax_to_scale_fn=recipe.amax_to_scale_fn,
            cast_to_dtype_fn=recipe.cast_to_dtype_fn,
        )

    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    compiled = torch.compile(fn, fullgraph=True)
    y = compiled(x)
