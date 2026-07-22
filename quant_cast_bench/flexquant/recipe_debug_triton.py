"""Triton-flavored recipes for debugging and benchmarking only.

Mirrors :class:`recipes.Recipe` but with the ``amax_to_scale_fn`` /
``cast_to_dtype_fn`` callbacks defined directly as ``@triton.jit`` functions,
so they can be passed straight into a hand-written Triton kernel via
:func:`api_triton_for_debugging.flex_cast_quant_dense_triton`. These recipes
are not for production use — they exist so we can A/B the HOP/Inductor path
against the explicit Triton path on the same tilings.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl

from .recipes import (
    _deepseek_fp8_128_128_reference,
    _deepseek_fp8_1_128_dim_m_reference,
)


@dataclass(frozen=True)
class RecipeTriton:
    name: str
    block_size: int | tuple[int, int]
    dim: int | tuple[int, int]
    qdata_dtype: torch.dtype
    scale_dtype: torch.dtype
    amax_to_scale_fn: Callable
    cast_to_dtype_fn: Callable

    # Plain PyTorch reference implementation (for debugging only; matches the
    # corresponding production recipe's _reference_fn).
    _reference_fn: Callable


@triton.jit
def _deepseek_fp8_amax_to_scale_fn_triton(amax):
    EPS: tl.constexpr = 1e-12
    FP8_MAX_C: tl.constexpr = 448.0
    amax_fp32 = tl.maximum(amax.to(tl.float32), EPS)
    return amax_fp32 / FP8_MAX_C


@triton.jit
def _deepseek_fp8_cast_to_dtype_fn_triton(tile, scale):
    y = tile * (1.0 / scale)
    return y.to(tl.float8e4nv)


deepseek_fp8_1_128_dim_m_triton = RecipeTriton(
    name="deepseek_fp8_1_128_dim_m_triton",
    block_size=128,
    dim=-2,
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn_triton,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn_triton,
    _reference_fn=_deepseek_fp8_1_128_dim_m_reference,
)

deepseek_fp8_128_128_triton = RecipeTriton(
    name="deepseek_fp8_128_128_triton",
    block_size=(128, 128),
    dim=(-2, -1),
    qdata_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.float32,
    amax_to_scale_fn=_deepseek_fp8_amax_to_scale_fn_triton,
    cast_to_dtype_fn=_deepseek_fp8_cast_to_dtype_fn_triton,
    _reference_fn=_deepseek_fp8_128_128_reference,
)
