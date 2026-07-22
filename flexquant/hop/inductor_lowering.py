"""Inductor lowering for FlexQuantHOP.

Picks one of the per-tiling Triton templates, builds subgraph buffers from the
two captured callbacks, and registers a kernel choice via maybe_append_choice.
Modeled after torch._inductor.kernel.flex.flex_attention:flex_attention.
"""

import os
from typing import Any

import torch
from torch._inductor.ir import FixedLayout
from torch._inductor.kernel.flex.common import (
    build_subgraph_buffer,
    create_placeholder,
    freeze_irnodes,
    maybe_realize,
)
from torch._inductor.lowering import empty_strided, register_lowering
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
    TritonTemplate,
)

from .hop import (
    _TILING_128_128,
    _TILING_1_128_DIM_M,
    _tiling_key,
    flex_cast_quant_dense_with_hop,
)


_HERE = os.path.dirname(__file__)


def _read_template(name: str) -> str:
    with open(os.path.join(_HERE, name)) as f:
        return f.read()


# ---- 128x128 weight-quant template ----------------------------------------


@SymbolicGridFn
def _grid_128_128(M, N, meta, *, cdiv):
    return (cdiv(M, meta["BLOCK_SIZE"]), cdiv(N, meta["BLOCK_SIZE"]), 1)


_TEMPLATE_128_128 = TritonTemplate(
    name="flex_quant_128_128",
    grid=_grid_128_128,
    source=_read_template("template_128_128.py.jinja"),
)


_AUTOTUNE_CONFIGS_128_128 = [
    {"BLOCK_SIZE": 128, "num_warps": w, "num_stages": s}
    for w in (4, 8)
    for s in (2, 4)
]


def _lower_128_128(x, amax_subgraph, cast_subgraph, block_size, qdata_dtype, scale_dtype):
    device = x.get_device()
    M = x.get_size()[0]
    N = x.get_size()[1]
    B1, B2 = block_size

    qdata_layout = FixedLayout(
        device,
        qdata_dtype,
        [M, N],
        stride=[N, 1],
    )
    n1 = (M + B1 - 1) // B1
    n2 = (N + B2 - 1) // B2
    scale = empty_strided([n1, n2], None, dtype=scale_dtype, device=device)

    amax_placeholders = [create_placeholder("amax", x.get_dtype(), device)]
    cast_placeholders = [
        create_placeholder("tile", x.get_dtype(), device),
        create_placeholder("scale", scale_dtype, device),
    ]

    amax_buffer = build_subgraph_buffer(amax_placeholders, amax_subgraph)
    freeze_irnodes(amax_buffer)
    cast_buffer = build_subgraph_buffer(cast_placeholders, cast_subgraph)
    freeze_irnodes(cast_buffer)

    choices: list[Any] = []
    for cfg in _AUTOTUNE_CONFIGS_128_128:
        _TEMPLATE_128_128.maybe_append_choice(
            choices=choices,
            input_nodes=[x, scale],
            layout=qdata_layout,
            subgraphs=[amax_buffer, cast_buffer],
            mutated_inputs=[scale],
            call_sizes=[M, N],
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    qdata, _ = autotune_select_algorithm(
        "flex_quant_128_128",
        choices,
        [x, scale],
        qdata_layout,
    )
    return (qdata, scale)


# ---- 1x128 dim_m act-quant template ---------------------------------------


@SymbolicGridFn
def _grid_1_128_dim_m(M, K, meta, *, cdiv):
    return (cdiv(M, meta["BLOCK_SIZE"]), cdiv(K, meta["NUM_GROUPS"]), 1)


_TEMPLATE_1_128_DIM_M = TritonTemplate(
    name="flex_quant_1_128_dim_m",
    grid=_grid_1_128_dim_m,
    source=_read_template("template_1_128_dim_m.py.jinja"),
)


_AUTOTUNE_CONFIGS_1_128_DIM_M = [
    {"BLOCK_SIZE": 128, "NUM_GROUPS": g, "num_warps": w, "num_stages": s}
    for g in (2, 16, 32, 64, 128)
    for w in (2, 4, 8)
    for s in (2, 4, 6)
]


def _lower_1_128_dim_m(x, amax_subgraph, cast_subgraph, block_size, qdata_dtype, scale_dtype):
    device = x.get_device()
    M = x.get_size()[0]
    K = x.get_size()[1]
    B = int(block_size)

    # Output qdata is (K, M) row-major, scale is (K, M // B) row-major.
    qdata_layout = FixedLayout(
        device,
        qdata_dtype,
        [K, M],
        stride=[M, 1],
    )
    n_m = (M + B - 1) // B
    scale = empty_strided([K, n_m], None, dtype=scale_dtype, device=device)

    amax_placeholders = [create_placeholder("amax", x.get_dtype(), device)]
    cast_placeholders = [
        create_placeholder("tile", x.get_dtype(), device),
        create_placeholder("scale", scale_dtype, device),
    ]

    amax_buffer = build_subgraph_buffer(amax_placeholders, amax_subgraph)
    freeze_irnodes(amax_buffer)
    cast_buffer = build_subgraph_buffer(cast_placeholders, cast_subgraph)
    freeze_irnodes(cast_buffer)

    choices: list[Any] = []
    for cfg in _AUTOTUNE_CONFIGS_1_128_DIM_M:
        _TEMPLATE_1_128_DIM_M.maybe_append_choice(
            choices=choices,
            input_nodes=[x, scale],
            layout=qdata_layout,
            subgraphs=[amax_buffer, cast_buffer],
            mutated_inputs=[scale],
            call_sizes=[M, K],
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            NUM_GROUPS=cfg["NUM_GROUPS"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    qdata, _ = autotune_select_algorithm(
        "flex_quant_1_128_dim_m",
        choices,
        [x, scale],
        qdata_layout,
    )
    return (qdata, scale)


# ---- Dispatcher -----------------------------------------------------------


@register_lowering(flex_cast_quant_dense_with_hop, type_promotion_kind=None)
def _flex_quant_lowering(
    x,
    amax_subgraph,
    cast_subgraph,
    block_size,
    dim,
    qdata_dtype,
    scale_dtype,
):
    """Pick the right Triton template based on the (block_size, dim) tiling."""
    assert qdata_dtype in (torch.float8_e4m3fn, torch.float32), (
        f"unsupported qdata_dtype: {qdata_dtype!r}"
    )
    assert scale_dtype == torch.float32, f"unsupported scale_dtype: {scale_dtype!r}"

    # Realize x so it has concrete strides; otherwise an unrealized Pointwise
    # (e.g. a fused preceding op like relu) has no stride info and every
    # template choice gets filtered out.
    (x,) = maybe_realize([x])

    key = _tiling_key(block_size, dim)
    if key == _TILING_128_128:
        return _lower_128_128(
            x, amax_subgraph, cast_subgraph, tuple(block_size), qdata_dtype, scale_dtype
        )
    if key == _TILING_1_128_DIM_M:
        return _lower_1_128_dim_m(
            x, amax_subgraph, cast_subgraph, int(block_size), qdata_dtype, scale_dtype
        )
    raise NotImplementedError(
        f"flex_quant lowering: unsupported tiling (block_size={block_size!r}, dim={dim!r})"
    )
