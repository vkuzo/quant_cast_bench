"""Inductor lowering for FlexTileMapHOP.

One path: a group-reduction `f` (deepseek fp8 1x128 dim-M) is walked by FxTritonEmitter, which
emits a Triton body string that we str.replace into the dim-M template's __EMITTER_BODY__ hole.
This bypasses Inductor's pointwise-only PointwiseSubgraphLowering (which raises on a reduction);
the group reduction lowers to a static tl.max and the transposed outputs to a tl.trans. Two
outputs: qdata (primary) + per-group scale (a mutated input), mirroring flexquant v1.

  ---- REMOVED: pointwise path (naive elementwise template) ----
  There used to be a second branch here for a pointwise `f` (e.g. relu): it built an Inductor
  subgraph buffer (`build_subgraph_buffer`) and inlined it at a naive template's `{{ modification }}`
  hole (`template_naive.py.jinja`, now also deleted), autotuning over BLOCK_SIZE. It was removed
  because a pointwise cast doesn't need this HOP+template machinery at all -- regular Inductor
  lowers plain pointwise ops fine, so a pointwise `f` should just be written as ordinary PyTorch
  and left to torch.compile. Only the reduction case (which Inductor's pointwise subgraph lowering
  genuinely can't express) still routes through this custom lowering. The HOP's eager body still
  runs any `f` directly (`hop.py::_flex_tile_map_eager`), so uncompiled pointwise use is unaffected.
"""

import hashlib
import os
from typing import Any

import torch
from torch._inductor.ir import FixedLayout
from torch._inductor.kernel.flex.common import maybe_realize
from torch._inductor.lowering import empty_strided, register_lowering
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
    TritonTemplate,
)

from .fx_triton_emitter import _FUNCTION_REDUCTIONS, FxTritonEmitter
from .hop import flex_tile_map_hop


_HERE = os.path.dirname(__file__)


def _read_template(name: str) -> str:
    with open(os.path.join(_HERE, name)) as f:
        return f.read()


# ---- reduction path (FxTritonEmitter -> dim-M template) --------------------

# dim-M templates, keyed by the reduction group width detected in `f`. Each is a hand-written
# template with the group baked in (128 for deepseek 1x128, 32 for mxfp8-floor 1x32); they will be
# unified into one group-parameterized template later (see future_ideas.md). To wire a new
# group-reduction recipe, add its template file + group here.
_DIM_M_TEMPLATES = {
    128: "template_deepseek_dim_m.py.jinja",
    32: "template_mxfp8_floor_dim_m.py.jinja",
}


@SymbolicGridFn
def _grid_reduce(M, N, meta, *, cdiv):
    return (cdiv(M, meta["BLOCK_M"]), cdiv(N, meta["BLOCK_N"]), 1)


def _splice_body(body: str, template_name: str) -> tuple[str, str]:
    """Build a template source with `body` spliced at the hole; return (name, source).

    The body can't be passed as a template kwarg (every kwarg is emitted as
    `name : tl.constexpr = value`), so we str.replace it into the source before jinja parsing.
    The template's `__EMITTER_BODY__` token sits at a 4-space indent, so the first body line
    inherits it; subsequent lines are indented explicitly (like Inductor's indent_except_first).
    """
    src = _read_template(template_name)
    lines = body.split("\n")
    indented = "\n".join([lines[0]] + ["    " + ln for ln in lines[1:]])
    src = src.replace("__EMITTER_BODY__", indented)
    # per-graph name so distinct `f` bodies don't collide in TritonTemplate.all_templates,
    # while an identical body (same source) dedups cleanly.
    name = f"flex_tile_map_reduce_{hashlib.sha256(src.encode()).hexdigest()[:12]}"
    return name, src


def _graph_output_dtypes(gm: torch.fx.GraphModule):
    """(qdata_dtype, scale_dtype) from the traced graph's two outputs (qdata, scale)."""
    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    qdata_node, scale_node = out_node.args[0]
    return qdata_node.meta["val"].dtype, scale_node.meta["val"].dtype


def _emit(x, gm: torch.fx.GraphModule, aux_names):
    """Walk `gm` with FxTritonEmitter and return (body, group, reduce_kind).

    to_dtype resolves the triton dtype via V.graph.get_current_device_or_throw(), which is only
    populated during device-specific codegen -- set it explicitly for the emit.
    """
    from torch._inductor.virtualized import V

    with V.graph.set_current_device(x.get_device()):
        body, group, reduce_kind = FxTritonEmitter(
            gm, output_names=["qdata_var", "scale_var"], aux_names=aux_names
        ).emit()
    if group is None:
        raise NotImplementedError("reduction path requires a group reshape in `f`")
    return body, group, reduce_kind


def _autotune(name, src, input_nodes, layout, mutated_inputs, call_sizes, configs):
    """Build the TritonTemplate, append every config as a choice, and autotune -> primary output."""
    template = TritonTemplate(name=name, grid=_grid_reduce, source=src)
    choices: list[Any] = []
    for cfg in configs:
        template.maybe_append_choice(
            choices=choices,
            input_nodes=input_nodes,
            layout=layout,
            mutated_inputs=mutated_inputs,
            call_sizes=call_sizes,
            **cfg,
        )
    out, _ = autotune_select_algorithm(name, choices, input_nodes, layout)
    return out


def _lower_reduction_dim_m(x, body, group, qdata_dtype, scale_dtype):
    """dim-M variant: reduce down rows in `group`-row groups and TRANSPOSE both outputs.

    BLOCK_M must be a multiple of the group; outputs are the transpose of the input: qdata (N, M),
    scale (N, M//group). The kernel tiles the input (M, N) but stores transposed tiles.
    """
    device = x.get_device()
    M, N = x.get_size()[0], x.get_size()[1]
    template_name = _DIM_M_TEMPLATES.get(group)
    if template_name is None:
        raise NotImplementedError(
            f"no dim-M template for reduction group={group} (have {sorted(_DIM_M_TEMPLATES)})"
        )
    name, src = _splice_body(body, template_name)
    qdata_layout = FixedLayout(device, qdata_dtype, [N, M], stride=[M, 1])
    scale = empty_strided([N, M // group], None, dtype=scale_dtype, device=device)
    configs = [
        {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": w, "num_stages": s}
        for bm in (group, group * 2)
        for bn in (32, 64, 128)
        for w in (4, 8)
        for s in (2, 4)
    ]
    qdata = _autotune(
        name, src, [x, scale], qdata_layout, [scale], [M, N], configs
    )
    return (qdata, scale)


def _lower_mxfp8_32x32(x, body, group, qdata_dtype, scale_dtype):
    """block_2d mxfp8 variant: one e8m0 scale per 32x32 SQUARE block, NO transpose.

    Reduces over the whole 32x32 block (both dims), so qdata keeps the input shape (M, N) and the
    scale is (M//32, N//32). Like the dim-M variants the PRIMARY (autotuned) output is the fp8
    qdata and the e8m0 scale is a mutated input. BLOCK_M and BLOCK_N must both be 32-multiples.
    """
    device = x.get_device()
    M, N = x.get_size()[0], x.get_size()[1]
    if group != 32:
        raise NotImplementedError(f"mxfp8 32x32 template expects a 32x32 block, got {group}")
    name, src = _splice_body(body, "template_mxfp8_32x32_floor.py.jinja")
    qdata_layout = FixedLayout(device, qdata_dtype, [M, N], stride=[N, 1])
    scale = empty_strided([M // 32, N // 32], [N // 32, 1], dtype=scale_dtype, device=device)
    configs = [
        {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": w, "num_stages": s}
        for bm in (32, 64, 128)
        for bn in (32, 64, 128, 256)
        for w in (4, 8)
        for s in (2, 4)
    ]
    qdata = _autotune(name, src, [x, scale], qdata_layout, [scale], [M, N], configs)
    return (qdata, scale)


def _lower_nvfp4(x, body, group, operands, qdata_dtype, scale_dtype):
    """dim-K nvfp4 variant: reduce along columns in `group`(=16)-element inner blocks, NO transpose.

    qdata is fp4-packed (M, N//2) (two e2m1 per byte); scale is the e4m3 inner scale (M, N//group).
    `operands[0]` is the per-tensor outer scale (REPLICATE aux), passed to the template as OUTER and
    loaded once. BLOCK_N must be a multiple of the 16-element inner block.

    Unlike the dim-M variants the PRIMARY (autotuned) output is the e4m3 SCALE, and the fp4-packed
    qdata is a mutated input: the autotuner zero_()s the primary buffer to benchmark each choice, and
    fill/zero_ is unimplemented for float4_e2m1fn_x2. Outputs are still returned (qdata, scale).
    """
    device = x.get_device()
    M, N = x.get_size()[0], x.get_size()[1]
    if group != 16:
        raise NotImplementedError(f"nvfp4 dim-K template expects a 16-element block, got {group}")
    if len(operands) != 1:
        raise NotImplementedError(
            f"nvfp4 dim-K template expects one aux (outer scale), got {len(operands)}"
        )
    (outer,) = maybe_realize(list(operands))

    name, src = _splice_body(body, "template_nvfp4.py.jinja")
    scale_layout = FixedLayout(device, scale_dtype, [M, N // group], stride=[N // group, 1])
    qdata = empty_strided([M, N // 2], [N // 2, 1], dtype=qdata_dtype, device=device)
    configs = [
        {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": w, "num_stages": s}
        for bm in (32, 64, 128)
        for bn in (16, 32, 64, 128)
        for w in (4, 8)
        for s in (2, 4)
    ]
    scale = _autotune(
        name, src, [x, qdata, outer], scale_layout, [qdata], [M, N], configs
    )
    return (qdata, scale)


# ---- dispatcher -----------------------------------------------------------


def _has_reduction(gm: torch.fx.GraphModule) -> bool:
    return any(
        n.op == "call_function" and n.target in _FUNCTION_REDUCTIONS for n in gm.graph.nodes
    )


@register_lowering(flex_tile_map_hop, type_promotion_kind=None)
def _flex_tile_map_lowering(x, f_subgraph, *operands):
    """Lower the HOP: only a group-reduction `f` is supported (bespoke emitter path).

    Routes on the reduction axis detected by the emitter: dim-M (split dim0, transposed outputs:
    deepseek 1x128 / mxfp8 1x32), dim-K (split the last dim, no transpose: nvfp4 1x16 with a
    per-tensor outer-scale aux), or block_2d (split BOTH dims into 32x32 blocks, no transpose:
    mxfp8 32x32). `operands` are the aux inputs (REPLICATE), passed to the template as extra input
    nodes. A pointwise `f` has no lowering here (see the module docstring): plain pointwise casts
    should be written as ordinary PyTorch and lowered by regular Inductor.
    """
    # Realize x so it has concrete strides; an unrealized Pointwise (e.g. a fused preceding op)
    # has no stride info and every template choice would get filtered out.
    (x,) = maybe_realize([x])
    gm = f_subgraph.graph_module

    if not _has_reduction(gm):
        raise NotImplementedError(
            "flex_tile_map TRITON_TEMPLATE lowering supports only group-reduction `f` (e.g. "
            "deepseek 1x128 dim-M, nvfp4 1x16 dim-K, mxfp8 32x32 block); the pointwise path was "
            "removed -- use regular Inductor for pointwise casts."
        )

    aux_names = [f"aux{i}_var" for i in range(len(operands))]
    body, group, reduce_kind = _emit(x, gm, aux_names)
    qdata_dtype, scale_dtype = _graph_output_dtypes(gm)

    if reduce_kind == "block_2d":
        return _lower_mxfp8_32x32(x, body, group, qdata_dtype, scale_dtype)
    if reduce_kind == "dim_k":
        return _lower_nvfp4(x, body, group, operands, qdata_dtype, scale_dtype)
    return _lower_reduction_dim_m(x, body, group, qdata_dtype, scale_dtype)
