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


# ---- reduction path (FxTritonEmitter -> deepseek template) -----------------


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


def _lower_reduction(x, gm: torch.fx.GraphModule):
    """Walk `gm` with FxTritonEmitter and splice the body into the deepseek template."""
    device = x.get_device()
    M = x.get_size()[0]
    N = x.get_size()[1]

    # output dtypes come straight from the traced graph's two outputs (qdata, scale).
    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    qdata_node, scale_node = out_node.args[0]
    qdata_dtype = qdata_node.meta["val"].dtype
    scale_dtype = scale_node.meta["val"].dtype

    # to_dtype resolves the triton dtype via V.graph.get_current_device_or_throw(), which is only
    # populated during device-specific codegen -- set it explicitly for the emit.
    from torch._inductor.virtualized import V

    with V.graph.set_current_device(device):
        body, group = FxTritonEmitter(gm, output_names=["qdata_var", "scale_var"]).emit()
    if group is None:
        raise NotImplementedError("reduction path requires a group reshape in `f`")

    # Only the dim-M variant is supported: reduce down rows in 128-groups and TRANSPOSE both
    # outputs. BLOCK_M must be a multiple of the group; outputs are the transpose of the input:
    # qdata (N, M), scale (N, M//group). The kernel tiles the input (M, N) but stores transposed
    # tiles. (The emitter rejects the dim-K "split last dim" shape up front, in _lower_view.)
    name, src = _splice_body(body, "template_deepseek_dim_m.py.jinja")
    qdata_layout = FixedLayout(device, qdata_dtype, [N, M], stride=[M, 1])
    scale = empty_strided([N, M // group], None, dtype=scale_dtype, device=device)
    configs = [
        {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": w, "num_stages": s}
        for bm in (group, group * 2)
        for bn in (32, 64, 128)
        for w in (4, 8)
        for s in (2, 4)
    ]

    template = TritonTemplate(name=name, grid=_grid_reduce, source=src)

    choices: list[Any] = []
    for cfg in configs:
        template.maybe_append_choice(
            choices=choices,
            input_nodes=[x, scale],
            layout=qdata_layout,
            mutated_inputs=[scale],
            call_sizes=[M, N],
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    qdata, _ = autotune_select_algorithm(name, choices, [x, scale], qdata_layout)
    return (qdata, scale)


# ---- dispatcher -----------------------------------------------------------


def _has_reduction(gm: torch.fx.GraphModule) -> bool:
    return any(
        n.op == "call_function" and n.target in _FUNCTION_REDUCTIONS for n in gm.graph.nodes
    )


@register_lowering(flex_tile_map_hop, type_promotion_kind=None)
def _flex_tile_map_lowering(x, f_subgraph):
    """Lower the HOP: only a group-reduction `f` is supported (bespoke emitter path).

    A pointwise `f` no longer has a lowering here (see the module docstring): plain pointwise casts
    should be written as ordinary PyTorch and lowered by regular Inductor, not routed through this
    HOP+template.
    """
    # Realize x so it has concrete strides; an unrealized Pointwise (e.g. a fused preceding op)
    # has no stride info and every template choice would get filtered out.
    (x,) = maybe_realize([x])

    if not _has_reduction(f_subgraph.graph_module):
        raise NotImplementedError(
            "flex_tile_map TRITON_TEMPLATE lowering supports only group-reduction `f` (e.g. "
            "deepseek 1x128 dim-M); the pointwise path was removed -- use regular Inductor for "
            "pointwise casts."
        )
    return _lower_reduction(x, f_subgraph.graph_module)
