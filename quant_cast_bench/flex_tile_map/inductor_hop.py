"""The INDUCTOR-backend HOP: a BaseHOP that is fusible into flex_gemm.

This is the SEPARATE, fusion-enabled op for the INDUCTOR backend, distinct from the
hand-rolled Triton-template HOP in ``hop/`` (which is not fused). It is a near-verbatim port of
the working reference at ``/home/dev/pytorch_scripts/flex_tile_map_v2/api.py``.

Why a ``BaseHOP`` (rather than extending the hand-rolled ``hop/hop.py``): BaseHOP supplies the
CompositeExplicitAutograd (eager ``f(input, *operands)``), fake, functionalize, autograd, and
ProxyTorchDispatchMode impls for free. Crucially it also gives us Dynamo freevar lifting via the
in-tree ``BaseHOPVariable`` -- so an epilogue ``f`` that captures a tensor (e.g. the backward VJP
``lambda go: go * c.cos()`` capturing the saved activation ``c``) traces correctly, with the
captured tensor appended to the operands as a lifted freevar. That is exactly what makes the
fusion fire in the BACKWARD graph, and it is the piece we would otherwise have to hand-roll.

The op is SUBGRAPH-FIRST (``flex_tile_map_inductor_hop(subgraph, input, *operands)``), matching
flex_gemm/flex_attention, so the fusion pass (see ``flex_gemm_to_tile_map_fusion.py``) needs no
arg-index adaptation.
"""

import torch
from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars

__all__ = ["flex_tile_map_inductor_hop", "flex_tile_map_inductor"]


class FlexTileMapInductor(BaseHOP):
    """A BaseHOP so Dynamo traces it under fullgraph=True (via BaseHOPVariable).

    Forward-only in practice (the user wraps it in a ``torch.autograd.Function``), so we keep
    BaseHOP's defaults; BaseHOP still supplies a correct autograd impl, which de-risks the
    backward fusion.
    """

    def __init__(self) -> None:
        super().__init__("flex_tile_map_inductor")


flex_tile_map_inductor_hop = FlexTileMapInductor()


def flex_tile_map_inductor(input, f, operands=()):
    """Apply the epilogue ``f`` to ``input`` as a standalone, fusible op.

    ``f(input, *operands) -> Tensor | tuple[Tensor, ...]``. ``operands`` are extra positional
    tensors ``f`` needs (the INDUCTOR backend's ``aux_inputs``); any tensors ``f`` captures by
    closure are lifted by Dynamo and appended to the operands automatically.
    """
    if torch.compiler.is_dynamo_compiling():
        # Dynamo speculates ``f`` into a subgraph itself; pass it through raw. It always traces to
        # a collection-returning subgraph, so BaseHOP autograd works on the compile path already.
        return flex_tile_map_inductor_hop(f, input, *operands)
    # Eager: BaseHOP.__call__ requires a wrapped callable (no free vars). Additionally, BaseHOP's
    # backward (``create_fw_bw_graph``) counts the subgraph's outputs via ``len(subgraph(...))`` --
    # a bare-tensor return makes that ``len`` the row count, so the joint fwd/bwd graph gets the
    # wrong number of outputs (``outs_to_grad length != tangents length``) and backward breaks. Wrap
    # a single-tensor epilogue so its subgraph returns a 1-tuple, then unwrap on the way out to keep
    # the eager return type identical to a plain ``f`` (bare tensor for single output, tuple for
    # multi). A tuple-returning ``f`` is passed through unchanged.
    state = {}

    def as_tuple(*args, **kwargs):
        out = f(*args, **kwargs)
        if isinstance(out, (tuple, list)):
            state["wrapped"] = False
            return tuple(out)
        state["wrapped"] = True
        return (out,)

    result = flex_tile_map_inductor_hop(FunctionWithNoFreeVars(as_tuple), input, *operands)
    if state.get("wrapped"):
        return result[0]
    return result
