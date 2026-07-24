"""The reference-path HOP: a BaseHOP that is fusible into flex_gemm.

This is the SEPARATE, fusion-enabled op for the REFERENCE backend, distinct from the
hand-rolled Triton-template HOP in ``hop/`` (which is not fused). It is a near-verbatim port of
the working reference at ``/home/dev/pytorch_scripts/flex_tile_map_v2/api.py``.

Why a ``BaseHOP`` (rather than extending the hand-rolled ``hop/hop.py``): BaseHOP supplies the
CompositeExplicitAutograd (eager ``f(input, *operands)``), fake, functionalize, autograd, and
ProxyTorchDispatchMode impls for free. Crucially it also gives us Dynamo freevar lifting via the
in-tree ``BaseHOPVariable`` -- so an epilogue ``f`` that captures a tensor (e.g. the backward VJP
``lambda go: go * c.cos()`` capturing the saved activation ``c``) traces correctly, with the
captured tensor appended to the operands as a lifted freevar. That is exactly what makes the
fusion fire in the BACKWARD graph, and it is the piece we would otherwise have to hand-roll.

The op is SUBGRAPH-FIRST (``flex_tile_map_ref_hop(subgraph, input, *operands)``), matching
flex_gemm/flex_attention, so the fusion pass (see ``flex_gemm_to_tile_map_fusion.py``) needs no
arg-index adaptation.
"""

import torch
from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars

__all__ = ["flex_tile_map_ref_hop", "flex_tile_map_ref"]


class FlexTileMapRef(BaseHOP):
    """A BaseHOP so Dynamo traces it under fullgraph=True (via BaseHOPVariable).

    Forward-only in practice (the user wraps it in a ``torch.autograd.Function``), so we keep
    BaseHOP's defaults; BaseHOP still supplies a correct autograd impl, which de-risks the
    backward fusion.
    """

    def __init__(self) -> None:
        super().__init__("flex_tile_map_ref")


flex_tile_map_ref_hop = FlexTileMapRef()


def flex_tile_map_ref(input, f, operands=()):
    """Apply the epilogue ``f`` to ``input`` as a standalone, fusible op.

    ``f(input, *operands) -> Tensor | tuple[Tensor, ...]``. ``operands`` are extra positional
    tensors ``f`` needs (the REFERENCE backend's ``aux_inputs``); any tensors ``f`` captures by
    closure are lifted by Dynamo and appended to the operands automatically.
    """
    if torch.compiler.is_dynamo_compiling():
        # Dynamo speculates ``f`` into a subgraph itself; pass it through raw.
        return flex_tile_map_ref_hop(f, input, *operands)
    # Eager: BaseHOP.__call__ requires a wrapped callable (no free vars).
    return flex_tile_map_ref_hop(FunctionWithNoFreeVars(f), input, *operands)
