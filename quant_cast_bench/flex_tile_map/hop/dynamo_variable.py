"""Dynamo variable for FlexTileMapHOP.

Captures the user's callback `f` as an FX subgraph at trace time and emits the HOP node so
Inductor's lowering can codegen it inside the Triton template at the {{ modification }} hole.

Modeled after quant_cast_bench/flexquant/hop/dynamo_variable.py (v1), stripped to a single
callback `f`. Registered into _hop_name_to_variable_class via monkey-patch on import.
"""

from typing import Sequence

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import proxy_args_kwargs
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.higher_order_ops import (
    _hop_name_to_variable_class,
    discard_graph_changes,
    make_attr,
    speculate_subgraph,
    TorchHigherOrderOperatorVariable,
)


class FlexTileMapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.flex_tile_map"

    @staticmethod
    def normalize_to_args(args, kwargs):
        flat_kwargs = pytree.tree_flatten(kwargs)[0]
        return list(args) + flat_kwargs

    def _trace_callback(self, tx, x: VariableTracker, fn: VariableTracker):
        """Trace `f` into a subgraph using a placeholder the SHAPE of the full input.

        Tracing at the real input shape means a group-reduction `f` (deepseek's
        `x.reshape(*lead, last//128, 128)`) produces a graph that is also valid for whole-tensor
        fake-shape inference (the HOP is invoked whole-tensor), and a pointwise `f` traces the
        same graph it would on any shape. Concrete dims don't leak into codegen: the reduction
        emitter rewrites them to the BLOCK_M/BLOCK_N block symbols.
        """
        with discard_graph_changes(tx):
            size_vt = x.call_method(tx, "size", [], {})
            placeholder = x.call_method(tx, "new_empty", [size_vt], {})

        (
            (_body_output, _body_spec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            fn,
            [placeholder],
            {},
            description=f"{self._HOP_NAME}: f",
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
        )

        body_name = tx.output.install_subgraph(
            "flex_tile_map_f",
            torch.fx.GraphModule(tx.output.nn_modules, body_graph),
        )
        body_node = make_attr(tx, body_name)
        lifted_args = tuple(arg for arg in body_lifted_freevars)
        return body_node, lifted_args

    def _call_function(self, tx, args: Sequence[VariableTracker], kwargs) -> VariableTracker:
        from torch._dynamo.variables.builder import wrap_fx_proxy

        (x, f) = self.normalize_to_args(list(args), kwargs)

        f_node, _f_lifted_args = self._trace_callback(tx, x, f)

        # x is the only Tensor we need to proxy; `f` becomes the subgraph attr node.
        inp_args, _ = proxy_args_kwargs([x], {})

        with torch.fx.experimental.proxy_tensor.set_original_aten_op(self.value):
            proxy = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    args=(inp_args[0], f_node),
                    kwargs={},
                ),
                example_value=None,
            )
        return proxy


# Monkey-patch the in-tree registry so Dynamo dispatches our HOP through this variable class.
_hop_name_to_variable_class["flex_tile_map"] = FlexTileMapHigherOrderVariable
