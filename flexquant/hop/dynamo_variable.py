"""Dynamo variable for FlexQuantHOP.

Captures the user's PyTorch callbacks (amax_to_scale_fn, cast_to_dtype_fn) as
FX subgraphs at trace time and emits the HOP node so Inductor's lowering can
codegen them inside the Triton template at the {{ modification }} holes.

Modeled after FlexAttentionHigherOrderVariable
(torch._dynamo.variables.higher_order_ops:4522).

Registered into _hop_name_to_variable_class via monkey-patch when this module
is imported.
"""

from typing import Sequence

import torch
import torch.utils._pytree as pytree
from torch._dynamo.variables.higher_order_ops import (
    _hop_name_to_variable_class,
    discard_graph_changes,
    make_attr,
    speculate_subgraph,
    TorchHigherOrderOperatorVariable,
)
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.utils import proxy_args_kwargs


class FlexQuantHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.flex_quant"

    @staticmethod
    def normalize_to_args(args, kwargs):
        flat_kwargs = pytree.tree_flatten(kwargs)[0]
        return list(args) + flat_kwargs

    def _trace_callback(
        self,
        tx,
        x: VariableTracker,
        fn: VariableTracker,
        fn_name: str,
        n_inputs: int,
    ):
        """Trace `fn` into a subgraph using 0-d placeholders of x's dtype.

        For amax_to_scale_fn: n_inputs=1 (just amax).
        For cast_to_dtype_fn: n_inputs=2 (tile, scale).

        The deepseek callbacks are pure pointwise — they work on a 0-d scalar
        the same as on (n1, n2, 1) tile-amaxes, so this matches both eager
        and template paths bit-for-bit.
        """
        with discard_graph_changes(tx):
            placeholders = []
            for _ in range(n_inputs):
                p = x.call_method(
                    tx,
                    "new_empty",
                    [VariableTracker.build(tx, [])],
                    {},
                )
                placeholders.append(p)

        (
            (_body_output, _body_spec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            fn,
            placeholders,
            {},
            description=f"{self._HOP_NAME}: {fn_name}",
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
        )

        body_name = tx.output.install_subgraph(
            fn_name,
            torch.fx.GraphModule(tx.output.nn_modules, body_graph),
        )
        body_node = make_attr(tx, body_name)
        lifted_args = tuple(arg for arg in body_lifted_freevars)
        return body_node, lifted_args

    def _call_function(
        self,
        tx,
        args: Sequence[VariableTracker],
        kwargs,
    ) -> VariableTracker:
        from torch._dynamo.variables.builder import wrap_fx_proxy

        all_args = self.normalize_to_args(list(args), kwargs)
        (
            x,
            amax_to_scale_fn,
            cast_to_dtype_fn,
            block_size,
            dim,
            qdata_dtype,
            scale_dtype,
        ) = all_args

        amax_node, amax_lifted_args = self._trace_callback(
            tx, x, amax_to_scale_fn, "amax_to_scale_fn", n_inputs=1
        )
        cast_node, cast_lifted_args = self._trace_callback(
            tx, x, cast_to_dtype_fn, "cast_to_dtype_fn", n_inputs=2
        )

        # x is the only Tensor we need to proxy; the others (block_size, dim,
        # dtypes) are Python constants the lowering reads directly.
        proxied_args = [x]
        inp_args, _ = proxy_args_kwargs(proxied_args, {})

        with torch.fx.experimental.proxy_tensor.set_original_aten_op(self.value):
            proxy = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    args=(
                        inp_args[0],
                        amax_node,
                        cast_node,
                        block_size.as_python_constant(),
                        dim.as_python_constant(),
                        qdata_dtype.as_python_constant(),
                        scale_dtype.as_python_constant(),
                    ),
                    kwargs={},
                ),
                example_value=None,
            )
        return proxy


# Monkey-patch the in-tree registry so Dynamo dispatches our HOP through this
# variable class.
_hop_name_to_variable_class["flex_quant"] = FlexQuantHigherOrderVariable
