"""FlexTileMapHOP: HigherOrderOperator for the flex_tile_map TRITON_TEMPLATE backend.

Modeled after quant_cast_bench/flexquant/hop/hop.py (v1) and
torch._higher_order_ops.flex_attention.FlexAttentionHOP, but stripped down to a single
user callback `f` and a single output -- no amax reduction, no scale, no mutated inputs.
Forward-only (no autograd), naive elementwise tiling.

`f` is the user's tile function `f(tile) -> (out,)` (a 1-tuple). At the API boundary it is a
python callable; inside a traced graph the node arg is the FX GraphModule produced by tracing
`f`. The output dtype is derived from `f` (via the traced subgraph), never passed as a constant.

Dispatch keys registered:
  CompositeExplicitAutograd: eager fallback -- runs `f(x)` directly. This is the path taken
                             when the caller did NOT torch.compile flex_tile_map.
  ProxyTorchDispatchMode:    re-traces `f` via reenter_make_fx and registers the resulting
                             GraphModule on the proxy tracer's root, then emits the HOP node
                             (the direct make_fx path; torch.compile uses the Dynamo variable).
"""

from typing import Callable, Tuple

import torch
from torch._C import DispatchKey
from torch._higher_order_ops.utils import reenter_make_fx, register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


class FlexTileMapHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_tile_map", cacheable=True)

    def __call__(self, x: torch.Tensor, f: Callable) -> Tuple[torch.Tensor, ...]:
        return super().__call__(x, f)


flex_tile_map_hop = FlexTileMapHOP()


@flex_tile_map_hop.py_impl(DispatchKey.CompositeExplicitAutograd)
def _flex_tile_map_eager(x: torch.Tensor, f: Callable) -> Tuple[torch.Tensor, ...]:
    """Eager body: run `f` on the whole tensor. `f` returns a 1-tuple `(out,)`."""
    return tuple(f(x))


def _trace_flex_tile_map(
    proxy_mode: ProxyTorchDispatchMode,
    x: torch.Tensor,
    f: Callable,
) -> Tuple[torch.Tensor, ...]:
    """Trace `f` into an FX subgraph and emit a HOP call_function node.

    Mirrors v1's _trace_flex_quant. Traces `f` on a tile the SHAPE of the full input, so a
    group-reduction `f` (deepseek's reshape into 128-groups) produces a graph that is also valid
    for full-tensor fake-shape inference (the HOP is invoked whole-tensor). Concrete dims don't
    leak into codegen: the reduction emitter rewrites them to the BLOCK_M/BLOCK_N block symbols,
    and the 128 group width is a literal constant in `f` regardless of the trace shape.
    """
    example_out = flex_tile_map_hop(x, f)

    tile_example = x.new_zeros(x.shape, dtype=x.dtype)
    f_graph = reenter_make_fx(f)(tile_example)

    if not isinstance(proxy_mode.tracer, torch.fx.Tracer):
        raise AssertionError(
            f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(proxy_mode.tracer)}"
        )

    qualname = proxy_mode.tracer.get_fresh_qualname("flex_tile_map_f")
    proxy_mode.tracer.root.register_module(qualname, f_graph)

    import torch.utils._pytree as pytree

    node_args = (x, f_graph)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        flex_tile_map_hop,
        proxy_args,
        {},
        name="flex_tile_map",
    )
    from torch.fx.experimental.proxy_tensor import track_tensor_tree

    return track_tensor_tree(example_out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@flex_tile_map_hop.py_impl(ProxyTorchDispatchMode)
def _flex_tile_map_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    x: torch.Tensor,
    f: Callable,
) -> Tuple[torch.Tensor, ...]:
    if mode is None:
        raise AssertionError("Mode should always be enabled for python fallback key")
    return _trace_flex_tile_map(mode, x, f)


@flex_tile_map_hop.py_impl(DispatchKey.Autograd)
def _flex_tile_map_autograd(x: torch.Tensor, f: Callable) -> Tuple[torch.Tensor, ...]:
    """Forward-only autograd dispatch -- runs the eager body. Backward is not implemented."""
    with torch._C._AutoDispatchBelowAutograd():
        return flex_tile_map_hop(x, f)


@flex_tile_map_hop.py_functionalize_impl
def _flex_tile_map_functionalize(ctx, x: torch.Tensor, f: Callable) -> Tuple[torch.Tensor, ...]:
    """Pass through functionalization: `f` is pure pointwise, so just unwrap `x`."""
    x_unwrapped = ctx.unwrap_tensors(x)
    with ctx.redispatch_to_next():
        functional_f = ctx.functionalize(f)
        out = flex_tile_map_hop(x_unwrapped, functional_f)
    return ctx.wrap_tensors(out)


@register_fake(flex_tile_map_hop)
def _flex_tile_map_fake(x: torch.Tensor, f: Callable) -> Tuple[torch.Tensor, ...]:
    """FakeTensor shape/dtype inference: run the traced subgraph on the fake input.

    `f` here is the traced GraphModule (or the raw callable at trace time), so running it on the
    full fake `x` yields correctly-shaped/typed fake outputs -- the output dtype comes from `f`,
    not a constant. Shapes are specialized static (the API marks `input` static before the HOP
    call, see api.py), so the subgraph never grows free-symbol SymInt placeholders.
    """
    return tuple(f(x))
