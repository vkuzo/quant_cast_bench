"""FlexQuantHOP: HigherOrderOperator for the flexquant 128x128 deepseek recipe.

Modeled after torch._higher_order_ops.flex_attention.FlexAttentionHOP. Forward-only
for now — autograd is left for a future patch.

Dispatch keys registered:
  CompositeExplicitAutograd: eager fallback, runs the existing reshape→amax→
                             callbacks→un-reshape body. Also used by FakeTensor.
  ProxyTorchDispatchMode:    re-traces the callbacks via reenter_make_fx and
                             registers the resulting GraphModules on the proxy
                             tracer's root, then emits the HOP node. This is
                             how Inductor receives the callbacks as FX subgraphs.
"""

from typing import Any, Callable, Tuple, Union

import torch
from torch._C import DispatchKey
from torch._higher_order_ops.utils import reenter_make_fx, register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


# (block_size, dim) pairs identifying each supported tiling. block_size and dim
# are runtime args (not constexpr) so a single HOP instance covers all
# tilings; the lowering picks the right Triton template per pair.
_TILING_128_128 = ((128, 128), (-2, -1))
_TILING_1_128_DIM_M = (128, -2)


class FlexQuantHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_quant", cacheable=True)

    def __call__(
        self,
        x: torch.Tensor,
        amax_to_scale_fn: Callable,
        cast_to_dtype_fn: Callable,
        block_size: Union[int, Tuple[int, int]],
        dim: Union[int, Tuple[int, int]],
        qdata_dtype: torch.dtype,
        scale_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(
            x,
            amax_to_scale_fn,
            cast_to_dtype_fn,
            block_size,
            dim,
            qdata_dtype,
            scale_dtype,
        )


flex_cast_quant_dense_with_hop = FlexQuantHOP()


def _tiling_key(block_size, dim):
    bs = tuple(block_size) if isinstance(block_size, (list, tuple)) else block_size
    d = tuple(dim) if isinstance(dim, (list, tuple)) else dim
    return (bs, d)


def _eager_body_128_128(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
    block_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reshape → amax → callbacks → un-reshape. Mirrors api.py:196-214."""
    M, K = x.shape
    B1, B2 = block_size
    assert M % B1 == 0 and K % B2 == 0
    n1, n2 = M // B1, K // B2

    x_b = (
        x.reshape(n1, B1, n2, B2)
        .transpose(-3, -2)
        .contiguous()
        .reshape(n1, n2, B1 * B2)
    )
    amax = x_b.abs().amax(dim=-1, keepdim=True)
    scale_bc = amax_to_scale_fn(amax)
    qdata_b = cast_to_dtype_fn(x_b, scale_bc)
    qdata = (
        qdata_b.reshape(n1, n2, B1, B2)
        .transpose(-3, -2)
        .contiguous()
        .reshape(M, K)
    )
    scale = scale_bc.squeeze(-1)
    return qdata, scale


def _eager_body_dim_m(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """1D blocks along M with transposed (K, M) output. Mirrors api.py:144-150."""
    M, K = x.shape
    assert M % block_size == 0
    n_blocks = M // block_size

    x_b = x.reshape(n_blocks, block_size, K)
    amax = x_b.abs().amax(dim=-2, keepdim=True)  # (n_blocks, 1, K)
    scale_bc = amax_to_scale_fn(amax)
    qdata_b = cast_to_dtype_fn(x_b, scale_bc)
    qdata = qdata_b.reshape(M, K).transpose(-2, -1).contiguous()
    scale = scale_bc.squeeze(-2).transpose(-2, -1).contiguous()
    return qdata, scale


def _eager_body(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
    block_size,
    dim,
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = _tiling_key(block_size, dim)
    if key == _TILING_128_128:
        return _eager_body_128_128(x, amax_to_scale_fn, cast_to_dtype_fn, tuple(block_size))
    if key == _TILING_1_128_DIM_M:
        return _eager_body_dim_m(x, amax_to_scale_fn, cast_to_dtype_fn, int(block_size))
    raise NotImplementedError(f"flex_quant: unsupported tiling (block_size={block_size!r}, dim={dim!r})")


@flex_cast_quant_dense_with_hop.py_impl(DispatchKey.CompositeExplicitAutograd)
def _flex_quant_eager(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
    block_size,
    dim,
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _eager_body(
        x, amax_to_scale_fn, cast_to_dtype_fn, block_size, dim, qdata_dtype, scale_dtype
    )


def _trace_flex_quant(
    proxy_mode: ProxyTorchDispatchMode,
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
    block_size,
    dim,
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trace the two callbacks into FX subgraphs and emit a HOP call_function node.

    Mirrors flex_attention.py:trace_flex_attention. Uses scalar 0-d placeholders
    for the callbacks since the deepseek callbacks are pure pointwise — they
    work the same on a scalar (HOP/template path) or on N-d tensors (eager).
    """
    # Eager run on the (real or fake) inputs to get the example output for the
    # FX node's `meta`.
    example_out = flex_cast_quant_dense_with_hop(
        x, amax_to_scale_fn, cast_to_dtype_fn, block_size, dim, qdata_dtype, scale_dtype
    )

    # Trace each callback into its own GraphModule. The placeholders are 0-d
    # tensors at the input dtype: amax_to_scale_fn takes a single amax,
    # cast_to_dtype_fn takes (tile, scale).
    amax_example = x.new_zeros((), dtype=x.dtype)
    tile_example = x.new_zeros((), dtype=x.dtype)
    scale_example = x.new_zeros((), dtype=scale_dtype)

    amax_graph = reenter_make_fx(amax_to_scale_fn)(amax_example)
    cast_graph = reenter_make_fx(cast_to_dtype_fn)(tile_example, scale_example)

    if not isinstance(proxy_mode.tracer, torch.fx.Tracer):
        raise AssertionError(
            f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(proxy_mode.tracer)}"
        )

    amax_qualname = proxy_mode.tracer.get_fresh_qualname("flex_quant_amax_to_scale")
    proxy_mode.tracer.root.register_module(amax_qualname, amax_graph)
    cast_qualname = proxy_mode.tracer.get_fresh_qualname("flex_quant_cast")
    proxy_mode.tracer.root.register_module(cast_qualname, cast_graph)

    node_args = (x, amax_graph, cast_graph, block_size, dim, qdata_dtype, scale_dtype)
    import torch.utils._pytree as pytree
    proxy_args = pytree.tree_map(
        proxy_mode.tracer.unwrap_proxy, node_args
    )
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        flex_cast_quant_dense_with_hop,
        proxy_args,
        {},
        name="flex_quant",
    )
    from torch.fx.experimental.proxy_tensor import track_tensor_tree

    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@flex_cast_quant_dense_with_hop.py_impl(ProxyTorchDispatchMode)
def _flex_quant_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
    block_size,
    dim,
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode is None:
        raise AssertionError("Mode should always be enabled for python fallback key")
    return _trace_flex_quant(
        mode,
        x,
        amax_to_scale_fn,
        cast_to_dtype_fn,
        block_size,
        dim,
        qdata_dtype,
        scale_dtype,
    )


@flex_cast_quant_dense_with_hop.py_impl(DispatchKey.Autograd)
def _flex_quant_autograd(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
    block_size,
    dim,
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-only autograd dispatch — runs the eager body and returns its
    result. Backward is not implemented (matches our scope)."""
    with torch._C._AutoDispatchBelowAutograd():
        return flex_cast_quant_dense_with_hop(
            x,
            amax_to_scale_fn,
            cast_to_dtype_fn,
            block_size,
            dim,
            qdata_dtype,
            scale_dtype,
        )


@flex_cast_quant_dense_with_hop.py_functionalize_impl
def _flex_quant_functionalize(
    ctx,
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
    block_size,
    dim,
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pass through functionalization. The callbacks are pure pointwise so
    nothing to functionalize beyond unwrapping the input tensor."""
    x_unwrapped = ctx.unwrap_tensors(x)
    with ctx.redispatch_to_next():
        functional_amax = ctx.functionalize(amax_to_scale_fn)
        functional_cast = ctx.functionalize(cast_to_dtype_fn)
        out = flex_cast_quant_dense_with_hop(
            x_unwrapped,
            functional_amax,
            functional_cast,
            block_size,
            dim,
            qdata_dtype,
            scale_dtype,
        )
    return ctx.wrap_tensors(out)


@register_fake(flex_cast_quant_dense_with_hop)
def _flex_quant_fake(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
    block_size,
    dim,
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FakeTensor shape inference — output shapes derive from (block_size, dim)."""
    M, K = x.shape
    key = _tiling_key(block_size, dim)
    if key == _TILING_128_128:
        B1, B2 = block_size
        n1, n2 = M // B1, K // B2
        qdata = x.new_empty((M, K), dtype=qdata_dtype)
        scale = x.new_empty((n1, n2), dtype=scale_dtype)
        return qdata, scale
    if key == _TILING_1_128_DIM_M:
        bs = int(block_size)
        n_blocks = M // bs
        qdata = x.new_empty((K, M), dtype=qdata_dtype)
        scale = x.new_empty((K, n_blocks), dtype=scale_dtype)
        return qdata, scale
    raise NotImplementedError(f"flex_quant: unsupported tiling (block_size={block_size!r}, dim={dim!r})")
