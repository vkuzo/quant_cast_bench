from enum import Enum
from typing import Callable

import torch
from torch.nn.functional import SwizzleType

# Side-effect import: registers the FlexQuant HOP, its Dynamo variable,
# and its Inductor lowering. Imported eagerly so the registrations are in
# place before the user's first torch.compile call.
from hop import flex_cast_quant_dense_with_hop
from swizzle import to_blocked_2d


class _HopMode(Enum):
    AUTO = "auto"
    HOP = "hop"
    NO_HOP = "no_hop"


def flex_cast_quant_dense(
    input: torch.Tensor,
    *,
    block_size: int | tuple[int, int] | list[int | tuple[int, int]],
    dim: int | tuple[int, int] | list[int | tuple[int, int]],
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype | list[torch.dtype],
    amax_to_scale_fn: Callable | list[Callable],
    cast_to_dtype_fn: Callable,
    scale_swizzle: SwizzleType | None = None,
    # arguments below are for debugging only
    _hop_mode: _HopMode = _HopMode.AUTO,
) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor]]:
    """Quantize a 2D tensor with user-defined per-tile scaling.

    The framework owns the layout-sensitive parts of quantization (which
    elements form a tile, how the tile reduction is computed, and how qdata
    and scale outputs are laid out in memory). The user owns two pointwise
    callbacks: how to turn a tile's amax into a scale, and how to cast a tile
    given its scale.

    Tiles are defined by ``block_size`` and ``dim``. Supported shapes
    (logical element counts; sub-byte qdata dtypes pack multiple values per
    byte, see below):

    - **1D blocked**: ``block_size: int``, ``dim: int``
      - dim=-1: output qdata is `(M, K)`, scale is `(M, K // block_size)`
      - dim=-2: output qdata is `(K, M)`, scale is `(K, M // block_size)`

    - **2D blocked** ``block_size: tuple[int, int]``, ``dim=(-2, -1)``
      - qdata is `(M, K)`, scale is `(M // B1, K // B2)`.

    Sub-byte qdata dtypes (currently ``torch.float4_e2m1fn_x2``) pack two
    values per byte along the innermost dim, so the returned qdata's
    ``shape[-1]`` is half the logical K. TODO(future): generalize this by
    exposing a ``qdata_bits_per_element`` argument and computing
    ``qdata_pack_factor = 8 // bits``. The current dtype-specific branch
    only handles ``float4_e2m1fn_x2``.

    **Two-level scaling.** When ``block_size`` is a list of length 2, the
    framework runs two-level scaling: a fine "inner" (per-block) scale plus
    a coarse "outer" (per-tensor) scale. List elements are ``[inner, outer]``
    everywhere — inputs (``block_size``, ``dim``, ``scale_dtype``,
    ``amax_to_scale_fn``), the cast callback's scale args, and the returned
    scale list. Constraints (this version):

    - The inner slot must be 1D-blocked dim=-1 (``block_size[0]: int``,
      ``dim[0] == -1``).
    - The outer slot must be ``block_size[1] == (-1, -1)`` and
      ``dim[1] == (-2, -1)`` (whole-tensor reduction → one fp32 scalar).
    - The HOP path is not supported; only the inductor path runs.

    With two-level scaling the callbacks take different signatures:

    - ``amax_to_scale_fn[0](local_amax, outer_scale) -> inner_scale``.
    - ``amax_to_scale_fn[1](amax) -> outer_scale`` (single arg).
    - ``cast_to_dtype_fn(tile, inner_scale, outer_scale) -> qdata``.

    The return becomes ``(qdata, [inner_scale, outer_scale])``.

    Args:
        input: 2D contiguous tensor.
        block_size: tile size along ``dim``. List of length 2 for two-level
            scaling (see above).
        dim: dimension(s) the tile spans. List of length 2 for two-level
            scaling.
        qdata_dtype: dtype of returned qdata. Statically known to enable
            template specialization.
        scale_dtype: dtype of returned scale, or list of two dtypes for
            two-level scaling (``[inner_dtype, outer_dtype]``).
        amax_to_scale_fn: ``(amax) -> scale`` for single-level, or a list
            of two callables for two-level (see above). Must be a pure
            pointwise op.
        cast_to_dtype_fn: ``(tile, scale) -> qdata`` for single-level, or
            ``(tile, inner_scale, outer_scale) -> qdata`` for two-level.
            Must be a pure pointwise op.
        scale_swizzle: optional scale layout transform applied to the returned
            scale. ``None`` (default) returns the natural row-major scale.
            ``SwizzleType.SWIZZLE_32_4_4`` returns the NVIDIA blocked layout
            ``_scaled_mm`` consumes, as a 2D tensor of shape
            ``(32 * ceil(M / 128), 16 * ceil(n_blocks / 4))``. Only supported
            on the single-level 1D ``dim=-1`` path.

    Returns:
        ``(qdata, scale)`` for single-level. ``(qdata, [inner_scale,
        outer_scale])`` for two-level.
    """
    assert input.ndim == 2
    assert input.is_contiguous()

    # Detect two-level vs single-level scaling. Two-level is signalled by a
    # list-typed block_size of length 2: [inner, outer].
    _two_level = isinstance(block_size, list)
    if _two_level:
        assert isinstance(block_size, list) and len(block_size) == 2
        assert isinstance(dim, list) and len(dim) == 2
        assert isinstance(amax_to_scale_fn, list) and len(amax_to_scale_fn) == 2
        assert isinstance(scale_dtype, list) and len(scale_dtype) == 2

        inner_block_size, outer_block_size = block_size
        inner_dim, outer_dim = dim
        inner_amax_to_scale_fn, outer_amax_to_scale_fn = amax_to_scale_fn
        inner_scale_dtype, outer_scale_dtype = scale_dtype

        # Inner slot: 1D-blocked dim=-1 only.
        assert isinstance(inner_block_size, int), \
            "two-level inner block_size must be int (1D-blocked)"
        assert inner_dim == -1, "two-level inner dim must be -1"

        # Outer slot: whole-tensor reduction only.
        assert tuple(outer_block_size) == (-1, -1), \
            "two-level outer block_size must be (-1, -1)"
        assert tuple(outer_dim) == (-2, -1), \
            "two-level outer dim must be (-2, -1)"

        # Pre-compute the outer scale.
        outer_amax = input.abs().to(torch.float32).amax()  # scalar fp32
        outer_scale = outer_amax_to_scale_fn(outer_amax)

        # Rebind block_size/dim/scale_dtype to inner-only values; the rest of
        # the function consumes these uniformly. The callbacks
        # (amax_to_scale_fn, cast_to_dtype_fn) are NOT rebound; the call sites
        # branch on _two_level explicitly.
        block_size = inner_block_size
        dim = inner_dim
        scale_dtype = inner_scale_dtype

        # HOP path doesn't know about two-level scaling.
        assert _hop_mode in (_HopMode.AUTO, _HopMode.NO_HOP), \
            "HOP path doesn't support two-level scaling"

    if scale_swizzle is not None:
        assert scale_swizzle == SwizzleType.SWIZZLE_32_4_4, \
            f"unsupported scale_swizzle: {scale_swizzle}"
        assert not _two_level, \
            "scale_swizzle is not supported with two-level scaling"

    block_size_t = (block_size,) if isinstance(block_size, int) else tuple(block_size)
    dim_t = (dim,) if isinstance(dim, int) else tuple(dim)
    assert len(block_size_t) == len(dim_t)

    n_block_dims = len(block_size_t)
    # normalized_dim_t can be either (0,), (1,) or (0, 1)
    normalized_dim_t = tuple(d if d >= 0 else d + input.ndim for d in dim_t)
    M, K = input.shape

    # Sub-byte qdata dtypes pack multiple values per byte and shrink the
    # logical element count along the innermost dim of cast_to_dtype_fn's
    # output.
    if qdata_dtype == torch.float4_e2m1fn_x2:
        qdata_pack_factor = 2  # 2 fp4 elements per byte
    else:
        qdata_pack_factor = 1

    # TODO(future):
    # * scale swizzling
    # * padding and other edge cases

    if n_block_dims == 1 and block_size_t[0] > 0:
        # 1D blocked scaling

        block_size_int = block_size_t[0]

        if normalized_dim_t == (1,):
            # dim=-1: reduce across K; output qdata in (M, K), scale in (M, n_blocks)

            # compile is known fast, no hop defined
            assert _hop_mode in (_HopMode.AUTO, _HopMode.NO_HOP), "unsupported"

            assert K % block_size_int == 0, (
                f"input.shape[-1]={K} must be divisible by block_size={block_size_int}"
            )
            n_blocks = K // block_size_int
            x_b = input.reshape(M, n_blocks, block_size_int)
            amax = x_b.abs().amax(dim=-1, keepdim=True)  # (M, n_blocks, 1)
            if _two_level:
                scale_bc = inner_amax_to_scale_fn(amax, outer_scale)
                qdata_b = cast_to_dtype_fn(x_b, scale_bc, outer_scale)
            else:
                scale_bc = amax_to_scale_fn(amax)
                qdata_b = cast_to_dtype_fn(x_b, scale_bc)
            qdata = qdata_b.reshape(M, K // qdata_pack_factor)
            scale = scale_bc.squeeze(-1)  # (M, n_blocks)
            if scale_swizzle is not None:
                scale = to_blocked_2d(scale)

        else:
            # dim=-2: reduce across M; output qdata/scale row-major in (K, M) layout

            assert scale_swizzle is None, \
                "scale_swizzle is only supported on the dim=-1 path"
            assert normalized_dim_t == (0,), "unsupported"
            assert M % block_size_int == 0, (
                f"input.shape[-2]={M} must be divisible by block_size={block_size_int}"
            )

            # Packing happens along the inner dim (K), and this path transposes
            # to (K, M) layout, which would have to swap bytes whose two fp4
            # halves are K-adjacent. Not supported yet.
            # TODO(future PR): add support for this in the reference path.
            assert qdata_pack_factor == 1, (
                "sub-byte qdata dtypes are not supported on the dim=-2 path"
            )

            # hop known fast, use it unless overridden
            _use_hop_path = _hop_mode in (_HopMode.AUTO, _HopMode.HOP)

            if _use_hop_path:
                assert (
                    block_size_int == 128 
                    and qdata_dtype == torch.float8_e4m3fn 
                    and scale_dtype == torch.float32
                    and qdata_pack_factor == 1
                ), "unsupported"
                qdata, scale = flex_cast_quant_dense_with_hop(
                    input,
                    amax_to_scale_fn,
                    cast_to_dtype_fn,
                    block_size_int,
                    -2,
                    qdata_dtype,
                    scale_dtype,
                )

            else:
                n_blocks = M // block_size_int
                x_b = input.reshape(n_blocks, block_size_int, K)
                amax = x_b.abs().amax(dim=-2, keepdim=True)  # (n_blocks, 1, K)
                scale_bc = amax_to_scale_fn(amax)
                qdata_b = cast_to_dtype_fn(x_b, scale_bc)
                qdata = qdata_b.reshape(M, K).transpose(-2, -1).contiguous()
                scale = scale_bc.squeeze(-2).transpose(-2, -1).contiguous()

    elif n_block_dims == 2:
        # 2D blocked scaling

        if normalized_dim_t == (0, 1):
            assert scale_swizzle is None, \
                "scale_swizzle is only supported on the dim=-1 path"
            B1, B2 = block_size_t
            assert B1 > 0 and B2 > 0
            assert M % B1 == 0 and K % B2 == 0, (
                f"input trailing dims {(M, K)} must be divisible by block_size={(B1, B2)}"
            )

            # hop known fast, use it unless overridden
            _use_hop_path = _hop_mode in (_HopMode.AUTO, _HopMode.HOP)

            if _use_hop_path:
                assert (
                    (B1, B2) == (128, 128)
                    and qdata_dtype == torch.float8_e4m3fn
                    and scale_dtype == torch.float32
                    and qdata_pack_factor == 1
                ), "unsupported"
                qdata, scale = flex_cast_quant_dense_with_hop(
                    input,
                    amax_to_scale_fn,
                    cast_to_dtype_fn,
                    (B1, B2),
                    (-2, -1),
                    qdata_dtype,
                    scale_dtype,
                )

            else:
                n1, n2 = M // B1, K // B2

                # (M, K) -> (n1, B1, n2, B2) -> (n1, n2, B1, B2) -> (n1, n2, B1*B2)
                x_b = (
                    input.reshape(n1, B1, n2, B2)
                    .transpose(-3, -2)
                    .contiguous()
                    .reshape(n1, n2, B1 * B2)
                )
                amax = x_b.abs().amax(dim=-1, keepdim=True)
                scale_bc = amax_to_scale_fn(amax)
                qdata_b = cast_to_dtype_fn(x_b, scale_bc)
                qdata = (
                    qdata_b.reshape(n1, n2, B1, B2 // qdata_pack_factor)
                    .transpose(-3, -2)
                    .contiguous()
                    .reshape(M, K // qdata_pack_factor)
                )
                scale = scale_bc.squeeze(-1)

        else:
            raise AssertionError(f"unsupported dim={dim} for 2D blocks")

    else:
        raise AssertionError(f"unsupported block_size rank: {n_block_dims}")

    if _two_level:
        return qdata, [scale, outer_scale]
    return qdata, scale
