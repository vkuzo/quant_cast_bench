from typing import Callable, Tuple, Union

import torch

from triton_kernels import (
    triton_fp8_blockwise_act_quant_transposed_lhs,
    triton_fp8_blockwise_weight_quant_128_128,
)


def flex_cast_quant_dense_triton(
    input: torch.Tensor,
    *,
    block_size: Union[int, Tuple[int, int]],
    dim: Union[int, Tuple[int, int]],
    qdata_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    amax_to_scale_fn_triton: Callable,
    cast_to_dtype_fn_triton: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-only counterpart to :func:`flex_cast_quant_dense`, for debugging and
    benchmarking

    Routes ``(block_size, dim, qdata_dtype, scale_dtype)`` to a hand-written
    Triton template. Unsupported combinations raise ``NotImplementedError``;
    there is no PyTorch fallback path.
    """
    assert input.ndim == 2
    assert input.is_contiguous()

    block_size_t = (block_size,) if isinstance(block_size, int) else tuple(block_size)
    dim_t = (dim,) if isinstance(dim, int) else tuple(dim)
    assert len(block_size_t) == len(dim_t)

    n_block_dims = len(block_size_t)
    normalized_dim_t = tuple(d if d >= 0 else d + input.ndim for d in dim_t)
    M, K = input.shape

    if n_block_dims == 1 and block_size_t[0] > 0:
        block_size_int = block_size_t[0]
        if normalized_dim_t == (1,):
            raise NotImplementedError(
                "no triton kernel for 1D blocks dim=-1"
            )
        elif normalized_dim_t == (0,):
            assert M % block_size_int == 0, (
                f"input.shape[-2]={M} must be divisible by block_size={block_size_int}"
            )
            if (
                block_size_int == 128
                and qdata_dtype == torch.float8_e4m3fn
                and scale_dtype == torch.float32
            ):
                qdata, scale = triton_fp8_blockwise_act_quant_transposed_lhs(
                    input, amax_to_scale_fn_triton, cast_to_dtype_fn_triton
                )
            else:
                raise NotImplementedError(
                    "triton kernel for 1D blocks dim=-2 only supports 1x128 deepseek"
                )
        else:
            raise AssertionError(f"unsupported dim={dim} for 1D blocks")

    elif n_block_dims == 2:
        if normalized_dim_t == (0, 1):
            B1, B2 = block_size_t
            assert B1 > 0 and B2 > 0
            assert M % B1 == 0 and K % B2 == 0, (
                f"input trailing dims {(M, K)} must be divisible by block_size={(B1, B2)}"
            )
            if (
                (B1, B2) == (128, 128)
                and qdata_dtype == torch.float8_e4m3fn
                and scale_dtype == torch.float32
            ):
                qdata, scale = triton_fp8_blockwise_weight_quant_128_128(
                    input, amax_to_scale_fn_triton, cast_to_dtype_fn_triton
                )
            else:
                raise NotImplementedError(
                    "triton kernel only supports 128x128 deepseek for 2D blocks"
                )
        else:
            raise AssertionError(f"unsupported dim={dim} for 2D blocks")

    else:
        raise AssertionError(f"unsupported block_size rank: {n_block_dims}")

    return qdata, scale
