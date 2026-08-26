"""Emulated (dequantize-and-matmul) mxfp8 MoE grouped GEMMs (forward + backward).

Companion to `moe_main.py` (the real `torch._scaled_grouped_mm` path). This module holds the
plain-PyTorch emulated recipe: quantize each operand to e4m3 data + e8m0 (power-of-two, block size
32) scales along the appropriate contraction axis, dequantize back to bf16, then call plain
`torch._grouped_mm`. It mirrors torchao's `_emulated_mxfp8_scaled_grouped_mm_2d_{3d,2d}` and
`_compute_{fwd,dgrad,wgrad}_emulated`, and needs no SM100 device (unlike the real path).

The three MoE training grouped GEMMs, with expert groups selected along the token dim by an int32
`offs` tensor (group `i` spans rows `[offs[i-1], offs[i])`):

    forward : out        = grouped_mm(act (M,K),        weight_t (E,K,N))          -> (M, N)
    dgrad   : grad_input = grouped_mm(grad_output (M,N), weight (E,N,K))            -> (M, K)
    wgrad   : grad_weight = per-group  grad_output(M,N)^T @ input_act(M,K)          -> (E, K, N)

where `weight_t = weight.transpose(-2, -1)` (`weight` is the natural `(E, N, K)` expert stack).
"""

import torch

from quant_cast_bench.quantize_tensor_api.moe_utils import BLOCK_SIZE, quantize_2d_act
from quant_cast_bench.quant_cast_gold.recipes import _compute_error, mxfp8_f


def quantize_3d_weight(mat2: torch.Tensor):
    """Quantize the 3d weight operand `mat2 = (E, K, N)` to mxfp8 with 1x32 blocks along K.

    Mirrors torchao's `_mxfp8_quantize_reference_3d` with `scale_block_dim2 == 1`: blocks run along
    the contraction dim K (quantize the K-last transpose, then transpose back).

    Returns:
        w_fp8:   `(E, K, N)` float8_e4m3fn.
        w_scale: `(E, K // 32, N)` e8m0 (float8_e8m0fnu), naive (unswizzled) layout -- one scale per
                 32 values along K, as `_emulated_mxfp8_..._2d_3d` expects.
    """
    assert mat2.ndim == 3, "mat2 must be 3D (E, K, N)"
    w_t = mat2.transpose(-2, -1).contiguous()  # (E, N, K), so mxfp8_f blocks along K
    q_t, scale = mxfp8_f(w_t)  # q_t (E, N, K), scale (E, N, K//32)
    w_fp8 = q_t.transpose(-2, -1).contiguous()  # (E, K, N)
    w_scale = scale.transpose(-2, -1).contiguous()  # (E, K//32, N)
    return w_fp8, w_scale


def quantize_3d_along_dim1(x: torch.Tensor):
    """Quantize a 3d tensor `(E, D0, D1)` to mxfp8 with 1x32 blocks along the middle dim D0.

    Mirrors torchao's `_quantize_3d_along_dim1_native`: transpose so D0 is last, block 1x32, then
    transpose back. Used for the dgrad weight, which must be blocked along its N dim (the dgrad
    contraction axis).

    Returns:
        qdata: `(E, D0, D1)` float8_e4m3fn.
        scale: `(E, D0 // 32, D1)` e8m0 (float8_e8m0fnu).
    """
    assert x.ndim == 3, "x must be 3D (E, D0, D1)"
    x_t = x.transpose(-2, -1).contiguous()  # (E, D1, D0), so mxfp8_f blocks along D0
    q_t, scale_t = mxfp8_f(x_t)  # q_t (E, D1, D0), scale_t (E, D1, D0//32)
    qdata = q_t.transpose(-2, -1).contiguous()  # (E, D0, D1)
    scale = scale_t.transpose(-2, -1).contiguous()  # (E, D0//32, D1)
    return qdata, scale


def emulated_mxfp8_grouped_mm_2d_3d(
    A_data: torch.Tensor,  # (M, C) e4m3, blocked 1x32 along C (the contraction dim)
    A_scale: torch.Tensor,  # (M, C//block_size) e8m0
    B_data: torch.Tensor,  # (E, C, P) e4m3, blocked 1x32 along C
    B_scale: torch.Tensor,  # (E, C//block_size, P) e8m0
    offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Emulated mxfp8 2d-3d grouped GEMM: dequantize both operands to bf16, then `torch._grouped_mm`.

    Port of torchao's `_emulated_mxfp8_scaled_grouped_mm_2d_3d`. Stands in for the SM100-only
    `torch._scaled_grouped_mm(A_data, B_data, A_scale_blocked, B_scale_blocked, offs=...)`.

    Generic over which GEMM uses it -- `C` is the shared contraction dim:
      forward: A=act (M,K),        B=weight_t (E,K,N)  -> out        (M,N),  C=K
      dgrad:   A=grad_output (M,N), B=weight (E,N,K)    -> grad_input (M,K),  C=N
    """
    assert A_data.ndim == 2, "A must be 2D"
    assert B_data.ndim == 3, "B must be 3D"
    M, C = A_data.shape
    E, C_b, P = B_data.shape
    assert C_b == C, "A and B must share the contraction dim"
    assert A_scale.shape == (M, C // block_size), f"unexpected A_scale shape {A_scale.shape}"
    assert B_scale.shape == (E, C // block_size, P), f"unexpected B_scale shape {B_scale.shape}"

    # Dequantize A: (M, C//bs, bs) * (M, C//bs, 1) -> (M, C)
    A = (
        A_data.reshape(M, C // block_size, block_size).to(torch.bfloat16)
        * A_scale.unsqueeze(-1).to(torch.bfloat16)
    ).reshape(M, C)

    # Dequantize B: (E, C//bs, bs, P) * (E, C//bs, 1, P) -> (E, C, P)
    B = (
        B_data.reshape(E, C // block_size, block_size, P).to(torch.bfloat16)
        * B_scale.unsqueeze(-2).to(torch.bfloat16)
    ).reshape(E, C, P)

    return torch._grouped_mm(A, B, offs=offs, out_dtype=out_dtype)


def emulated_mxfp8_grouped_mm_2d_2d(
    A_data: torch.Tensor,  # (P, S) e4m3, blocked 1x32 along S (the contraction dim)
    A_scale: torch.Tensor,  # (P, S//block_size) e8m0
    B_data: torch.Tensor,  # (Q, S) e4m3, blocked 1x32 along S
    B_scale: torch.Tensor,  # (Q, S//block_size) e8m0
    offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Emulated mxfp8 2d-2d grouped GEMM: dequantize both operands, then `torch._grouped_mm`.

    Port of torchao's `_emulated_mxfp8_scaled_grouped_mm_2d_2d`. Both operands share the
    group-partitioned contraction dim S; the result is `(E, P, Q)` (one `P x Q` block per group).
    Used by wgrad with A=grad_output^T (N, M) and B=input_act^T (K, M), S=M -> grad_weight (E,N,K).
    """
    assert A_data.ndim == 2 and B_data.ndim == 2, "A and B must be 2D"
    P, S = A_data.shape
    Q, S_b = B_data.shape
    assert S_b == S, "A and B must share the contraction dim S"
    assert A_scale.shape == (P, S // block_size), f"unexpected A_scale shape {A_scale.shape}"
    assert B_scale.shape == (Q, S // block_size), f"unexpected B_scale shape {B_scale.shape}"

    A = (
        A_data.reshape(P, S // block_size, block_size).to(torch.bfloat16)
        * A_scale.unsqueeze(-1).to(torch.bfloat16)
    ).reshape(P, S)
    B = (
        B_data.reshape(Q, S // block_size, block_size).to(torch.bfloat16)
        * B_scale.unsqueeze(-1).to(torch.bfloat16)
    ).reshape(Q, S)

    # A (P, S) @ B^T (S, Q) with S group-partitioned -> (E, P, Q)
    return torch._grouped_mm(A, B.transpose(-2, -1), offs=offs, out_dtype=out_dtype)


# ---------------------------------------------------------------------------
# High-level emulated GEMMs, one per training grouped-GEMM. These mirror torchao's
# _compute_{fwd,dgrad,wgrad}_emulated: cast the high-precision operands, then run the emulated
# grouped GEMM above.
# ---------------------------------------------------------------------------
def mxfp8_fwd_emulated(
    act: torch.Tensor,  # (M, K)
    weight_t: torch.Tensor,  # (E, K, N)
    offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Forward: `out = grouped_mm(act, weight_t)` in emulated mxfp8. Returns `(M, N)`."""
    act_fp8, act_scale = quantize_2d_act(act)  # blocks along K
    w_fp8, w_scale = quantize_3d_weight(weight_t)  # (E,K,N), scale (E,K//32,N); blocks along K
    return emulated_mxfp8_grouped_mm_2d_3d(
        act_fp8, act_scale, w_fp8, w_scale, offs=offs, out_dtype=out_dtype
    )


def mxfp8_dgrad_emulated(
    grad_output: torch.Tensor,  # (M, N)
    weight_t: torch.Tensor,  # (E, K, N)
    offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """dgrad: `grad_input = grouped_mm(grad_output, weight)` in emulated mxfp8. Returns `(M, K)`.

    The weight uses its natural orientation `weight = weight_t.transpose(-2,-1)` = `(E, N, K)` and
    is quantized along N (the dgrad contraction dim)."""
    go_fp8, go_scale = quantize_2d_act(grad_output)  # (M,N), scale (M,N//32); blocks along N
    weight = weight_t.transpose(-2, -1)  # (E, N, K)
    w_fp8, w_scale = quantize_3d_along_dim1(weight)  # (E,N,K), scale (E,N//32,K); blocks along N
    return emulated_mxfp8_grouped_mm_2d_3d(
        go_fp8, go_scale, w_fp8, w_scale, offs=offs, out_dtype=out_dtype
    )


def mxfp8_wgrad_emulated(
    grad_output: torch.Tensor,  # (M, N)
    input_act: torch.Tensor,  # (M, K)
    offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """wgrad: per-group `grad_output^T @ input_act` in emulated mxfp8. Returns `grad_weight_t (E,K,N)`.

    Both operands are transposed so the token dim M (the group-partitioned contraction axis) is
    last, then quantized 1x32 along M."""
    go_t_fp8, go_t_scale = quantize_2d_act(grad_output.transpose(-2, -1).contiguous())  # (N,M)
    ia_t_fp8, ia_t_scale = quantize_2d_act(input_act.transpose(-2, -1).contiguous())  # (K,M)
    grad_weight = emulated_mxfp8_grouped_mm_2d_2d(
        go_t_fp8, go_t_scale, ia_t_fp8, ia_t_scale, offs=offs, out_dtype=out_dtype
    )  # (E, N, K)
    return grad_weight.transpose(-2, -1)  # (E, K, N), matching weight_t


class MXFP8GroupedMM(torch.autograd.Function):
    """Differentiable emulated mxfp8 2d-3d grouped GEMM for MoE training.

    Thin wired copy of torchao's `_MXFP8GroupedMM` (minus token padding, MXTensor inputs, and
    kernel-preference dispatch): forward quantizes both operands and runs the emulated 2d-3d GEMM;
    backward computes grad_input (dgrad) and grad_weight_t (wgrad) via the emulated backward GEMMs.
    """

    @staticmethod
    def forward(ctx, input_act, weight_t, offs, out_dtype=torch.bfloat16):
        assert input_act.ndim == 2, "input_act must be 2D (M, K)"
        assert weight_t.ndim == 3, "weight_t must be 3D (E, K, N)"
        out = mxfp8_fwd_emulated(input_act, weight_t, offs, out_dtype=out_dtype)
        ctx.save_for_backward(input_act, weight_t, offs)
        ctx.out_dtype = out_dtype
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_act, weight_t, offs = ctx.saved_tensors
        grad_input = mxfp8_dgrad_emulated(grad_output, weight_t, offs, out_dtype=ctx.out_dtype)
        grad_weight_t = mxfp8_wgrad_emulated(grad_output, input_act, offs, out_dtype=ctx.out_dtype)
        return grad_input, grad_weight_t, None, None


def mxfp8_grouped_mm(input_act, weight_t, offs, out_dtype=torch.bfloat16):
    """Differentiable emulated mxfp8 grouped GEMM: `out = grouped_mm(input_act, weight_t)`."""
    return MXFP8GroupedMM.apply(input_act, weight_t, offs, out_dtype)


# Re-exported so tests / consumers share one SQNR definition with the gold recipes.
compute_error = _compute_error
