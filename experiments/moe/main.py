"""Plain-PyTorch reference casts for MoE mxfp8 grouped GEMM (forward + backward).

Token-choice MoE training uses three grouped GEMMs, all with expert groups selected along the
token dimension by an int32 `offs` tensor (group `i` spans rows `[offs[i-1], offs[i])`):

    forward : out        = grouped_mm(act (M,K),        weight_t (E,K,N))          -> (M, N)
    dgrad   : grad_input = grouped_mm(grad_output (M,N), weight (E,N,K))            -> (M, K)
    wgrad   : grad_weight = per-group  grad_output(M,N)^T @ input_act(M,K)          -> (E, K, N)

where `weight_t = weight.transpose(-2, -1)` (`weight` is the natural `(E, N, K)` expert stack).

This module holds the plain-PyTorch (no triton/cute/cuda, no torchao) recipe for the mxfp8 MoE
casts: quantize each operand to e4m3 data + e8m0 (power-of-two, block size 32) scales along the
appropriate contraction axis, then compute the grouped GEMM. It offers two paths, both plain
PyTorch and torchao-free:

  * Emulated: dequantize the mxfp8 tensors back to bf16 and call plain `torch._grouped_mm`,
    mirroring torchao's `_emulated_mxfp8_scaled_grouped_mm_2d_{3d,2d}`. Consumes the naive
    (unswizzled) e8m0 scales.
  * Real: call the actual mxfp8 `torch._scaled_grouped_mm`, mirroring torchao's
    `_compute_{fwd,dgrad,wgrad}_sm100` (torchao/prototype/moe_training/mxfp8_grouped_mm.py). This
    requires (a) the e8m0 scales in the NVIDIA blocked/swizzled tcgen05 layout (per-group padded to
    128 rows / 4 cols) and (b) token groups padded to a multiple of the block size so every group is
    block-aligned; the M-dim outputs are unpadded afterward. `torch._scaled_grouped_mm` is SM100-only
    -- this box is SM100 (`torch.cuda.get_device_capability() == (10, 0)`), so the real op runs here.

The e8m0 cast primitive (`mxfp8_f`) and the pure-PyTorch blocked-scale swizzle (`_to_blocked_4d`)
already live in this repo's gold recipes and are reused here rather than re-derived.
"""

import os
import sys
from enum import StrEnum

import torch

# Reuse this repo's plain-PyTorch mxfp8 primitives rather than re-deriving them:
#   mxfp8_f : 1x32 mxfp8 cast of the last dim -> (e4m3 qdata, e8m0 pow2 scale)
#   _compute_error: SQNR in dB
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quant_cast_bench.quant_cast_gold.recipes import (  # noqa: E402
    _compute_error,
    _to_blocked_4d,
    mxfp8_f,
)

BLOCK_SIZE = 32


class QuantOrientation(StrEnum):
    # Mirrors experiments/mxfp8_api/api.py: how the 1x32 block maps onto a token operand. For grouped
    # token tensors this also fixes the blocked-scale layout: NATURAL -> M-groups, TRANSPOSED ->
    # K-groups. Defined locally to keep this experiment self-contained.
    NATURAL = "natural"          # block along the last dim (contraction), qdata (M, C), M-groups scale
    TRANSPOSED = "transposed"    # block along the token dim M, qdata (C, M), K-groups scale
    BOTH = "both"                # fused: emit the NATURAL pair then the TRANSPOSED pair


def quantize_2d_act(act: torch.Tensor):
    """Quantize a 2d activation `(total_M, K)` to mxfp8 with 1x32 blocks along K.

    Returns:
        act_fp8:   `(total_M, K)` float8_e4m3fn.
        act_scale: `(total_M, K // 32)` e8m0 (float8_e8m0fnu), naive (unswizzled) layout.
    """
    assert act.ndim == 2, "act must be 2D"
    act_fp8, act_scale = mxfp8_f(act)  # blocks the last dim (K)
    return act_fp8, act_scale


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


# ===========================================================================
# Real (non-emulated) path: call the actual SM100 `torch._scaled_grouped_mm`.
#
# The qdata and `offs` are identical to the emulated path; the two differences are that the real op
# needs (a) the e8m0 scales in the NVIDIA blocked/swizzled tcgen05 layout and (b) token groups padded
# to a multiple of the block size. The blocked-scale helpers below are pure-PyTorch ports of torchao's
# `torch_to_blocked_*` (kernels/mxfp8/quant.py), substituting this repo's `_to_blocked_4d` for
# torchao's `to_blocked` (bit-identical flat buffer). The pad/unpad helpers port torchao's
# `torch_{pad,unpad}_token_groups`.
# ===========================================================================
def _ceil_div(a, b):
    return (a + b - 1) // b


def _pad_token_groups(inputs: torch.Tensor, group_offsets: torch.Tensor, alignment: int = BLOCK_SIZE):
    """Pad each token group's rows up to a multiple of `alignment` with zeros so every group is
    block-aligned. Port of torchao's `torch_pad_token_groups`. Over-allocates the output to the
    upper bound `num_tokens + num_groups * alignment` (matching torchao's kernel).

    Returns:
        padded_tokens:        `(upper_bound, dim)` zero-padded tokens.
        padded_start_offsets: `(num_groups,)` int start row of each group after padding.
        padded_offsets:       `(num_groups,)` int32 end offsets after padding.
    """
    inputs = inputs.contiguous()
    num_tokens, dim = inputs.shape
    num_groups = group_offsets.shape[0]
    zero = torch.zeros(1, dtype=group_offsets.dtype, device=group_offsets.device)
    group_sizes = torch.diff(group_offsets, prepend=zero)
    padded_sizes = _ceil_div(group_sizes, alignment) * alignment
    padded_offsets = torch.cumsum(padded_sizes, 0, dtype=torch.int32)
    padded_start_offsets = padded_offsets - padded_sizes

    output_rows = _ceil_div(num_tokens + num_groups * alignment, alignment) * alignment
    padded_tokens = inputs.new_zeros(output_rows, dim)
    chunks = inputs.split(group_sizes.tolist(), dim=0)
    for chunk, padded_start in zip(chunks, padded_start_offsets.tolist()):
        padded_tokens[padded_start : padded_start + chunk.shape[0]] = chunk
    return padded_tokens, padded_start_offsets, padded_offsets


def _unpad_token_groups(
    padded_inputs: torch.Tensor,
    group_offsets: torch.Tensor,
    padded_offsets: torch.Tensor,
) -> torch.Tensor:
    """Inverse of `_pad_token_groups`: gather each group's original-size chunk back into a
    `(num_tokens, dim)` tensor. Port of torchao's `torch_unpad_token_groups`. The padded start rows
    and the original token count are both recovered from `padded_offsets` + `group_offsets`, so the
    grouped cast only has to hand back `padded_offsets`."""
    zero = torch.zeros(1, dtype=group_offsets.dtype, device=group_offsets.device)
    group_sizes = torch.diff(group_offsets, prepend=zero)
    padded_start_offsets = padded_offsets - torch.diff(padded_offsets, prepend=zero)
    num_tokens = int(group_offsets[-1])
    chunks = [
        padded_inputs[start : start + size]
        for start, size in zip(padded_start_offsets.tolist(), group_sizes.tolist())
    ]
    unpadded = torch.cat(chunks, dim=0)
    assert unpadded.shape[0] == num_tokens, f"unpad got {unpadded.shape[0]}, expected {num_tokens}"
    return unpadded


def _to_blocked_2d_m_groups(x_scales: torch.Tensor, group_offs: torch.Tensor) -> torch.Tensor:
    """Blocked scale layout for 2d scales grouped along rows (the token dim M). Port of torchao's
    `torch_to_blocked_2d_M_groups`: each group's scales are swizzled with `_to_blocked_4d` and written
    at a running row offset, each group padded to a multiple of 128 rows."""
    assert x_scales.ndim == 2, "x_scales must be 2D"
    total_M, scale_cols = x_scales.shape
    num_groups = group_offs.shape[0]
    blocked_scales = x_scales.new_zeros(total_M + num_groups * 128, scale_cols)
    group_start_idx = 0
    prev_start_row = 0
    for group_end_idx in group_offs.tolist():
        group_size = group_end_idx - group_start_idx
        if group_size == 0:
            continue
        group_blocked = _to_blocked_4d(x_scales[group_start_idx:group_end_idx]).reshape(-1, scale_cols)
        group_rows_padded = _ceil_div(group_size, 128) * 128
        blocked_scales[prev_start_row : prev_start_row + group_rows_padded] = group_blocked
        prev_start_row += group_blocked.shape[0]
        group_start_idx = group_end_idx
    return blocked_scales


def _to_blocked_per_group_3d(scales: torch.Tensor) -> torch.Tensor:
    """Blocked scale layout for a 3d weight `(E, rows, cols)`: swizzle each expert's 2d scale with
    `_to_blocked_4d` and flatten. Port of torchao's `torch_to_blocked_per_group_3d`. Returns
    `(E, flat_len)`."""
    assert scales.ndim == 3, "scales must be 3D (E, rows, cols)"
    per_expert = [_to_blocked_4d(scales[i]).reshape(-1) for i in range(scales.shape[0])]
    return torch.stack(per_expert, dim=0).contiguous()


def _to_blocked_2d_k_groups(x_scales: torch.Tensor, group_offs: torch.Tensor) -> torch.Tensor:
    """Blocked scale layout for 2d scales grouped along cols (the contraction dim, in scale space).
    Port of torchao's `torch_to_blocked_2d_K_groups`: rows padded to 128, each group's cols padded to
    4; per (128,4) scale tile the swizzled (512,) block is scattered into the flat output at a running
    column offset."""
    assert x_scales.ndim == 2, "x_scales must be 2D"
    M, total_K = x_scales.shape
    padded_M = _ceil_div(M, 128) * 128
    num_groups = group_offs.shape[0]
    blocked_scales = x_scales.new_zeros(padded_M, total_K + num_groups * 4)
    blocked_flat = blocked_scales.view(-1)
    stride_per_block = 128 * 4  # 512, a swizzled (128,4) scale tile
    num_row_blocks = _ceil_div(M, 128)
    group_start_idx = 0
    prev_start_col = 0
    for group_end_idx in group_offs.tolist():
        group_size = group_end_idx - group_start_idx
        if group_size == 0:
            continue
        cols_after_padding = _ceil_div(group_size, 4) * 4
        num_col_blocks = cols_after_padding // 4
        group_blocked = _to_blocked_4d(x_scales[:, group_start_idx:group_end_idx]).reshape(
            num_row_blocks, num_col_blocks, -1
        )
        base = prev_start_col * padded_M
        for row_block in range(num_row_blocks):
            for col_block in range(num_col_blocks):
                offset = (
                    base
                    + row_block * num_col_blocks * stride_per_block
                    + col_block * stride_per_block
                )
                blocked_flat[offset : offset + stride_per_block] = group_blocked[row_block, col_block]
        prev_start_col += cols_after_padding
        group_start_idx = group_end_idx
    return blocked_scales


def quantize_to_mxfp8_grouped(
    input: torch.Tensor,  # (total_M, C)
    offs: torch.Tensor,
    orientation: QuantOrientation = QuantOrientation.NATURAL,
):
    """One-shot mxfp8 cast of a grouped token operand for the real `torch._scaled_grouped_mm`, the
    grouped analog of the dense `experiments/mxfp8_api/api.py::quantize_to_mxfp8`. Composes this
    module's existing helpers (token-group pad -> `mxfp8_f` -> blocked-scale swizzle); no new math.

    `orientation` is the only knob and it drives everything: NATURAL blocks the 1x32 along the last
    (contraction) dim and emits the M-groups blocked scale; TRANSPOSED blocks along the token dim M
    and emits the K-groups blocked scale (scale offsets = padded_offs // BLOCK_SIZE, internal). BOTH
    emits both pairs from a single padded read -- this is the fusion-visible case a future kernel would
    collapse into one pass. (A fuller superset -- scaling_type / swizzle_type / rounding_mode -- would
    mirror the dense API; only orientation is needed here today.)

    qdata is always returned row-major/contiguous; the caller composes the mat2 `.transpose(-2, -1)`
    view at the GEMM call site. `padded_offs` (token space) is returned for the op's `offs=` and for
    `_unpad_token_groups`; the padded start rows and original token count are recoverable from it.

    Returns:
        NATURAL:    `(q (Mp, C),  scale_blocked_m_groups, padded_offs)`
        TRANSPOSED: `(q (C,  Mp), scale_blocked_k_groups, padded_offs)`
        BOTH:       `(q_natural (Mp, C), scale_blocked_m_groups,
                      q_transposed (C, Mp), scale_blocked_k_groups, padded_offs)`
    """
    padded, _, padded_offs = _pad_token_groups(input, offs)

    if orientation in (QuantOrientation.NATURAL, QuantOrientation.BOTH):
        q_nat, s_nat = quantize_2d_act(padded)  # (Mp, C), 1x32 along C
        sb_nat = _to_blocked_2d_m_groups(s_nat, padded_offs)
    if orientation in (QuantOrientation.TRANSPOSED, QuantOrientation.BOTH):
        q_t, s_t = quantize_2d_act(padded.transpose(-2, -1).contiguous())  # (C, Mp), 1x32 along M
        sb_t = _to_blocked_2d_k_groups(s_t, padded_offs // BLOCK_SIZE)

    if orientation == QuantOrientation.NATURAL:
        return q_nat, sb_nat, padded_offs
    if orientation == QuantOrientation.TRANSPOSED:
        return q_t, sb_t, padded_offs
    return q_nat, sb_nat, q_t, sb_t, padded_offs


def quantize_to_mxfp8_3d(
    input: torch.Tensor,  # (E, R, C)
    orientation: QuantOrientation = QuantOrientation.NATURAL,
):
    """Batched (per-expert) mxfp8 cast of a 3d weight stack `(E, R, C)` for the real
    `torch._scaled_grouped_mm` -- the 3d analog of the dense `experiments/mxfp8_api::quantize_to_mxfp8`.
    Unlike `quantize_to_mxfp8_grouped`, the expert axis is a plain batch dim with NO offsets (each
    expert is a full dense matrix), so this stays a separate weight-only path. Composes this module's
    `mxfp8_f` + `_to_blocked_per_group_3d`; no new math.

    `orientation` picks the blocked axis: NATURAL blocks the 1x32 along the last dim C (qdata `(E,R,C)`,
    scale `(E,R,C//32)`); TRANSPOSED blocks along R (qdata `(E,C,R)`). BOTH emits both pairs from one
    read -- the fusion-visible case: for a `weight_t (E,K,N)` stack, NATURAL is the dgrad-B cast (block
    along N) and TRANSPOSED is the fwd-B cast (block along K), so BOTH yields both weight casts a
    forward+backward step needs in a single pass.

    qdata is returned row-major/contiguous; the caller composes the mat2 `.transpose(-2, -1)` view at
    the GEMM call site.

    Returns:
        NATURAL:    `(q (E,R,C), scale_blocked)`
        TRANSPOSED: `(q (E,C,R), scale_blocked)`
        BOTH:       `(q_natural (E,R,C), scale_blocked_natural,
                      q_transposed (E,C,R), scale_blocked_transposed)`
    """
    assert input.ndim == 3, "input must be 3D (E, R, C)"

    if orientation in (QuantOrientation.NATURAL, QuantOrientation.BOTH):
        q_nat, s_nat = mxfp8_f(input.contiguous())  # (E,R,C), 1x32 along C
        sb_nat = _to_blocked_per_group_3d(s_nat)
    if orientation in (QuantOrientation.TRANSPOSED, QuantOrientation.BOTH):
        q_t, s_t = mxfp8_f(input.transpose(-2, -1).contiguous())  # (E,C,R), 1x32 along R
        sb_t = _to_blocked_per_group_3d(s_t)

    if orientation == QuantOrientation.NATURAL:
        return q_nat, sb_nat
    if orientation == QuantOrientation.TRANSPOSED:
        return q_t, sb_t
    return q_nat, sb_nat, q_t, sb_t


def mxfp8_fwd_real(
    act: torch.Tensor,  # (M, K)
    weight_t: torch.Tensor,  # (E, K, N)
    offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Forward: `out = grouped_mm(act, weight_t)` via the real `torch._scaled_grouped_mm`. Returns
    `(M, N)`. Mirrors torchao's `_compute_fwd_sm100` (with token-group padding)."""
    # Activation: block 1x32 along K (the contraction dim) -> M-groups blocked scale.
    act_fp8, act_scale_blocked, padded_offs = quantize_to_mxfp8_grouped(
        act, offs, QuantOrientation.NATURAL
    )
    # Weight cast blocked 1x32 along K (the fwd contraction dim): TRANSPOSED gives a (E,N,K) row-major
    # buffer whose transpose (E,K,N) is the column-major mat2 view the real op requires.
    w_e4m3, w_scale_blocked = quantize_to_mxfp8_3d(weight_t, QuantOrientation.TRANSPOSED)  # (E,N,K)
    out = torch._scaled_grouped_mm(
        act_fp8, w_e4m3.transpose(-2, -1), act_scale_blocked, w_scale_blocked,
        offs=padded_offs, out_dtype=out_dtype,
    )
    return _unpad_token_groups(out, offs, padded_offs)


def mxfp8_bwd_real(
    grad_output: torch.Tensor,  # (M, N)
    input_act: torch.Tensor,  # (M, K)
    weight_t: torch.Tensor,  # (E, K, N)
    offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for the real mxfp8 grouped GEMM. Computes BOTH gradients from one entry point:
      * dgrad: `grad_input = grouped_mm(grad_output, weight)` (2d-3d), returns `(M, K)`;
      * wgrad: per-group `grad_output^T @ input_act` (2d-2d), returns `grad_weight_t (E, K, N)`.
    Mirrors torchao's `_compute_dgrad_sm100` + `_compute_wgrad_sm100`.

    Kept as one function so the work shared between the two GEMMs (currently separate PyTorch ops) is
    visible as kernel-fusion opportunities:
      * `grad_output` is padded once and feeds both GEMMs, but in two orientations -- dgrad wants it
        row-cast `(Mp, N)` (1x32 along N), wgrad wants it transposed-cast `(N, Mp)` (1x32 along M).
        Those two casts of one tensor are exactly a fused both-orientation (dim_km) mxfp8 cast: a
        single kernel could emit both qdata+scale pairs from one read of `padded_go`.
      * both `torch._scaled_grouped_mm` calls share `padded_offs` (and the dgrad call reuses the
        row-cast of `grad_output`), so a fused backward would quantize `grad_output` just once.
    """
    # --- grad_output cast in BOTH orientations (one padded read): row-orientation feeds dgrad
    # (1x32 along N, M-groups scale), transposed-orientation feeds wgrad (1x32 along M, K-groups
    # scale). This single call is exactly the fused both-orientation cast a kernel would collapse. ---
    go_fp8, go_scale_blocked, go_t_fp8, go_t_scale_blocked, padded_offs = quantize_to_mxfp8_grouped(
        grad_output, offs, QuantOrientation.BOTH
    )

    # === dgrad: grad_input = grouped_mm(grad_output, weight) ===
    # Weight blocked 1x32 along N (the dgrad contraction dim). weight_t is (E,K,N) with N last, so
    # NATURAL blocks along N directly; the transpose gives the (E,N,K) column-major mat2 view.
    q_kn, w_scale_blocked = quantize_to_mxfp8_3d(weight_t, QuantOrientation.NATURAL)  # (E,K,N)
    w_e4m3 = q_kn.transpose(-2, -1)  # (E,N,K) column-major view
    grad_input = torch._scaled_grouped_mm(
        go_fp8, w_e4m3, go_scale_blocked, w_scale_blocked, offs=padded_offs, out_dtype=out_dtype
    )
    grad_input = _unpad_token_groups(grad_input, offs, padded_offs)

    # === wgrad: grad_weight_t = per-group grad_output^T @ input_act ===
    # input_act transposed so the group-partitioned contraction dim M is last, blocked 1x32 along M
    # (K-groups scale layout). M being contracted, the result needs no unpadding.
    ia_t_fp8, ia_t_scale_blocked, _ = quantize_to_mxfp8_grouped(
        input_act, offs, QuantOrientation.TRANSPOSED
    )
    grad_weight = torch._scaled_grouped_mm(
        go_t_fp8,
        ia_t_fp8.transpose(-2, -1),
        go_t_scale_blocked,
        ia_t_scale_blocked,
        offs=padded_offs,
        out_dtype=out_dtype,
    )  # (E, N, K)

    # The op leaves an empty group's output slice uninitialized (stale memory), but a group with no
    # tokens contributes exactly zero weight gradient -- zero those experts to match the reference.
    # TODO fix this in core
    group_sizes = torch.diff(padded_offs, prepend=padded_offs.new_zeros(1))
    grad_weight[group_sizes == 0] = 0

    grad_weight_t = grad_weight.transpose(-2, -1)  # (E, K, N), matching weight_t

    return grad_input, grad_weight_t


class MXFP8GroupedMMReal(torch.autograd.Function):
    """Differentiable real mxfp8 2d-3d grouped GEMM for MoE training. Same structure as
    `MXFP8GroupedMM`, but forward/backward call the `*_real` GEMMs (real `torch._scaled_grouped_mm`
    with blocked scales + token-group padding) instead of the emulated ones."""

    @staticmethod
    def forward(ctx, input_act, weight_t, offs, out_dtype=torch.bfloat16):
        assert input_act.ndim == 2, "input_act must be 2D (M, K)"
        assert weight_t.ndim == 3, "weight_t must be 3D (E, K, N)"
        out = mxfp8_fwd_real(input_act, weight_t, offs, out_dtype=out_dtype)
        ctx.save_for_backward(input_act, weight_t, offs)
        ctx.out_dtype = out_dtype
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_act, weight_t, offs = ctx.saved_tensors
        grad_input, grad_weight_t = mxfp8_bwd_real(
            grad_output, input_act, weight_t, offs, out_dtype=ctx.out_dtype
        )
        return grad_input, grad_weight_t, None, None


def mxfp8_grouped_mm_real(input_act, weight_t, offs, out_dtype=torch.bfloat16):
    """Differentiable real mxfp8 grouped GEMM: `out = grouped_mm(input_act, weight_t)`."""
    return MXFP8GroupedMMReal.apply(input_act, weight_t, offs, out_dtype)


# Re-exported so tests / consumers share one SQNR definition with the gold recipes.
compute_error = _compute_error
