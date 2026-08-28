"""Plain-PyTorch reference for the real (non-emulated) MoE mxfp8 grouped GEMM (forward + backward).

Token-choice MoE training uses three grouped GEMMs, all with expert groups selected along the
token dimension by an int32 `offs` tensor (group `i` spans rows `[offs[i-1], offs[i])`):

    forward : out        = grouped_mm(act (M,K),        weight_t (E,K,N))          -> (M, N)
    dgrad   : grad_input = grouped_mm(grad_output (M,N), weight (E,N,K))            -> (M, K)
    wgrad   : grad_weight = per-group  grad_output(M,N)^T @ input_act(M,K)          -> (E, K, N)

where `weight_t = weight.transpose(-2, -1)` (`weight` is the natural `(E, N, K)` expert stack).

This module wires the real path: call the actual mxfp8 `torch._scaled_grouped_mm`, mirroring
torchao's `_compute_{fwd,dgrad,wgrad}_sm100` (torchao/prototype/moe_training/mxfp8_grouped_mm.py). It
requires (a) the e8m0 scales in the NVIDIA blocked/swizzled tcgen05 layout (per-group padded to 128
rows / 4 cols) and (b) token groups padded to a multiple of the block size so every group is
block-aligned; the M-dim outputs are unpadded afterward. `torch._scaled_grouped_mm` is SM100-only --
this box is SM100 (`torch.cuda.get_device_capability() == (10, 0)`), so the real op runs here.

The token/weight quant casts (blocked-scale swizzle) live in `api.py` (`quantize_tensor_grouped` for
tokens; `quantize_tensor` on the per-expert weights reshaped `(E,rows,cols) -> (E*rows, cols)`, valid
because a 1x32 block is row-local and each expert is 128-row-aligned) over `moe_utils`; the
token-group padding (`_pad_token_groups`) and the M-dim unpad of the finished output stay here, as
they're GEMM-shape steps that must agree across the co-operands of each GEMM, not casts. The
plain-PyTorch emulated (dequantize-and-matmul) companion path lives in `moe_emulated.py`.
"""

import torch

from quant_cast_bench.quantize_tensor_api.api import (
    InnerScaleCalc,
    ScalingType,
    SwizzleType,
    quantize_tensor,
    quantize_tensor_grouped,
    quantize_tensor_grouped_dual,
)
from quant_cast_bench.quantize_tensor_api.moe_utils import _pad_token_groups
from quant_cast_bench.quant_cast_gold.recipes import _compute_error


# ===========================================================================
# Real (non-emulated) path: call the actual SM100 `torch._scaled_grouped_mm`.
#
# The token/weight casts (swizzle) live in `api.py` (`quantize_tensor_grouped` for tokens;
# `quantize_tensor` on per-expert weights reshaped to 2D) over the `moe_utils` helpers; the token-group
# padding and the M-dim unpad of the finished output stay here, as they're GEMM-shape steps, not casts.
# ===========================================================================
def _unpad_token_groups(
    padded_inputs: torch.Tensor,
    group_offsets: torch.Tensor,
    padded_offsets: torch.Tensor,
) -> torch.Tensor:
    """Inverse of `_pad_token_groups`: gather each group's original-size chunk back into a
    `(num_tokens, dim)` tensor. Port of torchao's `torch_unpad_token_groups`. The padded start rows
    and the original token count are both recovered from `padded_offsets` + `group_offsets`, so the
    caller only has to keep the `padded_offsets` from `_pad_token_groups`."""
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


def mxfp8_fwd_real(
    padded_act: torch.Tensor,  # (Mp, K), token groups already block-aligned
    weight_t: torch.Tensor,  # (E, K, N)
    padded_offs: torch.Tensor,  # padded group-end offsets matching `padded_act`
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Forward: `out = grouped_mm(act, weight_t)` via the real `torch._scaled_grouped_mm`. Returns the
    padded output `(Mp, N)`. Mirrors torchao's `_compute_fwd_sm100`. Both the token-group padding of
    the input and the M-dim unpad of this output are the caller's job (`MXFP8GroupedMMReal`), so this
    function operates purely in padded space."""
    # Activation: block 1x32 along K (the contraction dim) -> M-groups blocked scale.
    act_fp8, act_scale_blocked = quantize_tensor_grouped(
        padded_act, padded_offs,
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.RCEIL_E8M0,
        scaling_type=ScalingType.BlockWise1x32,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
    )
    # Weight cast blocked 1x32 along K (the fwd contraction dim). The per-expert cast is just a batched
    # 2D cast: (E,N,K) collapses to (E*N, K) since a 1x32-along-K block is row-local (never crosses the
    # N or E boundary), and with N % 128 == 0 each expert occupies whole 128-row swizzle blocks, so the
    # (E*N//128, ...) swizzled scale reshapes straight back to the per-expert (E, flat) layout. The
    # (E,N,K) qdata's transpose (E,K,N) is the column-major mat2 view the real op requires.
    E, K, N = weight_t.shape
    assert N % 128 == 0, f"per-expert weight row count N must be a multiple of 128, got {N}"
    w_q, w_scale = quantize_tensor(
        weight_t.transpose(-2, -1).reshape(E * N, K),  # (E*N, K), 1x32 along K
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.RCEIL_E8M0,
        scaling_type=ScalingType.BlockWise1x32,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
    )
    w_e4m3 = w_q.reshape(E, N, K)  # (E,N,K)
    w_scale_blocked = w_scale.reshape(E, -1)  # (E, flat) per-expert blocked scale
    return torch._scaled_grouped_mm(
        act_fp8, w_e4m3.transpose(-2, -1), act_scale_blocked, w_scale_blocked,
        offs=padded_offs, out_dtype=out_dtype,
    )


def mxfp8_bwd_real(
    padded_grad_output: torch.Tensor,  # (Mp, N), token groups already block-aligned
    padded_input_act: torch.Tensor,  # (Mp, K), the padded fwd activation reused for wgrad
    weight_t: torch.Tensor,  # (E, K, N)
    padded_offs: torch.Tensor,  # padded group-end offsets matching both padded operands
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for the real mxfp8 grouped GEMM. Computes BOTH gradients from one entry point:
      * dgrad: `grad_input = grouped_mm(grad_output, weight)` (2d-3d), returns padded `(Mp, K)`;
      * wgrad: per-group `grad_output^T @ input_act` (2d-2d), returns `grad_weight_t (E, K, N)`.
    Mirrors torchao's `_compute_dgrad_sm100` + `_compute_wgrad_sm100`. Token-group padding of the
    inputs and the M-dim unpad of `grad_input` are the caller's job (`MXFP8GroupedMMReal`), so this
    operates purely in padded space; `padded_input_act` is the padded fwd activation carried over so
    wgrad doesn't re-pad it. (wgrad's output has M contracted away, so it is never padded.)

    Kept as one function so the work shared between the two GEMMs (currently separate PyTorch ops) is
    visible as kernel-fusion opportunities:
      * `grad_output` feeds both GEMMs, but in two orientations -- dgrad wants it row-cast `(Mp, N)`
        (1x32 along N), wgrad wants it transposed-cast `(N, Mp)` (1x32 along M). Those two casts of
        one tensor are exactly a fused both-orientation (dim_km) mxfp8 cast: a single kernel could
        emit both qdata+scale pairs from one read of `padded_grad_output`.
      * both `torch._scaled_grouped_mm` calls share `padded_offs` (and the dgrad call reuses the
        row-cast of `grad_output`), so a fused backward would quantize `grad_output` just once.
    """
    # --- grad_output cast in BOTH orientations (one read): row-orientation feeds dgrad (1x32 along N,
    # M-groups scale), transposed-orientation feeds wgrad (1x32 along M, K-groups scale). This single
    # call is exactly the fused both-orientation cast a kernel would collapse. ---
    go_fp8, go_scale_blocked, go_t_fp8, go_t_scale_blocked = (
        quantize_tensor_grouped_dual(
            padded_grad_output, padded_offs,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=InnerScaleCalc.RCEIL_E8M0,
            scaling_type=ScalingType.BlockWise1x32,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
    )

    # === dgrad: grad_input = grouped_mm(grad_output, weight) ===
    # Weight blocked 1x32 along N (the dgrad contraction dim). weight_t is (E,K,N) with N last, so it
    # blocks along N directly; the per-expert cast collapses to a single 2D call: (E,K,N) reshapes to
    # (E*K, N) (a 1x32-along-N block is row-local), and with K % 128 == 0 each expert lands on whole
    # 128-row swizzle blocks, so the swizzled scale reshapes back to the per-expert (E, flat) layout.
    # The (E,K,N) qdata's transpose (E,N,K) is the column-major mat2 view.
    E, K, N = weight_t.shape
    assert K % 128 == 0, f"per-expert weight row count K must be a multiple of 128, got {K}"
    q_kn_2d, w_scale = quantize_tensor(
        weight_t.reshape(E * K, N),  # (E*K, N), 1x32 along N
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.RCEIL_E8M0,
        scaling_type=ScalingType.BlockWise1x32,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
    )
    q_kn = q_kn_2d.reshape(E, K, N)  # (E,K,N)
    w_scale_blocked = w_scale.reshape(E, -1)  # (E, flat) per-expert blocked scale
    w_e4m3 = q_kn.transpose(-2, -1)  # (E,N,K) column-major view
    grad_input = torch._scaled_grouped_mm(
        go_fp8, w_e4m3, go_scale_blocked, w_scale_blocked, offs=padded_offs, out_dtype=out_dtype
    )  # (Mp, K), padded; the caller unpads.

    # === wgrad: grad_weight_t = per-group grad_output^T @ input_act ===
    # input_act transposed so the group-partitioned contraction dim M is last, blocked 1x32 along M
    # (K-groups scale layout). M being contracted, the result needs no unpadding. `padded_input_act`
    # is the fwd activation already padded to `padded_offs`, reused here instead of re-padding.
    ia_t_fp8, ia_t_scale_blocked = quantize_tensor_grouped(
        padded_input_act.t(), padded_offs,
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.RCEIL_E8M0,
        scaling_type=ScalingType.BlockWise1x32,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
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
        # Pad token groups to the block size once here; the padded activation and its offsets are
        # saved for backward, so wgrad reuses them instead of re-padding input_act. grad_output is
        # produced fresh in backward, so it's the only tensor still padded there (same `offs`, so it
        # lands on this same `padded_offs`).
        padded_act, _, padded_offs = _pad_token_groups(input_act, offs)
        padded_out = mxfp8_fwd_real(padded_act, weight_t, padded_offs, out_dtype=out_dtype)
        out = _unpad_token_groups(padded_out, offs, padded_offs)
        ctx.save_for_backward(padded_act, weight_t, offs, padded_offs)
        ctx.out_dtype = out_dtype
        return out

    @staticmethod
    def backward(ctx, grad_output):
        padded_act, weight_t, offs, padded_offs = ctx.saved_tensors
        padded_grad_output, _, _ = _pad_token_groups(grad_output, offs)
        padded_grad_input, grad_weight_t = mxfp8_bwd_real(
            padded_grad_output, padded_act, weight_t, padded_offs, out_dtype=ctx.out_dtype
        )
        grad_input = _unpad_token_groups(padded_grad_input, offs, padded_offs)
        return grad_input, grad_weight_t, None, None


def mxfp8_grouped_mm_real(input_act, weight_t, offs, out_dtype=torch.bfloat16):
    """Differentiable real mxfp8 grouped GEMM: `out = grouped_mm(input_act, weight_t)`."""
    return MXFP8GroupedMMReal.apply(input_act, weight_t, offs, out_dtype)


# Re-exported so tests / consumers share one SQNR definition with the gold recipes.
compute_error = _compute_error
