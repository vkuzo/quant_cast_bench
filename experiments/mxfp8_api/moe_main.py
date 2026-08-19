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

The token/weight quant casts (pad + blocked-scale swizzle) live in `api.py`
(`quantize_to_mxfp8_grouped` / `quantize_to_mxfp8_batched`) over `moe_utils`; only the M-dim unpad of
the finished output stays here, as it's a GEMM-output step, not a cast. The plain-PyTorch emulated
(dequantize-and-matmul) companion path lives in `moe_emulated.py`.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.mxfp8_api.api import (  # noqa: E402
    QuantOrientation,
    quantize_to_mxfp8_batched,
    quantize_to_mxfp8_grouped,
)
from quant_cast_bench.quant_cast_gold.recipes import _compute_error  # noqa: E402


# ===========================================================================
# Real (non-emulated) path: call the actual SM100 `torch._scaled_grouped_mm`.
#
# The token/weight casts (pad + swizzle) live in `api.py` (`quantize_to_mxfp8_grouped` /
# `quantize_to_mxfp8_batched`) over the `moe_utils` helpers; only the M-dim unpad of the finished
# output stays here, as it's a GEMM-output step, not a cast.
# ===========================================================================
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
        act, offs, orientation=QuantOrientation.NATURAL
    )
    # Weight cast blocked 1x32 along K (the fwd contraction dim): TRANSPOSED gives a (E,N,K) row-major
    # buffer whose transpose (E,K,N) is the column-major mat2 view the real op requires.
    w_e4m3, w_scale_blocked = quantize_to_mxfp8_batched(
        weight_t, orientation=QuantOrientation.TRANSPOSED
    )  # (E,N,K)
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
        grad_output, offs, orientation=QuantOrientation.BOTH
    )

    # === dgrad: grad_input = grouped_mm(grad_output, weight) ===
    # Weight blocked 1x32 along N (the dgrad contraction dim). weight_t is (E,K,N) with N last, so
    # NATURAL blocks along N directly; the transpose gives the (E,N,K) column-major mat2 view.
    q_kn, w_scale_blocked = quantize_to_mxfp8_batched(
        weight_t, orientation=QuantOrientation.NATURAL
    )  # (E,K,N)
    w_e4m3 = q_kn.transpose(-2, -1)  # (E,N,K) column-major view
    grad_input = torch._scaled_grouped_mm(
        go_fp8, w_e4m3, go_scale_blocked, w_scale_blocked, offs=padded_offs, out_dtype=out_dtype
    )
    grad_input = _unpad_token_groups(grad_input, offs, padded_offs)

    # === wgrad: grad_weight_t = per-group grad_output^T @ input_act ===
    # input_act transposed so the group-partitioned contraction dim M is last, blocked 1x32 along M
    # (K-groups scale layout). M being contracted, the result needs no unpadding.
    ia_t_fp8, ia_t_scale_blocked, _ = quantize_to_mxfp8_grouped(
        input_act, offs, orientation=QuantOrientation.TRANSPOSED
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
