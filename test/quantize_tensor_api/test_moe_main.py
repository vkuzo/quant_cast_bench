"""Correctness tests for the REAL MoE mxfp8 grouped-GEMM casts (forward + backward).

Each test quantizes random bf16 inputs to mxfp8 with the plain-PyTorch reference in
`quant_cast_bench/quantize_tensor_api/moe_main.py` (blocked scales + token-group padding -> the actual
`torch._scaled_grouped_mm`) and compares against the full-precision bf16 formulation of the same GEMM
via SQNR. The real path needs the SM100-only `torch._scaled_grouped_mm`; these tests are skipped off
SM100. The plain-PyTorch emulated companion path is tested in `test_moe_emulated.py`.
"""

import pytest
import torch

from quant_cast_bench.quantize_tensor_api.moe_main import (
    _unpad_token_groups,
    compute_error,
    mxfp8_bwd_real,
    mxfp8_fwd_real,
    mxfp8_grouped_mm_real,
)
from quant_cast_bench.quantize_tensor_api.moe_utils import _pad_token_groups

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# The real mxfp8 `torch._scaled_grouped_mm` is SM100-only.
requires_sm100 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)),
    reason="real torch._scaled_grouped_mm requires SM100",
)


def generate_jagged_offs(num_groups, M, multiple_of=32, device="cuda"):
    """Random sorted int32 group-end offsets along `total_M`, each a multiple of `multiple_of`,
    with the final offset equal to `M`. Groups may be empty (offsets can repeat)."""
    assert M % multiple_of == 0
    total_chunks = M // multiple_of
    if num_groups == 1:
        boundaries = torch.tensor([total_chunks], device=device)
    else:
        cuts = torch.randint(0, total_chunks + 1, (num_groups - 1,), device=device)
        cuts = cuts.sort().values
        boundaries = torch.cat([cuts, torch.tensor([total_chunks], device=device)])
    return (boundaries * multiple_of).to(torch.int32)


@requires_sm100
def test_mxfp8_forward_2d_3d_real():
    """Forward via the real op: out = grouped_mm(act, weight_t) -- 2d-3d, grouped along M."""
    torch.manual_seed(0)
    M, K, N = 256, 512, 1024
    num_experts = 8

    offs = generate_jagged_offs(num_experts, M)
    act = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    weight_t = weight.transpose(-2, -1)  # (E, K, N)

    ref = torch._grouped_mm(act, weight_t, offs=offs, out_dtype=torch.bfloat16)
    # mxfp8_fwd_real operates purely in padded space; the caller owns pad in / unpad out.
    padded_act, _, padded_offs = _pad_token_groups(act, offs)
    padded_out = mxfp8_fwd_real(padded_act, weight_t, padded_offs)
    out = _unpad_token_groups(padded_out, offs, padded_offs)

    sqnr = compute_error(ref.float(), out.float())
    assert sqnr > 20.0, f"mxfp8 real forward 2d-3d: sqnr={sqnr.item():.2f} dB too low"


@requires_sm100
def test_mxfp8_backward_2d_real():
    """Backward via the real op: the combined `mxfp8_bwd_real` returns both gradients --
    grad_input = grouped_mm(grad_output, weight) (dgrad, 2d-3d, grouped along M) and
    grad_weight_t = per-group grad_output^T @ input_act (wgrad, 2d-2d, grouped along M)."""
    torch.manual_seed(0)
    M, K, N = 256, 512, 1024
    num_experts = 8

    offs = generate_jagged_offs(num_experts, M)
    grad_output = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    input_act = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    weight_t = weight.transpose(-2, -1)  # (E, K, N)

    grad_input_ref = torch._grouped_mm(grad_output, weight, offs=offs, out_dtype=torch.bfloat16)
    grad_weight_t_ref = torch._grouped_mm(
        grad_output.transpose(-2, -1), input_act, offs=offs, out_dtype=torch.bfloat16
    ).transpose(-2, -1)

    # mxfp8_bwd_real operates purely in padded space; the caller owns pad in / unpad out (in the
    # autograd Function the padded input_act is carried over from forward, but grad_output and
    # input_act share `offs` so both land on the same padded offsets).
    padded_go, _, padded_offs = _pad_token_groups(grad_output, offs)
    padded_ia, _, _ = _pad_token_groups(input_act, offs)
    padded_grad_input, grad_weight_t = mxfp8_bwd_real(padded_go, padded_ia, weight_t, padded_offs)
    grad_input = _unpad_token_groups(padded_grad_input, offs, padded_offs)

    din_sqnr = compute_error(grad_input_ref.float(), grad_input.float())
    dw_sqnr = compute_error(grad_weight_t_ref.float(), grad_weight_t.float())
    assert din_sqnr > 20.0, f"mxfp8 real dgrad 2d-3d: sqnr={din_sqnr.item():.2f} dB too low"
    assert dw_sqnr > 20.0, f"mxfp8 real wgrad 2d-2d: sqnr={dw_sqnr.item():.2f} dB too low"


@requires_sm100
def test_mxfp8_fwd_bwd_e2e_real():
    """End-to-end: differentiable real mxfp8 grouped GEMM vs a bf16 autograd reference."""
    torch.manual_seed(0)
    M, K, N = 256, 512, 1024
    num_experts = 8

    offs = generate_jagged_offs(num_experts, M)
    act = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight_t = torch.randn(
        num_experts, K, N, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    act_ref = act.detach().clone().requires_grad_(True)
    weight_t_ref = weight_t.detach().clone().requires_grad_(True)

    out = mxfp8_grouped_mm_real(act, weight_t, offs)
    out_ref = torch._grouped_mm(act_ref, weight_t_ref, offs=offs, out_dtype=torch.bfloat16)

    out.backward(grad_out)
    out_ref.backward(grad_out)

    out_sqnr = compute_error(out_ref.float(), out.float())
    din_sqnr = compute_error(act_ref.grad.float(), act.grad.float())
    dw_sqnr = compute_error(weight_t_ref.grad.float(), weight_t.grad.float())

    assert out_sqnr > 20.0, f"e2e real output: sqnr={out_sqnr.item():.2f} dB too low"
    assert din_sqnr > 20.0, f"e2e real grad_input: sqnr={din_sqnr.item():.2f} dB too low"
    assert dw_sqnr > 20.0, f"e2e real grad_weight: sqnr={dw_sqnr.item():.2f} dB too low"


@requires_sm100
def test_mxfp8_fwd_bwd_e2e_real_unaligned_offsets():
    """End-to-end real path with group offsets NOT aligned to the block size, exercising the
    token-group pad/unpad (aligned offsets make padding a no-op)."""
    torch.manual_seed(0)
    M, K, N = 256, 512, 1024
    num_experts = 8

    # multiple_of=16 -> group boundaries are generally not multiples of 32; padding is real.
    offs = generate_jagged_offs(num_experts, M, multiple_of=16)
    act = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight_t = torch.randn(
        num_experts, K, N, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    act_ref = act.detach().clone().requires_grad_(True)
    weight_t_ref = weight_t.detach().clone().requires_grad_(True)

    out = mxfp8_grouped_mm_real(act, weight_t, offs)
    out_ref = torch._grouped_mm(act_ref, weight_t_ref, offs=offs, out_dtype=torch.bfloat16)

    out.backward(grad_out)
    out_ref.backward(grad_out)

    assert out.shape == (M, N)
    out_sqnr = compute_error(out_ref.float(), out.float())
    din_sqnr = compute_error(act_ref.grad.float(), act.grad.float())
    dw_sqnr = compute_error(weight_t_ref.grad.float(), weight_t.grad.float())

    assert out_sqnr > 20.0, f"e2e real (unaligned) output: sqnr={out_sqnr.item():.2f} dB too low"
    assert din_sqnr > 20.0, f"e2e real (unaligned) grad_input: sqnr={din_sqnr.item():.2f} dB too low"
    assert dw_sqnr > 20.0, f"e2e real (unaligned) grad_weight: sqnr={dw_sqnr.item():.2f} dB too low"
