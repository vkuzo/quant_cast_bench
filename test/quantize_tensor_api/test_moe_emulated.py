"""Correctness tests for the EMULATED MoE mxfp8 grouped-GEMM casts (forward + backward).

Each test quantizes random bf16 inputs to mxfp8 with the plain-PyTorch emulated reference in
`quant_cast_bench/quantize_tensor_api/moe_emulated.py` (dequantize -> `torch._grouped_mm`) and compares against the
full-precision bf16 formulation of the same GEMM via SQNR. The emulated path needs no SM100 device;
the real `torch._scaled_grouped_mm` path is covered separately in `test_moe_moe.py`.
"""

import pytest
import torch

from quant_cast_bench.quantize_tensor_api.moe_emulated import (
    compute_error,
    mxfp8_dgrad_emulated,
    mxfp8_fwd_emulated,
    mxfp8_grouped_mm,
    mxfp8_wgrad_emulated,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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


def test_mxfp8_forward_2d_3d():
    """Forward: out = grouped_mm(act, weight_t) -- 2d-3d, grouped along M."""
    torch.manual_seed(0)
    M, K, N = 1024, 1024, 1024
    num_experts = 8

    offs = generate_jagged_offs(num_experts, M)
    act = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    weight_t = weight.transpose(-2, -1)  # (E, K, N)

    # Full-precision reference: original bf16 inputs through the plain grouped GEMM.
    ref = torch._grouped_mm(act, weight_t, offs=offs, out_dtype=torch.bfloat16)

    # mxfp8 path (emulated).
    out = mxfp8_fwd_emulated(act, weight_t, offs)

    sqnr = compute_error(ref.float(), out.float())
    threshold = 20.0  # mxfp8 vs full-precision bf16 reference (both act + weight quantized)
    assert sqnr > threshold, f"mxfp8 forward 2d-3d: sqnr={sqnr.item():.2f} dB below {threshold} dB"


def test_mxfp8_grad_input_2d_3d():
    """dgrad: grad_input = grouped_mm(grad_output, weight) -- 2d-3d, grouped along M."""
    torch.manual_seed(0)
    M, K, N = 1024, 1024, 1024
    num_experts = 8

    offs = generate_jagged_offs(num_experts, M)
    grad_output = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    weight_t = weight.transpose(-2, -1)  # (E, K, N)

    # Full-precision reference: grad_output @ weight, weight in natural (E, N, K) orientation.
    ref = torch._grouped_mm(grad_output, weight, offs=offs, out_dtype=torch.bfloat16)

    # mxfp8 path (emulated).
    grad_input = mxfp8_dgrad_emulated(grad_output, weight_t, offs)

    sqnr = compute_error(ref.float(), grad_input.float())
    threshold = 20.0
    assert sqnr > threshold, f"mxfp8 dgrad 2d-3d: sqnr={sqnr.item():.2f} dB below {threshold} dB"


def test_mxfp8_grad_weight_2d_2d():
    """wgrad: grad_weight_t = per-group grad_output^T @ input_act -- 2d-2d, grouped along M."""
    torch.manual_seed(0)
    M, K, N = 1024, 1024, 1024
    num_experts = 8

    offs = generate_jagged_offs(num_experts, M)
    grad_output = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    input_act = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    # Full-precision reference (the `wgrad_with_hp` formula): (E, N, K) then transpose to (E, K, N).
    ref = torch._grouped_mm(
        grad_output.transpose(-2, -1), input_act, offs=offs, out_dtype=torch.bfloat16
    ).transpose(-2, -1)

    # mxfp8 path (emulated).
    grad_weight_t = mxfp8_wgrad_emulated(grad_output, input_act, offs)

    sqnr = compute_error(ref.float(), grad_weight_t.float())
    threshold = 20.0
    assert sqnr > threshold, f"mxfp8 wgrad 2d-2d: sqnr={sqnr.item():.2f} dB below {threshold} dB"


def test_mxfp8_fwd_bwd_e2e():
    """End-to-end: differentiable emulated mxfp8 grouped GEMM vs a bf16 autograd reference."""
    torch.manual_seed(0)
    M, K, N = 1024, 1024, 1024
    num_experts = 8

    offs = generate_jagged_offs(num_experts, M)
    act = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight_t = torch.randn(
        num_experts, K, N, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # Reference leaves: identical values, full-precision bf16 grouped GEMM + autograd.
    act_ref = act.detach().clone().requires_grad_(True)
    weight_t_ref = weight_t.detach().clone().requires_grad_(True)

    out = mxfp8_grouped_mm(act, weight_t, offs)
    out_ref = torch._grouped_mm(act_ref, weight_t_ref, offs=offs, out_dtype=torch.bfloat16)

    out.backward(grad_out)
    out_ref.backward(grad_out)

    out_sqnr = compute_error(out_ref.float(), out.float())
    din_sqnr = compute_error(act_ref.grad.float(), act.grad.float())
    dw_sqnr = compute_error(weight_t_ref.grad.float(), weight_t.grad.float())

    assert out_sqnr > 20.0, f"e2e output: sqnr={out_sqnr.item():.2f} dB too low"
    assert din_sqnr > 20.0, f"e2e grad_input: sqnr={din_sqnr.item():.2f} dB too low"
    assert dw_sqnr > 20.0, f"e2e grad_weight: sqnr={dw_sqnr.item():.2f} dB too low"
