"""Standalone correctness tests for the golden quant-cast recipes.

Each `QuantCastSingleKernelGold` must be internally consistent: running its `correctness_fn`
on `pt_ref_fn`'s own outputs has to pass. That's a gold-package concern (no flex_tile_map
involved), so it lives here rather than in flex_tile_map/test.py. Kept independent of
flex_tile_map -- inputs (and any aux args) come from each recipe's own `example_input_fn`.
"""

import os
import sys

import pytest
import torch
import torch.func._random as prng
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_bench.quant_cast_gold.recipes import (
    ALL_RECIPES,
    F4_E2M1_MAX,
    F8E4M3_MAX,
    _compute_error,
    hadamard_rht_matrix,
    hadamard_rht_f,
    nvfp4_gs_scale,
    nvfp4_gs_swizzle_dim_k_dim_m_rht_f,
    nvfp4_gs_swizzle_dim_k_dim_m_rht_sr_f,
    nvfp4_gs_swizzle_f,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.mark.parametrize(
    "name, gold",
    ALL_RECIPES,
    ids=[name for name, _ in ALL_RECIPES],
)
def test_ref_correctness(name, gold):
    # each gold recipe is internally consistent: pt_ref_fn's own outputs clear its correctness_fn.
    # example_input_fn builds the full positional inputs (x, *aux). Calls pt_ref_fn directly on
    # the whole tensor (no flex_tile_map). The whole tensor is one tile, so we pass the origin
    # position kwargs a INDUCTOR-style whole-tensor call would -- recipes that ignore them accept
    # **kwargs; sr_bf16_global needs them for its per-element global-position dither.
    torch.manual_seed(0)
    inputs = gold.example_input_fn(512, 512)

    outputs = gold.pt_ref_fn(*inputs, global_row=0, global_col=0, num_col=inputs[0].shape[1])
    gold.correctness_fn(inputs, outputs)  # raises AssertionError on failure


# ===========================================================================
# End-to-end nvfp4 linear (fwd + bwd) built ONLY from the gold casts + the real
# torch.nn.functional.scaled_mm GEMM (no triton kernels). Mirrors torchao's dense nvfp4
# pretraining recipe (nvfp4_mm_triton in torchao/prototype/moe_training/nvfp4_training/
# nvfp4_linear.py) -- same operand orientations, two-level scales, RHT placement, scaled_mm
# signature, AND stochastic-rounding (SR) placement: SR on exactly the two grad_output casts
# (dgrad-row + wgrad-col), round-to-nearest (RTN) on the activation and weight casts. SR keeps the
# fp4 grad_output cast an unbiased estimator (E[SR(v)] = v), so no deterministic per-element rounding
# bias leaks into grad_input / grad_weight -- the point of SR in low-precision training. Our SR is
# software SR (add a uniform dither into the discarded mantissa bits, then truncate; see
# nvfp4_gs_swizzle_dim_k_dim_m_rht_sr_f), not the NVIDIA cvt.rs hardware intrinsics; with fixed Philox
# keys the grads stay reproducible run-to-run here.
#
# Linear: out = input @ weight.T, input (M,K), weight (N,K), out (M,N). Three GEMMs:
#   fwd   out         = input @ W.T   : input row (blk K, no RHT) x weight row (blk K, no RHT)
#   dgrad grad_input  = dy @ W        : dy row (blk N, no RHT, SR) x W.T col (blk N, no RHT)
#   wgrad grad_weight = dy.T @ input  : dy col=RHT(dy.T) (blk M, SR) x input col=RHT(input.T) (blk M)
# RHT is applied ONLY in wgrad, to both operands; the two RHTs cancel (H @ H.T = I) so wgrad stays
# correct while the transform cuts the outer-product quantization variance. The activation needs a
# row + col-RHT cast in one shot -> nvfp4_gs_swizzle_dim_k_dim_m_rht_f (torchao's
# _rht_quantize_row_col, RTN); grad_output needs the same but SR ->
# nvfp4_gs_swizzle_dim_k_dim_m_rht_sr_f; the weight needs a plain row + col cast (no RHT) ->
# nvfp4_gs_swizzle_f (torchao's _weight_quantize_2d).
# ===========================================================================
requires_sm100 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)),
    reason="nvfp4 torch.nn.functional.scaled_mm emits Blackwell-only PTX; requires SM100",
)

def _rht_outer_scale(x, rht):
    """Per-tensor fp32 outer scale over |RHT(x.T)| (the RHT-path amax basis), same formula as the
    dim_k_dim_m_rht gold's own inputs helper."""
    (x_rht,) = hadamard_rht_f(x.t().contiguous(), rht)
    return x_rht.abs().to(torch.float32).amax() / (F8E4M3_MAX * F4_E2M1_MAX)


class _Nvfp4Linear(torch.autograd.Function):
    """Reference nvfp4 linear composed from the gold casts + torch.nn.functional.scaled_mm. RTN on
    the activation/weight casts, stochastic rounding on the two grad_output casts (as torchao does).
    See the module comment above for the per-GEMM cast/orientation/RHT breakdown."""

    @staticmethod
    def forward(ctx, input, weight, rht, key):
        # input: x, shape [M, K]
        # weight: w, shape [N, K]
        # grad_output: go, shape [M, K]
        # key: caller-supplied prng key (prng.key(seed)); threaded to backward for the SR casts.

        # Activation: row cast (no RHT) feeds fwd; col cast (RHT on input.T) is saved for wgrad.
        # The gold casts take 1/S (recipes' MSLK/torchao global_scale convention) while F.scaled_mm
        # takes S (the dequant multiplier), so reciprocate at each cast and keep S for the gemm scales.
        x_gs_k = nvfp4_gs_scale(input)  # outer scale S over |input|
        x_rht_g_s_m = _rht_outer_scale(input, rht)  # outer scale S over |RHT(input.T)|
        x_q_k, xs_k, x_rht_q_m, x_rht_s_m = nvfp4_gs_swizzle_dim_k_dim_m_rht_f(
            input, x_gs_k.reciprocal(), x_rht_g_s_m.reciprocal(), rht
        )
        # Weight: row cast (blk K) feeds fwd; transposed row cast (blk N) is the dgrad col operand.
        w_gs = nvfp4_gs_scale(weight)  # |W| == |W.T|, so one outer scale serves both
        w_q_k, ws_k = nvfp4_gs_swizzle_f(weight, w_gs.reciprocal())
        w_q_n, w_s_n = nvfp4_gs_swizzle_f(weight.t().contiguous(), w_gs.reciprocal())
        # fwd: (M,K) @ (K,N) -> (M,N). Each operand's scale is [1x16 block-wise e4m3 (the 4D swizzle
        # grid nvfp4_gs_swizzle_f emits, flattened as torchao does), tensor-wise fp32 outer scalar].
        out = F.scaled_mm(
            x_q_k, w_q_k.t(),
            scale_a=[xs_k.flatten(), x_gs_k], scale_b=[ws_k.flatten(), w_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )
        ctx.save_for_backward(x_rht_q_m, x_rht_s_m, x_rht_g_s_m, w_q_n, w_s_n, w_gs, rht, key)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_rht_q_m, x_rht_s_m, x_rht_g_s_m, w_q_n, w_s_n, w_gs, rht, key = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        # grad_output: row cast (no RHT) feeds dgrad; col cast (RHT on dy.T) feeds wgrad. Both use
        # STOCHASTIC ROUNDING -- torchao applies SR to exactly these two grad_output casts (the
        # activation and weight casts stay RTN), because an unbiased grad cast is what keeps the
        # gradient estimator unbiased in expectation over training steps. prng.split derives two
        # independent substreams (one per direction) from the caller's key, giving the two casts
        # uncorrelated dither. The caller controls reproducibility (and, if wanted, per-step freshness)
        # by choosing what key it passes to forward -- e.g. prng.fold_in(base_key, step); this Function
        # just splits whatever it is handed.
        go_gs_k = nvfp4_gs_scale(grad_output)  # outer scale S over |grad_output|
        go_rht_g_s_m = _rht_outer_scale(grad_output, rht)  # outer scale S over |RHT(grad_output.T)|
        key_k, key_m = prng.split(key, 2)
        # gold casts take 1/S, F.scaled_mm below takes S (see forward): reciprocate for the casts only.
        go_sr_q_k, gos_k, go_sr_q_m, go_s_m = nvfp4_gs_swizzle_dim_k_dim_m_rht_sr_f(
            grad_output, go_gs_k.reciprocal(), go_rht_g_s_m.reciprocal(), rht, key_k, key_m
        )
        # dgrad: dy (M,N) @ W (N,K) -> grad_input (M,K).
        grad_input = F.scaled_mm(
            go_sr_q_k, w_q_n.t(),
            scale_a=[gos_k.flatten(), go_gs_k], scale_b=[w_s_n.flatten(), w_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )
        # wgrad: RHT(dy.T) (N,M) @ RHT(input.T).T (M,K) -> grad_weight (N,K); the two RHTs cancel.
        grad_weight = F.scaled_mm(
            go_sr_q_m, x_rht_q_m.t(),
            scale_a=[go_s_m.flatten(), go_rht_g_s_m], scale_b=[x_rht_s_m.flatten(), x_rht_g_s_m],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )
        return grad_input, grad_weight, None, None  # extra None: rht, key


@requires_sm100
def test_nvfp4_linear_fwd_bwd_sqnr():
    # Full fwd+bwd of a linear in nvfp4 (gold casts + real scaled_mm) vs a plain bf16 torch.mm
    # reference, comparing output + both gradients by SQNR. M,K,N all %128 (nvfp4 scaled_mm needs it).
    torch.manual_seed(0)
    M, K, N = 256, 512, 1024
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")  # fixed upstream grad
    sign = torch.tensor([1, -1] * 8, device=x.device, dtype=x.dtype)  # fixed RHT sign vector
    rht = hadamard_rht_matrix(sign, x.device, x.dtype)
    key = prng.key(0, device=x.device)  # caller-supplied SR key; backward splits it into two streams

    # bf16 reference: out = x @ w.T, then backward with the same upstream grad.
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    (xr @ wr.t()).backward(grad_out)

    # nvfp4 path through the reference autograd Function.
    xq = x.clone().requires_grad_(True)
    wq = w.clone().requires_grad_(True)
    _Nvfp4Linear.apply(xq, wq, rht, key).backward(grad_out)

    out_ref = xr @ wr.t()
    out_q = _Nvfp4Linear.apply(x, w, rht, key)
    sqnr_out = _compute_error(out_ref.float(), out_q.float())
    sqnr_gx = _compute_error(xr.grad.float(), xq.grad.float())
    sqnr_gw = _compute_error(wr.grad.float(), wq.grad.float())

    # nvfp4 is 4-bit and every GEMM operand is quantized. The forward (RTN) sits ~17.4 dB across
    # seeds, so 15 dB (torchao's forward bar) leaves margin. The grads are ~2 dB lower (~15.6 dB):
    # their grad_output cast uses SR, which is unbiased but has higher per-element variance than RTN,
    # so a SINGLE-realization SQNR against the bf16 reference is worse than RTN would give here (SR's
    # win is unbiased accumulation over many steps, not one-shot error). Floor the grads at 13 dB for
    # a comfortable margin over that stable ~15.6.
    assert sqnr_out > 15.0, f"output sqnr={sqnr_out.item():.2f} dB below 15 dB"
    assert sqnr_gx > 13.0, f"grad_input sqnr={sqnr_gx.item():.2f} dB below 13 dB"
    assert sqnr_gw > 13.0, f"grad_weight sqnr={sqnr_gw.item():.2f} dB below 13 dB"
