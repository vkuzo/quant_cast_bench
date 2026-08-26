import os
import sys

import pytest
import torch
import torch.func._random as prng
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import (  # noqa: E402
    InnerScaleCalc,
    QuantOrientation,
    RoundingMode,
    ScalingType,
    SwizzleType,
    quantize_tensor,
    quantize_tensor_bidirectional,
    quantize_tensor_grouped,
    quantize_tensor_grouped_bidirectional,
)

from quant_cast_bench.quant_cast_gold.recipes import (  # noqa: E402
    F4_E2M1_MAX,
    F8E4M3_MAX,
    _compute_error,
    hadamard_rht_f,
    hadamard_rht_matrix,
    mxfp8_32x32_f,
    mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_f,
    mxfp8_dim_km_f,
    mxfp8_dim_km_swizzle_f,
    mxfp4_f,
    mxfp8_dim_m_f,
    mxfp8_dim_m_swizzle_f,
    mxfp8_f,
    mxfp8_swizzle_f,
    nvfp4_gs_f,
    nvfp4_gs_per_token_scale,
    nvfp4_gs_scale,
    nvfp4_gs_swizzle_dim_m_rht_f,
    nvfp4_gs_swizzle_f,
)

# The original all-gold reference _Nvfp4Linear lives in the repo's top-level test/ (not a package);
# add that dir to the path so test_nvfp4_linear_fwd_bwd_sqnr can cross-check the API-based
# _Nvfp4LinearSingleDirection (quantize_tensor weight casts) against it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "test"))
from test_quant_cast_gold import _Nvfp4Linear as _Nvfp4LinearRef  # noqa: E402

SHAPES = [(64, 32), (256, 512)]
# (32,1) needs M%128; ((1,32),(32,1)) also needs N%128 (kernel constraints) -- (64,32) fails both.
SHAPES_128 = [(256, 512)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES)
def test_rowwise_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    q, s = quantize_tensor(
        x,
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
        scaling_type=ScalingType.BlockWise1x32,
        orientation=QuantOrientation.NATURAL,
        swizzle_type=SwizzleType.NO_SWIZZLE,
    )
    q_ref, s_ref = mxfp8_f(x)
    # both paths pick the e8m0 scale by floor(log2(amax)) and divide, so the API (Triton) output is
    # byte-identical to the eager golden reference -- exact, not merely within tolerance.
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata differs from gold"
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale differs from gold"
    assert q.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.float8_e8m0fnu
    assert q.shape == (M, N) and s.shape == (M, N // 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES)
def test_mxfp4_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    # mxfp4: fp4 (e2m1) qdata + e8m0 rceil 1x32 block scale (E8M0_RCEIL tells it apart from nvfp4,
    # which shares the float4_e2m1fn_x2 qdata dtype). No Triton kernel yet, so the API dispatches to
    # the gold reference -> byte-identical by construction; the test pins the wiring/shape/dtype.
    q, s = quantize_tensor(
        x,
        qdata_dtype=torch.float4_e2m1fn_x2,
        inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
        scaling_type=ScalingType.BlockWise1x32,
        orientation=QuantOrientation.NATURAL,
        swizzle_type=SwizzleType.NO_SWIZZLE,
    )
    q_ref, s_ref = mxfp4_f(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata differs from gold"
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale differs from gold"
    assert q.dtype == torch.float4_e2m1fn_x2
    assert s.dtype == torch.float8_e8m0fnu
    assert q.shape == (M, N // 2) and s.shape == (M, N // 32)  # two fp4 codes packed per byte


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES_128)
def test_colwise_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    q, s = quantize_tensor(
        x,
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
        scaling_type=ScalingType.BlockWise1x32,
        orientation=QuantOrientation.TRANSPOSED,
        swizzle_type=SwizzleType.NO_SWIZZLE,
    )
    q_ref, s_ref = mxfp8_dim_m_f(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata differs from gold"
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale differs from gold"
    assert q.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.float8_e8m0fnu
    assert q.shape == (N, M) and s.shape == (N, M // 32)  # transposed outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES_128)
def test_both_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    qk, sk, qm, sm = quantize_tensor_bidirectional(
        x,
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
        scaling_type=ScalingType.BlockWise1x32,
        swizzle_type=SwizzleType.NO_SWIZZLE,
    )
    qk_ref, sk_ref, qm_ref, sm_ref = mxfp8_dim_km_f(x)
    for got, ref in [(qk, qk_ref), (sk, sk_ref), (qm, qm_ref), (sm, sm_ref)]:
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8)), "output differs from gold"
    assert qk.dtype == torch.float8_e4m3fn
    assert sk.dtype == torch.float8_e8m0fnu
    assert qm.dtype == torch.float8_e4m3fn
    assert sm.dtype == torch.float8_e8m0fnu
    # rowwise (dim-K) pair, then transposed colwise (dim-M) pair.
    assert qk.shape == (M, N) and sk.shape == (M, N // 32)
    assert qm.shape == (N, M) and sm.shape == (N, M // 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES)  # every SHAPES entry is a multiple of 32 in both dims
def test_32x32_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    q, s = quantize_tensor(
        x,
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
        scaling_type=ScalingType.BlockWise32x32,
        orientation=QuantOrientation.NATURAL,
        swizzle_type=SwizzleType.NO_SWIZZLE,
    )
    q_ref, s_ref = mxfp8_32x32_f(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata differs from gold"
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale differs from gold"
    assert q.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.float8_e8m0fnu
    assert q.shape == (M, N) and s.shape == (M // 32, N // 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES)
def test_rowwise_swizzle_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    q, s = quantize_tensor(
        x,
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
        scaling_type=ScalingType.BlockWise1x32,
        orientation=QuantOrientation.NATURAL,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
    )
    q_ref, s_ref = mxfp8_swizzle_f(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata differs from gold"
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale differs from gold"
    assert q.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.float8_e8m0fnu
    assert q.shape == (M, N)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES_128)
def test_colwise_swizzle_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    q, s = quantize_tensor(
        x,
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
        scaling_type=ScalingType.BlockWise1x32,
        orientation=QuantOrientation.TRANSPOSED,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
    )
    q_ref, s_ref = mxfp8_dim_m_swizzle_f(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata differs from gold"
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale differs from gold"
    assert q.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.float8_e8m0fnu
    assert q.shape == (N, M)  # transposed qdata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES_128)
def test_both_swizzle_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    qk, sk, qm, sm = quantize_tensor_bidirectional(
        x,
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
        scaling_type=ScalingType.BlockWise1x32,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
    )
    qk_ref, sk_ref, qm_ref, sm_ref = mxfp8_dim_km_swizzle_f(x)
    for got, ref in [(qk, qk_ref), (sk, sk_ref), (qm, qm_ref), (sm, sm_ref)]:
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8)), "output differs from gold"
    assert qk.dtype == torch.float8_e4m3fn
    assert sk.dtype == torch.float8_e8m0fnu
    assert qm.dtype == torch.float8_e4m3fn
    assert sm.dtype == torch.float8_e8m0fnu
    assert qk.shape == (M, N)
    assert qm.shape == (N, M)  # transposed qdata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES)  # every SHAPES entry is a multiple of 32 in both dims
def test_32x32_both_scales_natural_qdata_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    qk, sk, sm = quantize_tensor_bidirectional(
        x,
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
        scaling_type=ScalingType.BlockWise32x32,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        skip_transposed_qdata=True,
    )
    qk_ref, sk_ref, sm_ref = mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_f(x)
    for got, ref in [(qk, qk_ref), (sk, sk_ref), (sm, sm_ref)]:
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8)), "output differs from gold"
    assert qk.dtype == torch.float8_e4m3fn  # single (natural) qdata, both scales -- no dim-M qdata
    assert sk.dtype == torch.float8_e8m0fnu
    assert sm.dtype == torch.float8_e8m0fnu
    assert qk.shape == (M, N)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)),
    reason="nvfp4 kernel emits Blackwell-only PTX (cvt.e2m1x2.f32); requires SM100",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES_128)  # nvfp4 kernel needs M%128==0 and N%64==0
def test_nvfp4_matches_gold(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    # nvfp4 is two-level: the per-tensor fp32 outer scale is a global reduction the caller precomputes.
    outer_scale = nvfp4_gs_scale(x)
    q, s = quantize_tensor(
        x,
        qdata_dtype=torch.float4_e2m1fn_x2,
        inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
        scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        orientation=QuantOrientation.NATURAL,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        outer_scale=outer_scale,
    )
    q_ref, s_ref = nvfp4_gs_swizzle_f(x, outer_scale)
    assert q.dtype == torch.float4_e2m1fn_x2
    assert s.dtype == torch.float8_e4m3fn  # nvfp4 inner scale is e4m3 (not e8m0)
    assert q.shape == (M, N // 2)  # two fp4 codes packed per byte
    # The e4m3 inner scale is identical fp32 reduction math -> byte-exact. The fp4 qdata is byte-exact
    # in the common case, but the kernel's hardware cvt.e2m1x2.f32 and the gold's fp32 path can pick
    # different RNE tie-breaks on the coarse fp4 grid, so tolerate a tiny mismatch fraction (matching
    # test/test_quant_cast_triton.py's convention for the fp4 casts).
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale differs from gold"
    qdata_mismatch = (q.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean().item()
    assert qdata_mismatch < 0.01, f"qdata differs from gold in {qdata_mismatch:.3%} of bytes (RNE ties)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES)
def test_nvfp4_per_token_matches_gold_bitwise(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    # per-token nvfp4: the fp32 outer scale is per-row (M, 1) instead of a per-tensor scalar. No
    # Triton kernel yet, so the API maps it straight to the gold reference (nvfp4_gs_f, no swizzle)
    # -> byte-identical by construction. This runs eager on any CUDA device (bit-math fp4 path),
    # unlike the SM100-gated per-tensor kernel test above.
    outer_scale = nvfp4_gs_per_token_scale(x)
    assert outer_scale.shape == (M, 1)
    q, s = quantize_tensor(
        x,
        qdata_dtype=torch.float4_e2m1fn_x2,
        inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
        scaling_type=[ScalingType.BlockWise1x16, ScalingType.RowWise],
        orientation=QuantOrientation.NATURAL,
        swizzle_type=SwizzleType.NO_SWIZZLE,
        outer_scale=outer_scale,
    )
    q_ref, s_ref = nvfp4_gs_f(x, outer_scale)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata differs from gold"
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale differs from gold"
    assert q.dtype == torch.float4_e2m1fn_x2
    assert s.dtype == torch.float8_e4m3fn  # nvfp4 inner scale is e4m3 (not e8m0)
    assert q.shape == (M, N // 2) and s.shape == (M, N // 16)  # 1x16 blocks, two fp4 codes per byte


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES)
def test_nvfp4_dim_m_rht_matches_gold_bitwise(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    # dim-m (TRANSPOSED) swizzled nvfp4 WITH RHT: passing rht_tensor routes to the gold
    # nvfp4_gs_swizzle_dim_m_rht_f (RHT x.t() then nvfp4 along M, transposed (N, M//2) frame) -- the
    # wgrad-operand cast of nvfp4 training. No Triton kernel yet, so the API maps straight to the gold
    # -> byte-identical by construction; runs eager on any CUDA device (bit-math fp4 path).
    sign = torch.tensor([1, -1] * 8, device=x.device, dtype=x.dtype)  # fixed +/-1 RHT sign vector
    rht = hadamard_rht_matrix(sign, x.device, x.dtype)
    # two-level outer scale is over |RHT(x.t())| (the RHT-domain amax), not |x|.
    (x_t_rht,) = hadamard_rht_f(x.t().contiguous(), rht)
    outer_scale = x_t_rht.abs().to(torch.float32).amax() / (F8E4M3_MAX * F4_E2M1_MAX)
    q, s = quantize_tensor(
        x,
        qdata_dtype=torch.float4_e2m1fn_x2,
        inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
        scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        orientation=QuantOrientation.TRANSPOSED,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        outer_scale=outer_scale,
        rht_tensor=rht,
    )
    q_ref, s_ref = nvfp4_gs_swizzle_dim_m_rht_f(x, outer_scale, rht)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata differs from gold"
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale differs from gold"
    assert q.dtype == torch.float4_e2m1fn_x2
    assert s.dtype == torch.float8_e4m3fn  # nvfp4 inner scale is e4m3 (not e8m0)
    assert q.shape == (N, M // 2)  # transposed (N, M//2) frame, two fp4 codes per byte


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_unsupported_combo_raises():
    x = torch.randn(256, 512, device="cuda")
    with pytest.raises(ValueError):  # 32x32 only has a NATURAL kernel, not TRANSPOSED
        quantize_tensor(
            x,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
            scaling_type=ScalingType.BlockWise32x32,
            orientation=QuantOrientation.TRANSPOSED,
        )
    with pytest.raises(ValueError):  # 32x32 bidirectional (full both) is expressible but unwired
        quantize_tensor_bidirectional(
            x,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
            scaling_type=ScalingType.BlockWise32x32,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
    with pytest.raises(ValueError):  # a core ScalingType we don't have a kernel for
        quantize_tensor(
            x,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
            scaling_type=ScalingType.BlockWise1x128,
            orientation=QuantOrientation.NATURAL,
        )
    with pytest.raises(ValueError):  # RowWise granularity is unwired
        quantize_tensor(
            x,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
            scaling_type=ScalingType.RowWise,
            orientation=QuantOrientation.NATURAL,
        )
    with pytest.raises(ValueError):  # 32x32 has no swizzle kernel
        quantize_tensor(
            x,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
            scaling_type=ScalingType.BlockWise32x32,
            orientation=QuantOrientation.NATURAL,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
    with pytest.raises(ValueError):  # skip_transposed_qdata is 32x32-only, not 1x32
        quantize_tensor_bidirectional(
            x,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
            scaling_type=ScalingType.BlockWise1x32,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            skip_transposed_qdata=True,
        )
    with pytest.raises(ValueError):  # skip_transposed_qdata needs the swizzled layout
        quantize_tensor_bidirectional(
            x,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
            scaling_type=ScalingType.BlockWise32x32,
            swizzle_type=SwizzleType.NO_SWIZZLE,
            skip_transposed_qdata=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_grouped_return_arity():
    # offs are 32-aligned (groups are block-aligned; the caller owns any padding) and K a multiple of
    # 128 so the M-groups blocked-scale layout (4-col atoms) is exact.
    x = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([32, 64], dtype=torch.int32, device="cuda")

    mxfp8_kwargs = dict(
        qdata_dtype=torch.float8_e4m3fn,
        inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
        scaling_type=ScalingType.BlockWise1x32,
        swizzle_type=SwizzleType.SWIZZLE_32_4_4,
    )
    # single-orientation grouped cast always returns (qdata, blocked_scale).
    assert len(quantize_tensor_grouped(x, offs, orientation=QuantOrientation.NATURAL, **mxfp8_kwargs)) == 2
    assert len(quantize_tensor_grouped(x, offs, orientation=QuantOrientation.TRANSPOSED, **mxfp8_kwargs)) == 2
    # bidirectional returns both pairs: (q_nat, sb_nat, q_t, sb_t).
    assert len(quantize_tensor_grouped_bidirectional(x, offs, **mxfp8_kwargs)) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_grouped_bidirectional_skip_transposed_qdata_raises():
    # No 32x32 grouped kernel exists, so skip_transposed_qdata is unsupported on the grouped path.
    x = torch.randn(64, 64, device="cuda")
    offs = torch.tensor([32, 64], dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError):
        quantize_tensor_grouped_bidirectional(
            x, offs,
            qdata_dtype=torch.float8_e4m3fn,
            inner_scale_calc=InnerScaleCalc.E8M0_RCEIL,
            scaling_type=ScalingType.BlockWise1x32,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            skip_transposed_qdata=True,
        )

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
# Copied verbatim from test/test_quant_cast_gold.py: this still points at the original gold reference
# casts (quant_cast_gold.recipes), NOT the quantize_tensor_api. Converting these casts to the new API
# is future work.
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


class _Nvfp4LinearSingleDirection(torch.autograd.Function):
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
        x_gs_k = nvfp4_gs_scale(input)  # outer scale over |input|
        x_rht_g_s_m = _rht_outer_scale(input, rht)  # outer scale over |RHT(input.T)|
        # Activation casts through the quantize_tensor API (splitting the fused
        # nvfp4_gs_swizzle_dim_k_dim_m_rht_f into its two orientations). dim-k is plain nvfp4 (no RHT,
        # NATURAL) over |input|; dim-m applies the RHT to input.t() then nvfp4s along M (TRANSPOSED +
        # rht_tensor), scaled by |RHT(input.t())|.
        x_q_k, xs_k = quantize_tensor(
            input,
            qdata_dtype=torch.float4_e2m1fn_x2,
            inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
            scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            orientation=QuantOrientation.NATURAL,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            outer_scale=x_gs_k,
        )
        x_rht_q_m, x_rht_s_m = quantize_tensor(
            input,
            qdata_dtype=torch.float4_e2m1fn_x2,
            inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
            scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            orientation=QuantOrientation.TRANSPOSED,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            outer_scale=x_rht_g_s_m,
            rht_tensor=rht,
        )
        # Weight: row cast (blk K) feeds fwd; transposed row cast (blk N) is the dgrad col operand.
        w_gs = nvfp4_gs_scale(weight)  # |W| == |W.T|, so one outer scale serves both
        # Weight casts through the quantize_tensor API (same nvfp4 config as test_nvfp4_matches_gold:
        # E4M3_NVFP4 two-level, 1x16 blocks, swizzled scale, per-tensor outer scale). The col operand
        # is the API cast of weight.t() (NATURAL orientation, mirroring the gold's _weight_quantize_2d
        # transpose) rather than a TRANSPOSED-orientation call, so both go through the validated path.
        w_q_k, ws_k = quantize_tensor(
            weight,
            qdata_dtype=torch.float4_e2m1fn_x2,
            inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
            scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            orientation=QuantOrientation.NATURAL,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            outer_scale=w_gs,
        )
        w_q_n, w_s_n = quantize_tensor(
            weight,
            qdata_dtype=torch.float4_e2m1fn_x2,
            inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
            scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            orientation=QuantOrientation.TRANSPOSED,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            outer_scale=w_gs,
        )
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
        go_gs_k = nvfp4_gs_scale(grad_output)  # over |grad_output|
        go_rht_g_s_m = _rht_outer_scale(grad_output, rht)  # over |RHT(grad_output.T)|
        key_k, key_m = prng.split(key, 2)
        # grad_output casts through the quantize_tensor API with STOCHASTIC rounding (splitting the
        # fused nvfp4_gs_swizzle_dim_k_dim_m_rht_sr_f into its two orientations, one Philox key each).
        # dim-k is plain SR nvfp4 (no RHT, NATURAL) over |grad_output|; dim-m applies the RHT to
        # grad_output.t() then SR-nvfp4s along M (TRANSPOSED + rht_tensor), scaled by |RHT(dy.t())|.
        go_sr_q_k, gos_k = quantize_tensor(
            grad_output,
            qdata_dtype=torch.float4_e2m1fn_x2,
            inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
            scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            orientation=QuantOrientation.NATURAL,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            outer_scale=go_gs_k,
            rounding_mode=RoundingMode.STOCHASTIC,
            random_key=key_k,
        )
        go_sr_q_m, go_s_m = quantize_tensor(
            grad_output,
            qdata_dtype=torch.float4_e2m1fn_x2,
            inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
            scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            orientation=QuantOrientation.TRANSPOSED,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            outer_scale=go_rht_g_s_m,
            rht_tensor=rht,
            rounding_mode=RoundingMode.STOCHASTIC,
            random_key=key_m,
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


class _Nvfp4LinearBiDirection(torch.autograd.Function):
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
        x_gs_k = nvfp4_gs_scale(input)  # outer scale over |input|
        x_rht_g_s_m = _rht_outer_scale(input, rht)  # outer scale over |RHT(input.T)|
        # Activation cast through the fused dual-orientation quantize_tensor_bidirectional API. dim-k
        # is plain nvfp4 (no RHT, NATURAL) over |input|; dim-m applies the RHT to input.t() then
        # nvfp4s along M, scaled by |RHT(input.t())|. The two orientations use different outer scales
        # (passed as the (dim_k, dim_m) tuple); the RHT is dim-m (second-operand) only, so it goes in
        # as the (None, rht) tuple.
        x_q_k, xs_k, x_rht_q_m, x_rht_s_m = quantize_tensor_bidirectional(
            input,
            qdata_dtype=torch.float4_e2m1fn_x2,
            inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
            scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            outer_scale=(x_gs_k, x_rht_g_s_m),
            rht_tensor=(None, rht),
        )
        # Weight: row cast (blk K) feeds fwd; transposed row cast (blk N) is the dgrad col operand.
        w_gs = nvfp4_gs_scale(weight)  # |W| == |W.T|, so one outer scale serves both
        # Both weight casts in one read via the fused no-RHT quantize_tensor_bidirectional (nvfp4,
        # E4M3_NVFP4, 1x16 blocks, swizzled scale): dim-k (NATURAL) feeds fwd; dim-m (TRANSPOSED, ==
        # nvfp4 of weight.t()) is the dgrad col operand. No RHT, so the same outer scale (w_gs) serves
        # both (|W| == |W.T|) -- passed per-orientation as (w_gs, w_gs). Maps to Nvfp4GsDimKMSwizzleGold.
        w_q_k, ws_k, w_q_n, w_s_n = quantize_tensor_bidirectional(
            weight,
            qdata_dtype=torch.float4_e2m1fn_x2,
            inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
            scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            outer_scale=(w_gs, w_gs),
        )
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
        go_gs_k = nvfp4_gs_scale(grad_output)  # over |grad_output|
        go_rht_g_s_m = _rht_outer_scale(grad_output, rht)  # over |RHT(grad_output.T)|
        # Both grad_output casts in one fused bidirectional SR cast (nvfp4_gs_swizzle_dim_k_dim_m_rht_sr_f):
        # dim-k is plain SR nvfp4 (no RHT) over |grad_output|; dim-m applies the RHT to grad_output.t()
        # then SR-nvfp4s along M, scaled by |RHT(dy.t())|. Per-orientation outer_scale=(dim_k, dim_m);
        # the RHT is dim-m (second-operand) only, so rht_tensor=(None, rht). We hand it the single key --
        # the API splits it into one Philox substream per orientation (== prng.split(key, 2) here).
        go_sr_q_k, gos_k, go_sr_q_m, go_s_m = quantize_tensor_bidirectional(
            grad_output,
            qdata_dtype=torch.float4_e2m1fn_x2,
            inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
            scaling_type=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            outer_scale=(go_gs_k, go_rht_g_s_m),
            rht_tensor=(None, rht),
            rounding_mode=RoundingMode.STOCHASTIC,
            random_key=key,
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
@pytest.mark.parametrize("linear_fn", [_Nvfp4LinearSingleDirection, _Nvfp4LinearBiDirection])
def test_nvfp4_linear_fwd_bwd_sqnr(linear_fn):
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
    linear_fn.apply(xq, wq, rht, key).backward(grad_out)

    out_ref = xr @ wr.t()
    out_q = _Nvfp4LinearSingleDirection.apply(x, w, rht, key)
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

    # Cross-check against the original all-gold reference (test_quant_cast_gold._Nvfp4Linear), which
    # uses the fully-gold casts instead of the quantize_tensor API. Same rht/key/inputs, so the only
    # differences are the API casts that hit a Triton kernel: the weight row/col casts and the
    # activation dim-k (NATURAL) cast use hardware cvt.e2m1x2, which picks different RNE ties than the
    # gold fp32 path on <1% of fp4 codes. The grad_output casts are also API-routed here, but their SR
    # specs map to gold references (nvfp4_gs_swizzle_sr_f / nvfp4_gs_swizzle_dim_m_rht_sr_f), so they
    # add no divergence. wgrad (grad_weight = grad_output_col @ input_col) touches neither the weight
    # cast nor the dim-k activation cast -- only the dim-m activation and grad_output casts, both
    # gold-backed in the API -- so grad_weight stays BIT-IDENTICAL. out and grad_input use the
    # Triton-cast operands, so they match at high SQNR (~38 dB) rather than bitwise.
    xg = x.clone().requires_grad_(True)
    wg = w.clone().requires_grad_(True)
    out_gold = _Nvfp4LinearRef.apply(xg, wg, rht, key)
    out_gold.backward(grad_out)
    assert torch.equal(wq.grad, wg.grad), "grad_weight differs from the all-gold reference"
    sqnr_out_vs_gold = _compute_error(out_gold.float(), out_q.float())
    sqnr_gx_vs_gold = _compute_error(xg.grad.float(), xq.grad.float())
    assert sqnr_out_vs_gold > 30.0, f"output vs gold sqnr={sqnr_out_vs_gold.item():.2f} dB below 30 dB"
    assert sqnr_gx_vs_gold > 30.0, f"grad_input vs gold sqnr={sqnr_gx_vs_gold.item():.2f} dB below 30 dB"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
