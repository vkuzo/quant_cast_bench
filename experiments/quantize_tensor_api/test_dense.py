import os
import sys

import pytest
import torch

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
    mxfp8_32x32_f,
    mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_f,
    mxfp8_dim_km_f,
    mxfp8_dim_km_swizzle_f,
    mxfp8_dim_m_f,
    mxfp8_dim_m_swizzle_f,
    mxfp8_f,
    mxfp8_swizzle_f,
    nvfp4_gs_scale,
    nvfp4_gs_swizzle_f,
)

SHAPES = [(64, 32), (256, 512)]
# (32,1) needs M%128; ((1,32),(32,1)) also needs N%128 (kernel constraints) -- (64,32) fails both.
SHAPES_128 = [(256, 512)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("M,N", SHAPES)
def test_rowwise_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    q, s = quantize_tensor(x, swizzle_type=SwizzleType.NO_SWIZZLE)  # BlockWise1x32, NATURAL
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
@pytest.mark.parametrize("M,N", SHAPES_128)
def test_colwise_matches_gold_bitwise(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    q, s = quantize_tensor(
        x,
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
        scaling_type=ScalingType.BlockWise1x16,
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
def test_nvfp4_input_guards():
    x = torch.randn(256, 512, device="cuda")
    with pytest.raises(AssertionError):  # fp4 qdata needs the NVFP4 inner scale calc
        quantize_tensor(x, qdata_dtype=torch.float4_e2m1fn_x2, outer_scale=nvfp4_gs_scale(x))
    with pytest.raises(AssertionError):  # nvfp4 requires a precomputed outer_scale
        quantize_tensor(
            x,
            qdata_dtype=torch.float4_e2m1fn_x2,
            inner_scale_calc=InnerScaleCalc.E4M3_NVFP4,
            scaling_type=ScalingType.BlockWise1x16,
        )
    with pytest.raises(AssertionError):  # outer_scale is nvfp4-only, not valid for mxfp8
        quantize_tensor(x, outer_scale=nvfp4_gs_scale(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_input_guards():
    with pytest.raises(AssertionError):  # not 2D or 3D (3D is the per-expert path)
        quantize_tensor(torch.randn(2, 8, 8, 32, device="cuda"))
    with pytest.raises(AssertionError):  # N not a multiple of 32
        quantize_tensor(torch.randn(64, 48, device="cuda"))
    with pytest.raises(AssertionError):  # not contiguous
        quantize_tensor(torch.randn(64, 64, device="cuda").t())
    with pytest.raises(NotImplementedError):  # stochastic rounding not implemented yet
        quantize_tensor(torch.randn(64, 64, device="cuda"), rounding_mode=RoundingMode.STOCHASTIC)
    with pytest.raises(NotImplementedError):  # random_key (SR) not implemented yet
        quantize_tensor(torch.randn(64, 64, device="cuda"), random_key=torch.randint(0, 2**31, (1,), device="cuda"))
    with pytest.raises(AssertionError):  # only float8_e4m3fn qdata is supported today
        quantize_tensor(torch.randn(64, 64, device="cuda"), qdata_dtype=torch.float8_e5m2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_unsupported_combo_raises():
    x = torch.randn(256, 512, device="cuda")
    with pytest.raises(ValueError):  # 32x32 only has a NATURAL kernel, not TRANSPOSED
        quantize_tensor(x, scaling_type=ScalingType.BlockWise32x32, orientation=QuantOrientation.TRANSPOSED)
    with pytest.raises(ValueError):  # 32x32 bidirectional (full both) is expressible but unwired
        quantize_tensor_bidirectional(x, scaling_type=ScalingType.BlockWise32x32)
    with pytest.raises(ValueError):  # a core ScalingType we don't have a kernel for
        quantize_tensor(x, scaling_type=ScalingType.BlockWise1x128, orientation=QuantOrientation.NATURAL)
    with pytest.raises(ValueError):  # RowWise granularity is unwired
        quantize_tensor(x, scaling_type=ScalingType.RowWise, orientation=QuantOrientation.NATURAL)
    with pytest.raises(ValueError):  # 32x32 has no swizzle kernel
        quantize_tensor(
            x,
            scaling_type=ScalingType.BlockWise32x32,
            orientation=QuantOrientation.NATURAL,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
    with pytest.raises(ValueError):  # skip_transposed_qdata is 32x32-only, not 1x32
        quantize_tensor_bidirectional(
            x,
            scaling_type=ScalingType.BlockWise1x32,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            skip_transposed_qdata=True,
        )
    with pytest.raises(ValueError):  # skip_transposed_qdata needs the swizzled layout
        quantize_tensor_bidirectional(
            x,
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

    # single-orientation grouped cast always returns (qdata, blocked_scale).
    assert len(quantize_tensor_grouped(x, offs, orientation=QuantOrientation.NATURAL)) == 2
    assert len(quantize_tensor_grouped(x, offs, orientation=QuantOrientation.TRANSPOSED)) == 2
    # bidirectional returns both pairs: (q_nat, sb_nat, q_t, sb_t).
    assert len(quantize_tensor_grouped_bidirectional(x, offs)) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_grouped_bidirectional_skip_transposed_qdata_raises():
    # No 32x32 grouped kernel exists, so skip_transposed_qdata is unsupported on the grouped path.
    x = torch.randn(64, 64, device="cuda")
    offs = torch.tensor([32, 64], dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError):
        quantize_tensor_grouped_bidirectional(x, offs, skip_transposed_qdata=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
