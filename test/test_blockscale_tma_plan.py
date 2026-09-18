import pytest
import torch

from quant_cast_bench.quant_cast_cute_hand.blockscale_tma_plan import (
    BlockscaledTmaPlan,
    ScaleAlgo,
    select_blockscaled_tma_plan,
)


@pytest.mark.parametrize(
    "M,K,input_dtype,orientation,stochastic,square,scale_algo,rht,expected",
    [
        (
            2048,
            2048,
            torch.bfloat16,
            "dim_k",
            False,
            False,
            ScaleAlgo.RCEIL_E8M0,
            False,
            BlockscaledTmaPlan(32, 128, 16, False, 64, 16),
        ),
        (
            4096,
            4096,
            torch.bfloat16,
            "dim_k",
            True,
            False,
            ScaleAlgo.RCEIL_E8M0,
            False,
            BlockscaledTmaPlan(128, 128, 1, False, 32, 32),
        ),
        (
            4096,
            4096,
            torch.bfloat16,
            "dim_m",
            False,
            False,
            ScaleAlgo.RCEIL_E8M0,
            False,
            BlockscaledTmaPlan(64, 256, 1, False, 64, 16),
        ),
        (
            4096,
            4096,
            torch.float32,
            "dim_m",
            False,
            False,
            ScaleAlgo.RCEIL_E8M0,
            False,
            BlockscaledTmaPlan(64, 128, 1, False, 64, 32),
        ),
        (
            4096,
            4096,
            torch.bfloat16,
            "dim_km",
            False,
            False,
            ScaleAlgo.RCEIL_E8M0,
            False,
            BlockscaledTmaPlan(64, 128, 1, False, 64, 32),
        ),
        (
            3072,
            3072,
            torch.bfloat16,
            "dim_k",
            False,
            False,
            ScaleAlgo.NVFP4_FP8_E4M3,
            False,
            BlockscaledTmaPlan(128, 64, 1, False, 24, 48),
        ),
        (
            3104,
            3104,
            torch.bfloat16,
            "dim_k",
            False,
            False,
            ScaleAlgo.NVFP4_FP8_E4M3,
            False,
            BlockscaledTmaPlan(128, 64, 1, True, 25, 49),
        ),
        (
            4096,
            4096,
            torch.bfloat16,
            "dim_m",
            False,
            False,
            ScaleAlgo.NVFP4_FP8_E4M3,
            False,
            BlockscaledTmaPlan(64, 128, 1, False, 64, 32),
        ),
        (
            8192,
            8192,
            torch.bfloat16,
            "dim_km",
            True,
            False,
            ScaleAlgo.NVFP4_FP8_E4M3,
            True,
            BlockscaledTmaPlan(64, 128, 2, False, 128, 64),
        ),
    ],
)
def test_select_blockscaled_tma_plan(
    M,
    K,
    input_dtype,
    orientation,
    stochastic,
    square,
    scale_algo,
    rht,
    expected,
):
    assert select_blockscaled_tma_plan(
        M,
        K,
        input_dtype=input_dtype,
        quant_orientation=orientation,
        is_stochastic_qdata_rounding=stochastic,
        is_square_scaling=square,
        scale_algo=scale_algo,
        has_dim_m_rht=rht,
    ) == expected


def test_select_blockscaled_tma_plan_rejects_invalid_orientation():
    with pytest.raises(ValueError, match="unsupported quant_orientation"):
        select_blockscaled_tma_plan(
            128,
            128,
            input_dtype=torch.bfloat16,
            quant_orientation="invalid",
            is_stochastic_qdata_rounding=False,
            is_square_scaling=False,
            scale_algo=ScaleAlgo.RCEIL_E8M0,
            has_dim_m_rht=False,
        )
