import pytest

from quant_cast_bench.quant_cast_cute_hand.nvfp4_pipelined.nvfp4_pipelined_plan import (
    select_nvfp4_pipelined_tiles_per_cta,
)


@pytest.mark.parametrize(
    "M,K,stochastic,do_dim_k,expected",
    [
        (2048, 2048, False, True, 2),
        (4096, 4096, False, True, 8),
        (8192, 8192, False, True, 16),
        (8192, 8192, True, True, 16),
        (8192, 8192, True, False, 32),
    ],
)
def test_select_nvfp4_pipelined_tiles_per_cta(
    M,
    K,
    stochastic,
    do_dim_k,
    expected,
):
    assert select_nvfp4_pipelined_tiles_per_cta(
        M,
        K,
        stochastic=stochastic,
        do_dim_k=do_dim_k,
    ) == expected
