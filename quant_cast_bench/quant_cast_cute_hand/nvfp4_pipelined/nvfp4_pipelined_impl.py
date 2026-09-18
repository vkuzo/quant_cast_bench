"""PyTorch-facing implementation for persistent TMA/UMMA NVFP4 RHT quantization."""

import torch

from quant_cast_bench.quant_cast_cute.recipes import QuantCastCuteRecipe
from quant_cast_bench.quant_cast_cute_hand.nvfp4_pipelined.nvfp4_pipelined_kernels import (
    _compile_nvfp4_rht_pipelined,
)
from quant_cast_bench.quant_cast_cute_hand.nvfp4_pipelined.nvfp4_pipelined_plan import (
    select_nvfp4_pipelined_tiles_per_cta,
)
from quant_cast_bench.quant_cast_cute_hand.utils import (
    _allocate_nvfp4_swizzle_outputs,
    _validate_nvfp4_swizzle_inputs,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    Nvfp4GsDimMSwizzleRHTSRGold,
    Nvfp4GsSwizzle_DimK_DimMRHT_Gold,
    Nvfp4GsSwizzle_DimKSR_DimMRHTSR_Gold,
    Nvfp4GsSwizzleDimMRHTGold,
)


def _nvfp4_rht_pipelined_on_current_device(
    input,
    outer_scale_m,
    rht_sign,
    key=None,
    *,
    outer_scale_k=None,
    stochastic,
    do_dim_k,
):
    M, N = _validate_nvfp4_swizzle_inputs(
        input, outer_scale_m, "nvfp4 RHT pipelined", 32
    )
    assert M % 128 == 0 and N % 128 == 0, (
        "nvfp4 RHT pipelined requires M % 128 == 0 and K % 128 == 0"
    )
    assert outer_scale_m.device == input.device
    assert outer_scale_m.dtype == torch.float32 and outer_scale_m.numel() == 1
    if do_dim_k:
        assert outer_scale_k is not None
        assert outer_scale_k.device == input.device
        assert outer_scale_k.dtype == torch.float32
        assert outer_scale_k.numel() == 1
    else:
        assert outer_scale_k is None
    assert rht_sign.shape == (16,)
    assert rht_sign.dtype == torch.bfloat16 and rht_sign.device == input.device
    assert rht_sign.is_contiguous()
    if stochastic:
        assert key is not None, "stochastic rounding requires a key"
        assert key.device == input.device
        assert key.dtype == torch.uint64 and key.numel() == 2
    else:
        assert key is None

    if do_dim_k:
        output_k, scale_k, nrb_k, ncb_k = (
            _allocate_nvfp4_swizzle_outputs(input, M, N)
        )
    else:
        output_k = None
        scale_k = None
        nrb_k = None
        ncb_k = None
    output_m, scale_m, nrb_m, ncb_m = _allocate_nvfp4_swizzle_outputs(
        input, N, M
    )
    tiles_per_cta = select_nvfp4_pipelined_tiles_per_cta(
        M,
        N,
        stochastic=stochastic,
        do_dim_k=do_dim_k,
    )
    seed = key.reshape(-1).view(torch.int64) if stochastic else None
    fn = _compile_nvfp4_rht_pipelined(
        stochastic,
        do_dim_k,
        tiles_per_cta,
    )
    fn(
        input,
        output_k,
        scale_k,
        outer_scale_k.reshape(1) if do_dim_k else None,
        output_m,
        scale_m,
        outer_scale_m.reshape(1),
        rht_sign,
        seed,
        M,
        N,
    )
    outputs_m = (
        output_m.view(torch.float4_e2m1fn_x2),
        scale_m.view(nrb_m, ncb_m, 32, 16).view(torch.float8_e4m3fn),
    )
    if not do_dim_k:
        return outputs_m
    return (
        output_k.view(torch.float4_e2m1fn_x2),
        scale_k.view(nrb_k, ncb_k, 32, 16).view(torch.float8_e4m3fn),
        *outputs_m,
    )


def _nvfp4_rht_pipelined(
    input,
    outer_scale_m,
    rht_sign,
    key=None,
    *,
    outer_scale_k=None,
    stochastic,
    do_dim_k,
):
    device = input.get_device()

    def launch_on_current_device():
        return _nvfp4_rht_pipelined_on_current_device(
            input,
            outer_scale_m,
            rht_sign,
            key,
            outer_scale_k=outer_scale_k,
            stochastic=stochastic,
            do_dim_k=do_dim_k,
        )

    if device == torch.cuda.current_device():
        return launch_on_current_device()
    with torch.cuda.device(device):
        return launch_on_current_device()


def nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
    input, outer_scale_k, outer_scale_m, rht_sign, **kwargs
):
    return _nvfp4_rht_pipelined(
        input,
        outer_scale_m,
        rht_sign,
        outer_scale_k=outer_scale_k,
        stochastic=False,
        do_dim_k=True,
    )


NVFP4_SWIZZLE_DIM_K_DIM_M_RHT_PIPELINED = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzle_DimK_DimMRHT_Gold,
    cute_fn=nvfp4_swizzle_dim_k_dim_m_rht_pipelined,
)


def nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined(
    input,
    outer_scale_k,
    outer_scale_m,
    rht_sign,
    key,
    **kwargs,
):
    return _nvfp4_rht_pipelined(
        input,
        outer_scale_m,
        rht_sign,
        key,
        outer_scale_k=outer_scale_k,
        stochastic=True,
        do_dim_k=True,
    )


NVFP4_SWIZZLE_DIM_K_SR_DIM_M_RHT_SR_PIPELINED = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzle_DimKSR_DimMRHTSR_Gold,
    cute_fn=nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined,
)


def nvfp4_dim_m_rht_swizzle_pipelined(
    input,
    outer_scale,
    rht_sign,
    **kwargs,
):
    return _nvfp4_rht_pipelined(
        input,
        outer_scale,
        rht_sign,
        stochastic=False,
        do_dim_k=False,
    )


NVFP4_DIM_M_RHT_SWIZZLE_PIPELINED = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzleDimMRHTGold,
    cute_fn=nvfp4_dim_m_rht_swizzle_pipelined,
)


def nvfp4_dim_m_swizzle_rht_sr_pipelined(
    input,
    outer_scale,
    rht_sign,
    key,
    **kwargs,
):
    return _nvfp4_rht_pipelined(
        input,
        outer_scale,
        rht_sign,
        key,
        stochastic=True,
        do_dim_k=False,
    )


NVFP4_DIM_M_SWIZZLE_RHT_SR_PIPELINED = QuantCastCuteRecipe.from_gold(
    Nvfp4GsDimMSwizzleRHTSRGold,
    cute_fn=nvfp4_dim_m_swizzle_rht_sr_pipelined,
)
