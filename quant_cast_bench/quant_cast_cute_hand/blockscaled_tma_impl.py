"""PyTorch-facing implementation for TMA-based block-scaled quantization."""

from functools import cache, partial

import torch

from quant_cast_bench.quant_cast_cute.recipes import QuantCastCuteRecipe
from quant_cast_bench.quant_cast_cute_hand.blockscale_tma_plan import (
    ScaleAlgo,
    select_blockscaled_tma_plan,
)
from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma_kernels import (
    _QUANT_ORIENTATION_DIM_K,
    _QUANT_ORIENTATION_DIM_KM,
    _QUANT_ORIENTATION_DIM_M,
    _TORCH_TO_CUTE_QDATA_DTYPE,
    _compile_blockscaled_tma,
)
from quant_cast_bench.quant_cast_cute_hand.utils import _ceil_div
from quant_cast_bench.quant_cast_gold.recipes import (
    Mxfp4DimKMSwizzleGold,
    Mxfp4DimMSwizzleGold,
    Mxfp4SwizzleGold,
    Mxfp832x32SwizzleGold,
    Mxfp8DimKmSwizzleGold,
    Mxfp8DimKmSwizzleSRGold,
    Mxfp8DimMSwizzleGold,
    Mxfp8DimMSwizzleSRGold,
    Mxfp8SwizzleGold,
    Mxfp8SwizzleSRGold,
    Nvfp4GsDimKMSwizzleGold,
    Nvfp4GsDimMSwizzleGold,
    Nvfp4GsSwizzleGold,
)


_INT32_MAX = 2**31 - 1
_CUDA_GRID_X_MAX = _INT32_MAX
_CUDA_GRID_Y_MAX = 2**16 - 1
_INPUT_ALIGNMENT_BYTES = 16


@cache
def _cuda_capability(device: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _blockscaled_tma_impl_on_current_device(
    input: torch.Tensor,
    quant_orientation: str,
    key: torch.Tensor | None,
    rounding_mode: str,
    is_square_scaling: bool,
    qdata_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_algo: ScaleAlgo = ScaleAlgo.RCEIL_E8M0,
    outer_scale_k: torch.Tensor | None = None,
    outer_scale_m: torch.Tensor | None = None,
):
    if quant_orientation not in ("dim_k", "dim_m", "dim_km"):
        raise ValueError(f"unsupported quant_orientation: {quant_orientation}")
    do_dim_k = quant_orientation != "dim_m"
    do_dim_m = quant_orientation != "dim_k"
    is_nvfp4 = scale_algo == ScaleAlgo.NVFP4_FP8_E4M3

    if input.dim() != 2:
        raise ValueError(
            f"blockscaled TMA requires a 2D input; got {input.dim()} dimensions"
        )
    if not input.is_contiguous():
        raise ValueError("blockscaled TMA requires a contiguous input")
    if is_nvfp4:
        assert input.dtype == torch.bfloat16, "nvfp4_swizzle_tma is bf16-only"
    elif input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "blockscaled TMA supports only bf16, fp16, and fp32 input"
        )
    if qdata_dtype not in _TORCH_TO_CUTE_QDATA_DTYPE:
        raise ValueError(f"unsupported qdata dtype: {qdata_dtype}")
    if input.data_ptr() % _INPUT_ALIGNMENT_BYTES != 0:
        raise ValueError("blockscaled TMA requires a 16-byte-aligned input")

    if is_nvfp4:
        assert qdata_dtype == torch.float4_e2m1fn_x2
        assert not is_square_scaling
    else:
        if outer_scale_k is not None or outer_scale_m is not None:
            raise ValueError("RCEIL scaling does not use outer scales")
    rounding_mode = str(getattr(rounding_mode, "value", rounding_mode)).lower()
    if is_nvfp4:
        if rounding_mode != "rtne":
            raise ValueError("NVFP4 TMA supports only RTNE")
    elif rounding_mode not in ("rtne", "stochastic"):
        raise ValueError(f"unsupported rounding_mode: {rounding_mode}")
    is_stochastic_qdata_rounding = rounding_mode == "stochastic"
    is_packed_fp4_qdata = qdata_dtype == torch.float4_e2m1fn_x2

    if not is_nvfp4 and is_packed_fp4_qdata and is_stochastic_qdata_rounding:
        raise ValueError("packed FP4 qdata currently supports only RTNE")
    if not is_nvfp4 and is_packed_fp4_qdata and is_square_scaling:
        raise ValueError("packed FP4 qdata does not support square scaling")
    if is_square_scaling:
        if quant_orientation != "dim_k":
            raise ValueError("32x32 v2 currently supports only dim-k output")
        if is_stochastic_qdata_rounding:
            raise ValueError("32x32 v2 currently supports only RTNE")
    if is_nvfp4:
        assert key is None, "RTNE rounding does not use a Philox key"
    else:
        if is_stochastic_qdata_rounding:
            if key is None:
                raise ValueError("stochastic rounding requires a Philox key")
            if not isinstance(key, torch.Tensor):
                raise ValueError("Philox key must be a torch.Tensor")
            if key.device != input.device:
                raise ValueError("input and Philox key must be on the same device")
            if key.dtype != torch.uint64 or key.numel() != 2:
                raise ValueError("Philox key must be uint64[2]")
        elif key is not None:
            raise ValueError("RTNE rounding does not use a Philox key")

    M, K = input.shape
    if M > _INT32_MAX or K > _INT32_MAX:
        raise ValueError(
            "blockscaled TMA requires each logical dimension to fit in signed int32; "
            f"got shape ({M}, {K})"
        )
    if is_nvfp4:
        assert M > 0 and K > 0, "nvfp4_swizzle_tma requires non-empty dimensions"
        k_multiple = 32 if do_dim_k else 16
        assert K % k_multiple == 0, (
            f"nvfp4_swizzle_tma requires K % {k_multiple} == 0"
        )
        if do_dim_m:
            assert M % 32 == 0, "nvfp4 dim-M TMA requires M % 32 == 0"

        def validate_outer_scale(
            outer_scale: torch.Tensor | None, name: str
        ) -> None:
            assert outer_scale is not None, f"{name} outer scale is required"
            assert outer_scale.device == input.device, (
                f"input and {name} outer scale must be on the same device"
            )
            assert outer_scale.dtype == torch.float32 and outer_scale.numel() == 1, (
                f"{name} outer scale must be a float32 scalar"
            )

        if do_dim_k:
            validate_outer_scale(outer_scale_k, "dim-K")
        else:
            assert outer_scale_k is None, "dim-M does not use a dim-K outer scale"
        if do_dim_m:
            validate_outer_scale(outer_scale_m, "dim-M")
        else:
            assert outer_scale_m is None, "dim-K does not use a dim-M outer scale"
    else:
        if is_square_scaling and M % 32 != 0:
            raise ValueError("32x32 v2 requires M % 32 == 0")
        if do_dim_m:
            if M % 32 != 0:
                raise ValueError("v2 dim-M requires M % 32 == 0")
            if K % 16 != 0:
                raise ValueError("v2 dim-M requires K % 16 == 0")
            if do_dim_k and K % 32 != 0:
                raise ValueError("v2 dim-K requires K % 32 == 0")
        elif K % 32 != 0:
            raise ValueError("v2 requires K % 32 == 0")

    scale_group_size = 16 if is_nvfp4 else 32
    qdata_k_divisor = 2 if is_packed_fp4_qdata else 1
    qdata_storage_dtype = torch.uint8 if is_packed_fp4_qdata else qdata_dtype

    nrb_k = ncb_k = None
    if do_dim_k:
        nrb_k = _ceil_div(M, 128)
        ncb_k = _ceil_div(K // scale_group_size, 4)

    nrb_m = ncb_m = None
    if do_dim_m:
        nrb_m = _ceil_div(K, 128)
        ncb_m = _ceil_div(M // scale_group_size, 4)

    # CUDA cannot launch a zero-sized grid. Return the correctly oriented empty
    # tensors directly, preserving the padded scale layout in every mode.
    if not is_nvfp4 and (M == 0 or K == 0):
        output_k = scale_k = None
        if do_dim_k:
            output_k = torch.empty(
                M,
                K // qdata_k_divisor,
                dtype=qdata_storage_dtype,
                device=input.device,
            )
            scale_k = torch.empty(
                nrb_k, ncb_k, 32, 16, dtype=torch.uint8, device=input.device
            ).view(torch.float8_e8m0fnu)

        output_m = scale_m = None
        if do_dim_m:
            output_m = torch.empty(
                K,
                M // qdata_k_divisor,
                dtype=qdata_storage_dtype,
                device=input.device,
            )
            scale_m = torch.empty(
                nrb_m, ncb_m, 32, 16, dtype=torch.uint8, device=input.device
            ).view(torch.float8_e8m0fnu)

        if is_packed_fp4_qdata:
            if do_dim_k:
                output_k = output_k.view(torch.float4_e2m1fn_x2)
            if do_dim_m:
                output_m = output_m.view(torch.float4_e2m1fn_x2)
        if quant_orientation == "dim_k":
            return output_k, scale_k
        if quant_orientation == "dim_m":
            return output_m, scale_m
        return output_k, scale_k, output_m, scale_m

    quant_orientation_id = {
        "dim_k": _QUANT_ORIENTATION_DIM_K,
        "dim_m": _QUANT_ORIENTATION_DIM_M,
        "dim_km": _QUANT_ORIENTATION_DIM_KM,
    }[quant_orientation]

    plan = select_blockscaled_tma_plan(
        M,
        K,
        input_dtype=input.dtype,
        quant_orientation=quant_orientation,
        is_stochastic_qdata_rounding=is_stochastic_qdata_rounding,
        is_square_scaling=is_square_scaling,
        scale_algo=scale_algo,
    )
    if plan.grid_k > _CUDA_GRID_X_MAX or plan.grid_m > _CUDA_GRID_Y_MAX:
        raise ValueError(
            "blockscaled TMA launch grid exceeds CUDA limits: "
            f"grid=({plan.grid_k}, {plan.grid_m}, 1), "
            f"maximum=({_CUDA_GRID_X_MAX}, {_CUDA_GRID_Y_MAX}, 65535)"
        )

    output_m = scale_m = None
    if do_dim_m:
        output_m = torch.empty(
            K,
            M // qdata_k_divisor,
            dtype=qdata_storage_dtype,
            device=input.device,
        )
        scale_m = torch.empty(
            nrb_m * ncb_m * 32 * 16,
            dtype=torch.uint8,
            device=input.device,
        )

    output_k = scale_k = None
    if do_dim_k:
        output_k = torch.empty(
            M,
            K // qdata_k_divisor,
            dtype=qdata_storage_dtype,
            device=input.device,
        )
        # Every slot is written by the kernel, so zero-initialization would launch a redundant
        # memset.
        scale_k = torch.empty(
            nrb_k * ncb_k * 32 * 16,
            dtype=torch.uint8,
            device=input.device,
        )
    seed = key.reshape(-1).view(torch.int64) if key is not None else None
    outer_scale_k_arg = outer_scale_k.reshape(1) if outer_scale_k is not None else None
    outer_scale_m_arg = outer_scale_m.reshape(1) if outer_scale_m is not None else None

    fn = _compile_blockscaled_tma(
        input.dtype,
        plan.tile_m_size,
        plan.tile_k_size,
        plan.cluster_k,
        plan.needs_boundary_masking,
        quant_orientation_id,
        is_stochastic_qdata_rounding,
        is_square_scaling,
        qdata_dtype,
        scale_algo,
    )
    fn(
        input,
        output_k,
        scale_k,
        outer_scale_k_arg,
        output_m,
        scale_m,
        outer_scale_m_arg,
        seed,
        M,
        K,
    )

    scale_dtype = torch.float8_e4m3fn if is_nvfp4 else torch.float8_e8m0fnu
    if do_dim_m:
        scale_m = scale_m.view(nrb_m, ncb_m, 32, 16).view(scale_dtype)
    if do_dim_k:
        scale_k = scale_k.view(nrb_k, ncb_k, 32, 16).view(scale_dtype)
    if is_packed_fp4_qdata:
        if do_dim_k:
            output_k = output_k.view(torch.float4_e2m1fn_x2)
        if do_dim_m:
            output_m = output_m.view(torch.float4_e2m1fn_x2)

    if quant_orientation == "dim_k":
        return output_k, scale_k
    if quant_orientation == "dim_m":
        return output_m, scale_m
    return output_k, scale_k, output_m, scale_m


def _blockscaled_tma_impl(
    input: torch.Tensor,
    quant_orientation: str,
    key: torch.Tensor | None,
    rounding_mode: str,
    is_square_scaling: bool,
    qdata_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_algo: ScaleAlgo = ScaleAlgo.RCEIL_E8M0,
    outer_scale_k: torch.Tensor | None = None,
    outer_scale_m: torch.Tensor | None = None,
    **kwargs,
):
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise ValueError(f"unexpected keyword arguments: {unexpected}")
    if not isinstance(input, torch.Tensor):
        raise ValueError("blockscaled TMA input must be a torch.Tensor")
    if input.device.type != "cuda":
        raise ValueError("blockscaled TMA requires a CUDA input")

    device = input.get_device()
    capability = _cuda_capability(device)
    if capability < (10, 0):
        raise RuntimeError(
            "blockscaled TMA requires CUDA capability 10.0 or newer; "
            f"device {input.device} has capability {capability[0]}.{capability[1]}"
        )

    def launch_on_current_device():
        return _blockscaled_tma_impl_on_current_device(
            input,
            quant_orientation=quant_orientation,
            key=key,
            rounding_mode=rounding_mode,
            is_square_scaling=is_square_scaling,
            qdata_dtype=qdata_dtype,
            scale_algo=scale_algo,
            outer_scale_k=outer_scale_k,
            outer_scale_m=outer_scale_m,
        )

    if device == torch.cuda.current_device():
        return launch_on_current_device()
    with torch.cuda.device(device):
        return launch_on_current_device()


def mxfp8_swizzle_v2(
    input: torch.Tensor,
    quant_orientation: str = "dim_k",
    key: torch.Tensor | None = None,
    rounding_mode: str = "rtne",
    **kwargs,
):
    return _blockscaled_tma_impl(
        input,
        quant_orientation=quant_orientation,
        key=key,
        rounding_mode=rounding_mode,
        is_square_scaling=False,
        **kwargs,
    )


MXFP8_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleGold, cute_fn=mxfp8_swizzle_v2
)


def mxfp8_32x32_swizzle_v2(input: torch.Tensor, **kwargs):
    return _blockscaled_tma_impl(
        input,
        quant_orientation="dim_k",
        key=None,
        rounding_mode="rtne",
        is_square_scaling=True,
        **kwargs,
    )


MXFP8_32X32_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp832x32SwizzleGold, cute_fn=mxfp8_32x32_swizzle_v2
)


def _mxfp8_swizzle_sr_v2(input, key, **kwargs):
    return mxfp8_swizzle_v2(
        input, quant_orientation="dim_k", key=key, rounding_mode="stochastic", **kwargs
    )


MXFP8_SWIZZLE_SR_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleSRGold, cute_fn=_mxfp8_swizzle_sr_v2
)


MXFP8_DIM_M_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimMSwizzleGold, cute_fn=partial(mxfp8_swizzle_v2, quant_orientation="dim_m")
)


def _mxfp8_dim_m_swizzle_sr_v2(input, key, **kwargs):
    return mxfp8_swizzle_v2(
        input, quant_orientation="dim_m", key=key, rounding_mode="stochastic", **kwargs
    )


MXFP8_DIM_M_SWIZZLE_SR_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimMSwizzleSRGold, cute_fn=_mxfp8_dim_m_swizzle_sr_v2
)


MXFP8_DIM_KM_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimKmSwizzleGold, cute_fn=partial(mxfp8_swizzle_v2, quant_orientation="dim_km")
)


def _mxfp8_dim_km_swizzle_sr_v2(input, key, **kwargs):
    return mxfp8_swizzle_v2(
        input, quant_orientation="dim_km", key=key, rounding_mode="stochastic", **kwargs
    )


MXFP8_DIM_KM_SWIZZLE_SR_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimKmSwizzleSRGold, cute_fn=_mxfp8_dim_km_swizzle_sr_v2
)


def mxfp4_swizzle_v2(
    input: torch.Tensor,
    quant_orientation: str = "dim_k",
    **kwargs,
):
    return _blockscaled_tma_impl(
        input,
        quant_orientation=quant_orientation,
        key=None,
        rounding_mode="rtne",
        is_square_scaling=False,
        qdata_dtype=torch.float4_e2m1fn_x2,
        **kwargs,
    )


MXFP4_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp4SwizzleGold,
    cute_fn=mxfp4_swizzle_v2,
)


def mxfp4_dim_m_swizzle_v2(input: torch.Tensor, **kwargs):
    return mxfp4_swizzle_v2(input, quant_orientation="dim_m", **kwargs)


MXFP4_DIM_M_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp4DimMSwizzleGold,
    cute_fn=mxfp4_dim_m_swizzle_v2,
)


def mxfp4_dim_km_swizzle_v2(input: torch.Tensor, **kwargs):
    return mxfp4_swizzle_v2(input, quant_orientation="dim_km", **kwargs)


MXFP4_DIM_KM_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp4DimKMSwizzleGold,
    cute_fn=mxfp4_dim_km_swizzle_v2,
)


def nvfp4_swizzle_tma(
    input: torch.Tensor,
    outer_scale: torch.Tensor,
    outer_scale_m: torch.Tensor | None = None,
    mode: str = "dim_k",
    **kwargs,
):
    assert not kwargs, f"unexpected keyword arguments: {', '.join(sorted(kwargs))}"
    assert mode in ("dim_k", "dim_m", "dim_km"), f"unsupported mode: {mode}"
    if mode == "dim_k":
        outer_scale_k_arg, outer_scale_m_arg = outer_scale, outer_scale_m
    elif mode == "dim_m":
        assert outer_scale_m is None, "dim-m takes one outer scale"
        outer_scale_k_arg, outer_scale_m_arg = None, outer_scale
    else:
        outer_scale_k_arg, outer_scale_m_arg = outer_scale, outer_scale_m

    return _blockscaled_tma_impl(
        input,
        quant_orientation=mode,
        key=None,
        rounding_mode="rtne",
        is_square_scaling=False,
        qdata_dtype=torch.float4_e2m1fn_x2,
        scale_algo=ScaleAlgo.NVFP4_FP8_E4M3,
        outer_scale_k=outer_scale_k_arg,
        outer_scale_m=outer_scale_m_arg,
    )


NVFP4_SWIZZLE_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzleGold, cute_fn=nvfp4_swizzle_tma
)


def nvfp4_dim_m_swizzle_tma(input, outer_scale, **kwargs):
    return nvfp4_swizzle_tma(
        input, outer_scale, mode="dim_m", **kwargs
    )


NVFP4_DIM_M_SWIZZLE_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsDimMSwizzleGold, cute_fn=nvfp4_dim_m_swizzle_tma
)


def nvfp4_dim_km_swizzle_tma(input, outer_scale_k, outer_scale_m, **kwargs):
    return nvfp4_swizzle_tma(
        input,
        outer_scale_k,
        outer_scale_m=outer_scale_m,
        mode="dim_km",
        **kwargs,
    )


NVFP4_DIM_KM_SWIZZLE_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsDimKMSwizzleGold, cute_fn=nvfp4_dim_km_swizzle_tma
)
