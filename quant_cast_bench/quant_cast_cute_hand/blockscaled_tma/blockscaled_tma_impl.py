"""PyTorch-facing implementation for TMA-based block-scaled quantization."""

from dataclasses import dataclass
from functools import cache, partial
from typing import TypeAlias

import torch

from quant_cast_bench.quant_cast_cute.recipes import QuantCastCuteRecipe
from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma.blockscaled_tma_config import (
    RoundingVariant,
    ScaleAlgo,
)
from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma.blockscaled_tma_plan import (
    select_blockscaled_tma_plan,
)
from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma.blockscaled_tma_kernels import (
    _QUANT_ORIENTATION_DIM_K,
    _QUANT_ORIENTATION_DIM_KM,
    _QUANT_ORIENTATION_DIM_M,
    _TORCH_TO_CUTE_QDATA_DTYPE,
    _compile_blockscaled_tma,
)
from quant_cast_bench.quant_cast_cute_hand.utils import _ceil_div
from quant_cast_bench.quant_cast_gold.recipes import (
    Mxfp4Gold,
    Mxfp4DimKMSwizzleGold,
    Mxfp4DimMSwizzleGold,
    Mxfp4SwizzleGold,
    Mxfp832x32SwizzleGold,
    Mxfp8Gold,
    Mxfp8DimKmSwizzleGold,
    Mxfp8DimKmSwizzleSRGold,
    Mxfp8DimMSwizzleGold,
    Mxfp8DimMSwizzleSRGold,
    Mxfp8SwizzleGold,
    Mxfp8SwizzleSRGold,
    Mxfp8SwizzleStatefulSRGold,
    Nvfp4Gs16x16SwizzleGold,
    Nvfp4GsGold,
    Nvfp4GsDimKMSwizzleGold,
    Nvfp4GsDimMSwizzleGold,
    Nvfp4GsSwizzleGold,
)


_INT32_MAX = 2**31 - 1
_CUDA_GRID_X_MAX = _INT32_MAX
_CUDA_GRID_Y_MAX = 2**16 - 1
_INPUT_ALIGNMENT_BYTES = 16

_TwoTensorOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor]
_FourTensorOutput: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
_BlockscaledTmaOutput: TypeAlias = _TwoTensorOutput | _FourTensorOutput


@dataclass(frozen=True)
class _PhiloxLaunch:
    """Flattened Philox arguments for one compiled-kernel invocation.

    RTNE uses none of the fields and leaves every field as ``None``.

    Attributes:
        seed: The two-word user key for ``STATELESS_SR``, or the graph-managed
            device seed tensor for ``STATEFUL_SR_CAPTURE``.
        offset: The graph-managed device offset tensor for
            ``STATEFUL_SR_CAPTURE``.
        seed_scalar: The host generator seed for ``STATEFUL_SR_EAGER``.
        offset_words_scalar: The host generator's word-unit offset for
            ``STATEFUL_SR_EAGER``.
        intragraph_offset_words: The per-launch word-unit offset within a
            captured graph for ``STATEFUL_SR_CAPTURE``.
    """

    seed: torch.Tensor | None = None
    offset: torch.Tensor | None = None
    seed_scalar: int | None = None
    offset_words_scalar: int | None = None
    intragraph_offset_words: int | None = None


def _select_rounding_variant(
    input: torch.Tensor,
    rounding_mode: str,
    key: torch.Tensor | None,
    generator: torch.Generator | None,
) -> RoundingVariant:
    """Validate the public rounding inputs and select one compile-time variant."""
    if rounding_mode == "rtne":
        if key is not None:
            raise ValueError("RTNE rounding does not use a Philox key")
        if generator is not None:
            raise ValueError("RTNE rounding does not use a generator")
        return RoundingVariant.RTNE

    if key is None and generator is None:
        raise ValueError("stochastic rounding requires a Philox key or generator")
    if key is not None and generator is not None:
        raise ValueError("stochastic rounding accepts only one of key or generator")
    if key is not None:
        if not isinstance(key, torch.Tensor):
            raise ValueError("Philox key must be a torch.Tensor")
        if key.device != input.device:
            raise ValueError("input and Philox key must be on the same device")
        if key.dtype != torch.uint64 or key.numel() != 2:
            raise ValueError("Philox key must be uint64[2]")
        return RoundingVariant.STATELESS_SR

    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator")
    generator_device = torch.device(generator.device)
    if generator_device.type != "cuda" or generator_device.index not in (
        None,
        input.device.index,
    ):
        raise ValueError("input and generator must be on the same device")
    return (
        RoundingVariant.STATEFUL_SR_CAPTURE
        if torch.cuda.is_current_stream_capturing()
        else RoundingVariant.STATEFUL_SR_EAGER
    )


def _prepare_philox_launch(
    rounding_variant: RoundingVariant,
    key: torch.Tensor | None,
    generator: torch.Generator | None,
) -> _PhiloxLaunch:
    """Reserve generator state, if needed, immediately before the kernel launch."""
    if rounding_variant == RoundingVariant.RTNE:
        return _PhiloxLaunch()
    if rounding_variant == RoundingVariant.STATELESS_SR:
        assert key is not None
        return _PhiloxLaunch(seed=key.reshape(-1).view(torch.int64))

    assert rounding_variant.is_stateful
    assert generator is not None
    seed_state, offset_state, intragraph_offset_state = generator.philox_state(4)
    if rounding_variant == RoundingVariant.STATEFUL_SR_CAPTURE:
        if not seed_state.is_cuda or not offset_state.is_cuda:
            raise RuntimeError("CUDA graph capture requires device-resident Philox state")
        # These alias graph-managed state that is refreshed before every replay.
        return _PhiloxLaunch(
            seed=seed_state,
            offset=offset_state,
            intragraph_offset_words=int(intragraph_offset_state.item()),
        )

    assert rounding_variant == RoundingVariant.STATEFUL_SR_EAGER
    if seed_state.is_cuda or offset_state.is_cuda:
        raise RuntimeError("eager execution requires host-resident Philox state")
    # Eager generator state lives on the host. Ordinary CUDA scalar launch arguments avoid
    # tiny CPU-to-GPU tensor copies.
    return _PhiloxLaunch(
        seed_scalar=int(seed_state.item()),
        offset_words_scalar=int(offset_state.item()),
    )


@cache
def _cuda_capability(device: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _blockscaled_tma_impl_on_current_device(
    input: torch.Tensor,
    quant_orientation: str,
    key: torch.Tensor | None,
    rounding_mode: str,
    is_square_scaling: bool,
    is_scale_swizzled: bool,
    generator: torch.Generator | None = None,
    qdata_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_algo: ScaleAlgo = ScaleAlgo.RCEIL_E8M0,
    outer_scale_k: torch.Tensor | None = None,
    outer_scale_m: torch.Tensor | None = None,
) -> _BlockscaledTmaOutput:
    if quant_orientation not in ("dim_k", "dim_m", "dim_km"):
        raise ValueError(f"unsupported quant_orientation: {quant_orientation}")
    do_dim_k = quant_orientation != "dim_m"
    do_dim_m = quant_orientation != "dim_k"
    is_nvfp4 = scale_algo == ScaleAlgo.NVFP4_FP8_E4M3

    if not is_scale_swizzled:
        if quant_orientation != "dim_k":
            raise ValueError("compact scales currently support only dim-k output")
        if is_square_scaling:
            raise ValueError("compact scales do not support square scaling")

    if input.dim() != 2:
        raise ValueError(
            f"blockscaled TMA requires a 2D input; got {input.dim()} dimensions"
        )
    if not input.is_contiguous():
        raise ValueError("blockscaled TMA requires a contiguous input")
    if input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            "blockscaled TMA supports only bf16, fp16, and fp32 input"
        )
    if qdata_dtype not in _TORCH_TO_CUTE_QDATA_DTYPE:
        raise ValueError(f"unsupported qdata dtype: {qdata_dtype}")
    if input.data_ptr() % _INPUT_ALIGNMENT_BYTES != 0:
        raise ValueError("blockscaled TMA requires a 16-byte-aligned input")

    if is_nvfp4:
        if qdata_dtype != torch.float4_e2m1fn_x2:
            raise ValueError("NVFP4 scaling requires float4_e2m1fn_x2 qdata")
    else:
        if outer_scale_k is not None or outer_scale_m is not None:
            raise ValueError("RCEIL scaling does not use outer scales")
    rounding_mode = str(getattr(rounding_mode, "value", rounding_mode)).lower()
    if is_nvfp4:
        if rounding_mode != "rtne":
            raise ValueError("NVFP4 TMA supports only RTNE")
    elif rounding_mode not in ("rtne", "stochastic"):
        raise ValueError(f"unsupported rounding_mode: {rounding_mode}")
    rounding_variant = _select_rounding_variant(
        input, rounding_mode, key, generator
    )
    is_stochastic_qdata_rounding = rounding_variant.is_stochastic
    is_packed_fp4_qdata = qdata_dtype == torch.float4_e2m1fn_x2

    if not is_nvfp4 and is_packed_fp4_qdata and is_stochastic_qdata_rounding:
        raise ValueError("packed FP4 qdata currently supports only RTNE")
    if not is_nvfp4 and is_packed_fp4_qdata and is_square_scaling:
        raise ValueError("packed FP4 qdata does not support square scaling")
    if is_square_scaling:
        if quant_orientation != "dim_k":
            raise ValueError("square scaling currently supports only dim-k output")
        if is_stochastic_qdata_rounding:
            raise ValueError("square scaling currently supports only RTNE")
    M, K = input.shape
    if M > _INT32_MAX or K > _INT32_MAX:
        raise ValueError(
            "blockscaled TMA requires each logical dimension to fit in signed int32; "
            f"got shape ({M}, {K})"
        )
    if is_nvfp4:
        k_multiple = 32 if do_dim_k else 16
        if K % k_multiple != 0:
            raise ValueError(f"nvfp4_swizzle_tma requires K % {k_multiple} == 0")
        if is_square_scaling and M % 16 != 0:
            raise ValueError("NVFP4 16x16 scaling requires M % 16 == 0")
        if do_dim_m and M % 32 != 0:
            raise ValueError("nvfp4 dim-M TMA requires M % 32 == 0")

        def validate_outer_scale(
            outer_scale: torch.Tensor | None, name: str
        ) -> None:
            if outer_scale is None:
                raise ValueError(f"{name} outer scale is required")
            if not isinstance(outer_scale, torch.Tensor):
                raise TypeError(f"{name} outer scale must be a torch.Tensor")
            if outer_scale.device != input.device:
                raise ValueError(
                    f"input and {name} outer scale must be on the same device"
                )
            if outer_scale.dtype != torch.float32 or outer_scale.numel() != 1:
                raise ValueError(f"{name} outer scale must be a float32 scalar")

        if do_dim_k:
            validate_outer_scale(outer_scale_k, "dim-K")
        else:
            if outer_scale_k is not None:
                raise ValueError("dim-M does not use a dim-K outer scale")
        if do_dim_m:
            validate_outer_scale(outer_scale_m, "dim-M")
        else:
            if outer_scale_m is not None:
                raise ValueError("dim-K does not use a dim-M outer scale")
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
    scale_dtype = torch.float8_e4m3fn if is_nvfp4 else torch.float8_e8m0fnu

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
    if M == 0 or K == 0:
        output_k = scale_k = None
        if do_dim_k:
            output_k = torch.empty(
                M,
                K // qdata_k_divisor,
                dtype=qdata_storage_dtype,
                device=input.device,
            )
            if is_scale_swizzled:
                scale_k = torch.empty(
                    nrb_k, ncb_k, 32, 16, dtype=torch.uint8, device=input.device
                ).view(scale_dtype)
            else:
                scale_k = torch.empty(
                    M,
                    K // scale_group_size,
                    dtype=torch.uint8,
                    device=input.device,
                ).view(scale_dtype)

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
            ).view(scale_dtype)

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
        if is_scale_swizzled:
            scale_k = torch.empty(
                nrb_k * ncb_k * 32 * 16,
                dtype=torch.uint8,
                device=input.device,
            )
        else:
            scale_k = torch.empty(
                M * (K // scale_group_size),
                dtype=torch.uint8,
                device=input.device,
            )
    outer_scale_k_arg = outer_scale_k.reshape(1) if outer_scale_k is not None else None
    outer_scale_m_arg = outer_scale_m.reshape(1) if outer_scale_m is not None else None

    fn = _compile_blockscaled_tma(
        input.dtype,
        plan.tile_m_size,
        plan.tile_k_size,
        plan.cluster_k,
        plan.needs_boundary_masking,
        quant_orientation_id,
        rounding_variant,
        is_square_scaling,
        is_scale_swizzled,
        qdata_dtype,
        scale_algo,
    )
    # Reserve state only after validation and compilation have succeeded, immediately before launch.
    philox = _prepare_philox_launch(rounding_variant, key, generator)
    fn(
        input,
        output_k,
        scale_k,
        outer_scale_k_arg,
        output_m,
        scale_m,
        outer_scale_m_arg,
        philox.seed,
        philox.offset,
        philox.seed_scalar,
        philox.offset_words_scalar,
        philox.intragraph_offset_words,
        M,
        K,
        plan.grid_m,
        plan.grid_k,
    )

    if do_dim_m:
        scale_m = scale_m.view(nrb_m, ncb_m, 32, 16).view(scale_dtype)
    if do_dim_k:
        if is_scale_swizzled:
            scale_k = scale_k.view(nrb_k, ncb_k, 32, 16)
        else:
            scale_k = scale_k.view(M, K // scale_group_size)
        scale_k = scale_k.view(scale_dtype)
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
    is_scale_swizzled: bool,
    generator: torch.Generator | None = None,
    qdata_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_algo: ScaleAlgo = ScaleAlgo.RCEIL_E8M0,
    outer_scale_k: torch.Tensor | None = None,
    outer_scale_m: torch.Tensor | None = None,
    **kwargs: object,
) -> _BlockscaledTmaOutput:
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

    def launch_on_current_device() -> _BlockscaledTmaOutput:
        return _blockscaled_tma_impl_on_current_device(
            input,
            quant_orientation=quant_orientation,
            key=key,
            rounding_mode=rounding_mode,
            is_square_scaling=is_square_scaling,
            is_scale_swizzled=is_scale_swizzled,
            generator=generator,
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
    **kwargs: object,
) -> _BlockscaledTmaOutput:
    return _blockscaled_tma_impl(
        input,
        quant_orientation=quant_orientation,
        key=key,
        rounding_mode=rounding_mode,
        is_square_scaling=False,
        is_scale_swizzled=True,
        **kwargs,
    )


MXFP8_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleGold, cute_fn=mxfp8_swizzle_v2
)


def mxfp8(input: torch.Tensor, **kwargs: object) -> _TwoTensorOutput:
    return _blockscaled_tma_impl(
        input,
        quant_orientation="dim_k",
        key=None,
        rounding_mode="rtne",
        is_square_scaling=False,
        is_scale_swizzled=False,
        **kwargs,
    )


MXFP8 = QuantCastCuteRecipe.from_gold(Mxfp8Gold, cute_fn=mxfp8)


def mxfp8_32x32_swizzle_v2(
    input: torch.Tensor, **kwargs: object
) -> _TwoTensorOutput:
    return _blockscaled_tma_impl(
        input,
        quant_orientation="dim_k",
        key=None,
        rounding_mode="rtne",
        is_square_scaling=True,
        is_scale_swizzled=True,
        **kwargs,
    )


MXFP8_32X32_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp832x32SwizzleGold, cute_fn=mxfp8_32x32_swizzle_v2
)


def _mxfp8_swizzle_sr_v2(
    input: torch.Tensor,
    key: torch.Tensor,
    **kwargs: object,
) -> _TwoTensorOutput:
    return mxfp8_swizzle_v2(
        input, quant_orientation="dim_k", key=key, rounding_mode="stochastic", **kwargs
    )


MXFP8_SWIZZLE_SR_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleSRGold, cute_fn=_mxfp8_swizzle_sr_v2
)


def mxfp8_swizzle_stateful_sr_v2(
    input: torch.Tensor,
    generator: torch.Generator,
    **kwargs: object,
) -> _TwoTensorOutput:
    return _blockscaled_tma_impl(
        input,
        quant_orientation="dim_k",
        key=None,
        rounding_mode="stochastic",
        is_square_scaling=False,
        is_scale_swizzled=True,
        generator=generator,
        **kwargs,
    )


MXFP8_SWIZZLE_STATEFUL_SR_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleStatefulSRGold,
    cute_fn=mxfp8_swizzle_stateful_sr_v2,
)


MXFP8_DIM_M_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimMSwizzleGold, cute_fn=partial(mxfp8_swizzle_v2, quant_orientation="dim_m")
)


def _mxfp8_dim_m_swizzle_sr_v2(
    input: torch.Tensor,
    key: torch.Tensor,
    **kwargs: object,
) -> _TwoTensorOutput:
    return mxfp8_swizzle_v2(
        input, quant_orientation="dim_m", key=key, rounding_mode="stochastic", **kwargs
    )


MXFP8_DIM_M_SWIZZLE_SR_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimMSwizzleSRGold, cute_fn=_mxfp8_dim_m_swizzle_sr_v2
)


MXFP8_DIM_KM_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimKmSwizzleGold, cute_fn=partial(mxfp8_swizzle_v2, quant_orientation="dim_km")
)


def _mxfp8_dim_km_swizzle_sr_v2(
    input: torch.Tensor,
    key: torch.Tensor,
    **kwargs: object,
) -> _FourTensorOutput:
    return mxfp8_swizzle_v2(
        input, quant_orientation="dim_km", key=key, rounding_mode="stochastic", **kwargs
    )


MXFP8_DIM_KM_SWIZZLE_SR_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimKmSwizzleSRGold, cute_fn=_mxfp8_dim_km_swizzle_sr_v2
)


def mxfp4_swizzle_v2(
    input: torch.Tensor,
    quant_orientation: str = "dim_k",
    **kwargs: object,
) -> _BlockscaledTmaOutput:
    return _blockscaled_tma_impl(
        input,
        quant_orientation=quant_orientation,
        key=None,
        rounding_mode="rtne",
        is_square_scaling=False,
        is_scale_swizzled=True,
        qdata_dtype=torch.float4_e2m1fn_x2,
        **kwargs,
    )


MXFP4_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp4SwizzleGold,
    cute_fn=mxfp4_swizzle_v2,
)


def mxfp4(input: torch.Tensor, **kwargs: object) -> _TwoTensorOutput:
    return _blockscaled_tma_impl(
        input,
        quant_orientation="dim_k",
        key=None,
        rounding_mode="rtne",
        is_square_scaling=False,
        is_scale_swizzled=False,
        qdata_dtype=torch.float4_e2m1fn_x2,
        **kwargs,
    )


MXFP4 = QuantCastCuteRecipe.from_gold(Mxfp4Gold, cute_fn=mxfp4)


def mxfp4_dim_m_swizzle_v2(
    input: torch.Tensor, **kwargs: object
) -> _TwoTensorOutput:
    return mxfp4_swizzle_v2(input, quant_orientation="dim_m", **kwargs)


MXFP4_DIM_M_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp4DimMSwizzleGold,
    cute_fn=mxfp4_dim_m_swizzle_v2,
)


def mxfp4_dim_km_swizzle_v2(
    input: torch.Tensor, **kwargs: object
) -> _FourTensorOutput:
    return mxfp4_swizzle_v2(input, quant_orientation="dim_km", **kwargs)


MXFP4_DIM_KM_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp4DimKMSwizzleGold,
    cute_fn=mxfp4_dim_km_swizzle_v2,
)


def nvfp4_swizzle_tma(
    input: torch.Tensor,
    outer_scale: torch.Tensor,
    outer_scale_m: torch.Tensor | None = None,
    quant_orientation: str = "dim_k",
    **kwargs: object,
) -> _BlockscaledTmaOutput:
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise ValueError(f"unexpected keyword arguments: {unexpected}")
    if quant_orientation not in ("dim_k", "dim_m", "dim_km"):
        raise ValueError(f"unsupported quant_orientation: {quant_orientation}")
    if quant_orientation == "dim_k":
        outer_scale_k_arg, outer_scale_m_arg = outer_scale, outer_scale_m
    elif quant_orientation == "dim_m":
        if outer_scale_m is not None:
            raise ValueError("dim-m takes one outer scale")
        outer_scale_k_arg, outer_scale_m_arg = None, outer_scale
    else:
        outer_scale_k_arg, outer_scale_m_arg = outer_scale, outer_scale_m

    return _blockscaled_tma_impl(
        input,
        quant_orientation=quant_orientation,
        key=None,
        rounding_mode="rtne",
        is_square_scaling=False,
        is_scale_swizzled=True,
        qdata_dtype=torch.float4_e2m1fn_x2,
        scale_algo=ScaleAlgo.NVFP4_FP8_E4M3,
        outer_scale_k=outer_scale_k_arg,
        outer_scale_m=outer_scale_m_arg,
    )


NVFP4_SWIZZLE_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzleGold, cute_fn=nvfp4_swizzle_tma
)


def nvfp4_swizzle_16x16_tma(
    input: torch.Tensor,
    outer_scale: torch.Tensor,
    **kwargs: object,
) -> _TwoTensorOutput:
    return _blockscaled_tma_impl(
        input,
        quant_orientation="dim_k",
        key=None,
        rounding_mode="rtne",
        is_square_scaling=True,
        is_scale_swizzled=True,
        qdata_dtype=torch.float4_e2m1fn_x2,
        scale_algo=ScaleAlgo.NVFP4_FP8_E4M3,
        outer_scale_k=outer_scale,
        **kwargs,
    )


NVFP4_SWIZZLE_16X16_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4Gs16x16SwizzleGold,
    cute_fn=nvfp4_swizzle_16x16_tma,
)


def nvfp4(
    input: torch.Tensor,
    outer_scale: torch.Tensor,
    **kwargs: object,
) -> _TwoTensorOutput:
    return _blockscaled_tma_impl(
        input,
        quant_orientation="dim_k",
        key=None,
        rounding_mode="rtne",
        is_square_scaling=False,
        is_scale_swizzled=False,
        qdata_dtype=torch.float4_e2m1fn_x2,
        scale_algo=ScaleAlgo.NVFP4_FP8_E4M3,
        outer_scale_k=outer_scale,
        **kwargs,
    )


NVFP4 = QuantCastCuteRecipe.from_gold(Nvfp4GsGold, cute_fn=nvfp4)


def nvfp4_dim_m_swizzle_tma(
    input: torch.Tensor,
    outer_scale: torch.Tensor,
    **kwargs: object,
) -> _TwoTensorOutput:
    return nvfp4_swizzle_tma(
        input, outer_scale, quant_orientation="dim_m", **kwargs
    )


NVFP4_DIM_M_SWIZZLE_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsDimMSwizzleGold, cute_fn=nvfp4_dim_m_swizzle_tma
)


def nvfp4_dim_km_swizzle_tma(
    input: torch.Tensor,
    outer_scale_k: torch.Tensor,
    outer_scale_m: torch.Tensor,
    **kwargs: object,
) -> _FourTensorOutput:
    return nvfp4_swizzle_tma(
        input,
        outer_scale_k,
        outer_scale_m=outer_scale_m,
        quant_orientation="dim_km",
        **kwargs,
    )


NVFP4_DIM_KM_SWIZZLE_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsDimKMSwizzleGold, cute_fn=nvfp4_dim_km_swizzle_tma
)
