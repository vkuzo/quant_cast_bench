"""Launch policy for TMA-based block-scaled quantization kernels."""

from enum import IntEnum
from typing import NamedTuple

import torch


class ScaleAlgo(IntEnum):
    """Compile-time algorithm used to derive and encode each block scale."""

    RCEIL_E8M0 = 0
    NVFP4_FP8_E4M3 = 1


_DIM_K_TILE_M_SIZE_128 = 128
_DIM_K_MAX_TILE_K_SIZE_128 = 128
_DIM_M_KM_SMALL_TILE_32_128 = (32, 128)
_DIM_M_LARGE_TILE_64_256 = (64, 256)
_DIM_M_FLOAT32_LARGE_TILE_64_128 = (64, 128)
_DIM_KM_LARGE_TILE_64_128 = (64, 128)


class BlockscaledTmaPlan(NamedTuple):
    """Compile-time launch parameters and runtime grid for one invocation."""

    tile_m_size: int
    tile_k_size: int
    cluster_k: int
    needs_boundary_masking: bool
    grid_m: int
    grid_k: int


def _ceil_div(num: int, den: int) -> int:
    return (num + den - 1) // den


def select_blockscaled_tma_plan(
    M: int,
    K: int,
    *,
    input_dtype: torch.dtype,
    quant_orientation: str,
    is_stochastic_qdata_rounding: bool,
    is_square_scaling: bool,
    scale_algo: ScaleAlgo,
) -> BlockscaledTmaPlan:
    """Select the existing B200-tuned tile, cluster, masking, and grid policy."""
    if quant_orientation not in ("dim_k", "dim_m", "dim_km"):
        raise ValueError(f"unsupported quant_orientation: {quant_orientation}")

    do_dim_k = quant_orientation != "dim_m"
    do_dim_m = quant_orientation != "dim_k"
    is_nvfp4 = scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
    scale_group_size = 16 if is_nvfp4 else 32

    nrb_k = ncb_k = None
    if do_dim_k:
        nrb_k = _ceil_div(M, 128)
        ncb_k = _ceil_div(K // scale_group_size, 4)

    nrb_m = ncb_m = None
    if do_dim_m:
        nrb_m = _ceil_div(K, 128)
        ncb_m = _ceil_div(M // scale_group_size, 4)

    if is_nvfp4 and quant_orientation == "dim_k":
        assert ncb_k is not None
        tile_m_size, tile_k_size = (
            (32, 128)
            if M * K <= 2048 * 2048 and ncb_k % 2 == 0
            else (
                128,
                128 if M * K >= 4096 * 4096 and ncb_k % 2 == 0 else 64,
            )
        )
        cluster_k = 1
        needs_boundary_masking = M % 128 != 0 or K % 128 != 0
    elif is_nvfp4:
        tile_m_size, tile_k_size = (
            _DIM_M_KM_SMALL_TILE_32_128
            if M * K <= 2048 * 2048
            else _DIM_KM_LARGE_TILE_64_128
        )
        cluster_k = 1
        needs_boundary_masking = M % 128 != 0 or K % 128 != 0
    elif quant_orientation == "dim_k":
        assert nrb_k is not None
        assert ncb_k is not None
        # First choose the original adaptive K width. For small problems that would use K=64,
        # rotate the same-size 128x64 tile to 32x128: it keeps 128 one-group threads but gives TMA
        # contiguous rows and exposes more M-parallel CTAs.
        num_128_tiles = _ceil_div(M, _DIM_K_TILE_M_SIZE_128) * _ceil_div(
            K, _DIM_K_MAX_TILE_K_SIZE_128
        )
        if num_128_tiles <= 64:
            tile_k_size = 32
        elif num_128_tiles <= 512:
            tile_k_size = 64
        else:
            tile_k_size = 128

        # Do not compute padding merely to reach the selected width for very narrow matrices.
        if K <= 32:
            tile_k_size = 32
        elif K <= 64:
            tile_k_size = min(tile_k_size, 64)
        if M * K <= 2048 * 2048 and tile_k_size >= 64:
            tile_m_size, tile_k_size = 32, 128
        else:
            tile_m_size = 128
        grid_k = _ceil_div(K, tile_k_size)
        # RTNE benefits from K-oriented clustering and its locality. Philox supplies enough
        # arithmetic latency hiding that independent CTAs are faster than the clustered schedule.
        cluster_k = (
            # Square scaling has enough independent CTAs at small/medium sizes that clustering only
            # constrains scheduling; large 16-bit shapes retain v2's K-locality-oriented clusters.
            # FP32's larger shared-memory tiles reduce residency enough that clustering loses.
            1
            if input_dtype == torch.float32
            or is_stochastic_qdata_rounding
            or (is_square_scaling and M * K <= 4096 * 4096)
            else next(
                c for c in (16, 8, 4, 2, 1) if c <= ncb_k and grid_k % c == 0
            )
        )
        needs_boundary_masking = M != nrb_k * 128 or K != ncb_k * 128
    elif quant_orientation == "dim_m":
        assert nrb_m is not None
        assert ncb_m is not None
        # TODO: For MXFP4 dim-M above 2048^2, try a 64x128 tile with an ordinary
        # non-cluster launch. On B200 this improved geomean throughput by 2.20% over
        # 81 unpadded shapes and 3.08% over nine +32 padded shapes (up to 6.84%);
        # retain the current 32x128 clustered path at 2048^2, where the change lost 3.1%.
        tile_m_size, tile_k_size = (
            _DIM_M_KM_SMALL_TILE_32_128
            if M * K <= 2048 * 2048
            else (
                _DIM_M_FLOAT32_LARGE_TILE_64_128
                if input_dtype == torch.float32
                else _DIM_M_LARGE_TILE_64_256
            )
        )
        padded_M = ncb_m * 128
        padded_K = _ceil_div(K, tile_k_size) * tile_k_size
        grid_m = padded_M // tile_m_size
        grid_k = padded_K // tile_k_size
        cluster_k = (
            next(
                c for c in (16, 8, 4, 2, 1) if c <= grid_k and grid_k % c == 0
            )
            if grid_m * grid_k <= 512
            else 1
        )
        needs_boundary_masking = M != ncb_m * 128 or K != nrb_m * 128
    else:
        assert nrb_m is not None
        assert ncb_m is not None
        tile_m_size, tile_k_size = (
            _DIM_M_KM_SMALL_TILE_32_128
            if M * K <= 2048 * 2048
            else _DIM_KM_LARGE_TILE_64_128
        )
        cluster_k = 1
        needs_boundary_masking = M != ncb_m * 128 or K != nrb_m * 128

    grid_k = _ceil_div(K, tile_k_size)
    if is_nvfp4 and quant_orientation == "dim_m":
        grid_m = _ceil_div(M, 64) * 64 // tile_m_size
    else:
        grid_m = _ceil_div(M, 128) * (128 // tile_m_size)

    return BlockscaledTmaPlan(
        tile_m_size,
        tile_k_size,
        cluster_k,
        needs_boundary_masking,
        grid_m,
        grid_k,
    )
