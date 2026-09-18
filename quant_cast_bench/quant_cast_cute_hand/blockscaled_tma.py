"""TMA-based block-scaled quantization kernels and recipe definitions."""

from enum import IntEnum
from functools import cache, partial
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache
from torch._vendor.quack.cache import EXTRA_SOURCE_DIRS

from quant_cast_bench.quant_cast_cute.recipes import (
    QuantCastCuteRecipe,
    _nvfp4_scale_e4m3,
    _philox_4x32,
)
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
    Nvfp4GsDimMSwizzleRHTSRGold,
    Nvfp4GsSwizzle_DimK_DimMRHT_Gold,
    Nvfp4GsSwizzle_DimKSR_DimMRHTSR_Gold,
    Nvfp4GsSwizzleDimMRHTGold,
    Nvfp4GsSwizzleGold,
)
from quant_cast_bench.quant_cast_cute_hand.utils import (
    _ceil_div,
    _cvt_rn_satfinite_e2m1x2_f32_x4,
    _cvt_rs_satfinite_e2m1x4_f32_x8,
    _cvt_rs_satfinite_e4m3x4_f32,
    _e8m0,
    _e8m0_with_max_pos,
    _nvfp4_load_philox_key,
    _nvfp4_rht_fwht_x16,
    _store_swizzled_scale_groups_as_uint,
)

# Include this prototype's Python sources in QuACK's persistent-cache fingerprint. This must
# happen before the process's first jit_cache lookup, when that fingerprint is memoized.
_BLOCKSCALED_TMA_SOURCE_DIR = Path(__file__).resolve().parent
if _BLOCKSCALED_TMA_SOURCE_DIR not in EXTRA_SOURCE_DIRS:
    EXTRA_SOURCE_DIRS.append(_BLOCKSCALED_TMA_SOURCE_DIR)


_TORCH_TO_CUTE_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}

_TORCH_TO_CUTE_QDATA_DTYPE = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float4_e2m1fn_x2: cutlass.Float4E2M1FN,
}


_QUANT_ORIENTATION_DIM_K = 0
_QUANT_ORIENTATION_DIM_M = 1
_QUANT_ORIENTATION_DIM_KM = 2

_DIM_K_TILE_M_SIZE_128 = 128
_DIM_K_MAX_TILE_K_SIZE_128 = 128
_MIN_CTA_WARPS_4 = 4
_MIN_CTA_THREADS_128 = _MIN_CTA_WARPS_4 * 32

_DIM_M_KM_SMALL_TILE_32_128 = (32, 128)
_DIM_M_LARGE_TILE_64_256 = (64, 256)
_DIM_M_FLOAT32_LARGE_TILE_64_128 = (64, 128)
_DIM_KM_LARGE_TILE_64_128 = (64, 128)

_INT32_MAX = 2**31 - 1
_CUDA_GRID_X_MAX = _INT32_MAX
_CUDA_GRID_Y_MAX = 2**16 - 1
_INPUT_ALIGNMENT_BYTES = 16

_NVFP4_GROUP = 16
_NVFP4_DIRECT_HALF = 8


class ScaleAlgo(IntEnum):
    """Compile-time algorithm used to derive and encode each block scale."""

    RCEIL_E8M0 = 0
    NVFP4_FP8_E4M3 = 1


@cache
def _cuda_capability(device: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


@cute.jit
def _mxfp8_v2_quantize_stochastic_x32(
    values: cute.TensorSSA,
    rcp: cutlass.Float32,
    counter_start: cutlass.Uint64,
    philox_k0: cutlass.Uint32,
    philox_k1: cutlass.Uint32,
) -> cute.TensorSSA:
    """Quantize one contiguous 32-value output run with two Philox counters."""
    scaled = values * rcp
    qwords = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Uint32)
    for half in cutlass.range_constexpr(2):
        ctr = counter_start + cutlass.Uint64(half)
        c0 = cutlass.Uint32(ctr & cutlass.Uint64(0xFFFFFFFF))
        c1 = cutlass.Uint32(ctr >> 32)
        zero = cutlass.Uint32(0)
        r0, r1, r2, r3 = _philox_4x32(
            c0, c1, zero, zero, philox_k0, philox_k1
        )
        value = half * 16
        word = half * 4
        qwords[word + 0] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 0], scaled[value + 1],
            scaled[value + 2], scaled[value + 3], r0,
        )
        qwords[word + 1] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 4], scaled[value + 5],
            scaled[value + 6], scaled[value + 7], r1,
        )
        qwords[word + 2] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 8], scaled[value + 9],
            scaled[value + 10], scaled[value + 11], r2,
        )
        qwords[word + 3] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[value + 12], scaled[value + 13],
            scaled[value + 14], scaled[value + 15], r3,
        )
    return cute.recast_tensor(qwords, dtype=cutlass.Float8E4M3FN).load()


@cute.jit
def _quantize_fp4_rtne(
    values: cute.TensorSSA,
    rcp: cutlass.Float32,
    value_count: cutlass.Constexpr,
) -> cute.TensorSSA:
    """Quantize an x16 or x32 FP32 group into packed E2M1 bytes with RTNE."""
    scaled = values * rcp
    qwords = cute.make_rmem_tensor(
        cute.make_layout(value_count // 8), cutlass.Uint32
    )
    for chunk in cutlass.range_constexpr(value_count // 8):
        offset = chunk * 8
        qwords[chunk] = _cvt_rn_satfinite_e2m1x2_f32_x4(
            scaled[offset + 0],
            scaled[offset + 1],
            scaled[offset + 2],
            scaled[offset + 3],
            scaled[offset + 4],
            scaled[offset + 5],
            scaled[offset + 6],
            scaled[offset + 7],
        )
    return cute.recast_tensor(qwords, dtype=cutlass.Uint8).load()


@cute.jit
def _quantize_fp4_stochastic_x16(
    values: cute.TensorSSA,
    rcp: cutlass.Float32,
    sr_counter: cutlass.Uint64,
    philox_k0: cutlass.Uint32,
    philox_k1: cutlass.Uint32,
) -> cute.TensorSSA:
    """Quantize 16 scaled FP32 values with one Philox counter and native E2M1 SR."""
    c0 = cutlass.Uint32(sr_counter & cutlass.Uint64(0xFFFFFFFF))
    c1 = cutlass.Uint32(sr_counter >> 32)
    zero = cutlass.Uint32(0)
    r0, r1, r2, r3 = _philox_4x32(
        c0, c1, zero, zero, philox_k0, philox_k1
    )
    qwords = cute.make_rmem_tensor(2, cutlass.Uint32)
    qwords[0] = _cvt_rs_satfinite_e2m1x4_f32_x8(
        values[0] * rcp,
        values[1] * rcp,
        values[2] * rcp,
        values[3] * rcp,
        values[4] * rcp,
        values[5] * rcp,
        values[6] * rcp,
        values[7] * rcp,
        r0,
        r2,
    )
    qwords[1] = _cvt_rs_satfinite_e2m1x4_f32_x8(
        values[8] * rcp,
        values[9] * rcp,
        values[10] * rcp,
        values[11] * rcp,
        values[12] * rcp,
        values[13] * rcp,
        values[14] * rcp,
        values[15] * rcp,
        r1,
        r3,
    )
    return cute.recast_tensor(qwords, dtype=cutlass.Uint8).load()


@cute.jit
def _blockscaled_quantize_group(
    values: cute.TensorSSA,
    outer,
    value_count: cutlass.Constexpr,
    qdata_dtype: cutlass.Constexpr,
    scale_algo: cutlass.Constexpr,
    is_square_scaling: cutlass.Constexpr,
    is_stochastic_qdata_rounding: cutlass.Constexpr,
    sr_counter: cutlass.Uint64 | None,
    philox_k0: cutlass.Uint32 | None,
    philox_k1: cutlass.Uint32 | None,
):
    """Calculate one block scale and quantize the values that share it."""
    if cutlass.const_expr(is_stochastic_qdata_rounding):
        assert sr_counter is not None
        assert philox_k0 is not None
        assert philox_k1 is not None
    else:
        assert sr_counter is None
        assert philox_k0 is None
        assert philox_k1 is None

    amax = cute.math.absf(values).reduce(
        cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
    )
    if cutlass.const_expr(is_square_scaling):
        amax = cute.arch.warp_reduction_max(amax)

    # ScaleAlgo selects only the scale calculation. The surrounding data path is shared.
    if cutlass.const_expr(scale_algo == ScaleAlgo.NVFP4_FP8_E4M3):
        scale, reciprocal = _nvfp4_scale_e4m3(amax, outer, outer)
    elif cutlass.const_expr(qdata_dtype == cutlass.Float4E2M1FN):
        reciprocal, scale = _e8m0_with_max_pos(amax, 6.0)
    else:
        reciprocal, scale = _e8m0(amax)

    if cutlass.const_expr(qdata_dtype == cutlass.Float4E2M1FN):
        if cutlass.const_expr(is_stochastic_qdata_rounding):
            qdata = _quantize_fp4_stochastic_x16(
                values, reciprocal, sr_counter, philox_k0, philox_k1
            )
        else:
            qdata = _quantize_fp4_rtne(values, reciprocal, value_count)
    elif cutlass.const_expr(is_stochastic_qdata_rounding):
        qdata = _mxfp8_v2_quantize_stochastic_x32(
            values, reciprocal, sr_counter, philox_k0, philox_k1
        )
    else:
        qdata = (values * reciprocal).to(cutlass.Float8E4M3FN)
    return qdata, scale


class _BlockscaledTma:
    """Compile-time MXFP8, MXFP4, or NVFP4 TMA kernel configuration."""

    def __init__(
        self,
        input_element_type,
        tile_m_size: int,
        tile_k_size: int,
        cluster_k: int,
        needs_boundary_masking: bool,
        quant_orientation: int,
        is_stochastic_qdata_rounding: bool,
        is_square_scaling: bool,
        qdata_dtype=cutlass.Float8E4M3FN,
        scale_algo: ScaleAlgo = ScaleAlgo.RCEIL_E8M0,
        has_dim_m_rht: bool = False,
    ) -> None:
        self.input_element_type = input_element_type
        self.tile_m_size = tile_m_size
        self.tile_k_size = tile_k_size
        self.cluster_k = cluster_k
        self.needs_boundary_masking = needs_boundary_masking
        self.quant_orientation = quant_orientation
        self.is_stochastic_qdata_rounding = is_stochastic_qdata_rounding
        self.is_square_scaling = is_square_scaling
        self.qdata_dtype = qdata_dtype
        self.scale_algo = scale_algo
        self.has_dim_m_rht = has_dim_m_rht
        self.scale_group_size = (
            16 if scale_algo == ScaleAlgo.NVFP4_FP8_E4M3 else 32
        )

    @cute.kernel
    def kernel(
        self,
        input_tma_atom: cute.CopyAtom,
        input_tma_tensor: cute.Tensor,
        output_k_tma_atom: cute.CopyAtom | None,
        output_k_tma_tensor: cute.Tensor | None,
        output_m_tma_atom: cute.CopyAtom | None,
        output_m_tma_tensor: cute.Tensor | None,
        mScaleKLogical: cute.Tensor | None,
        mScaleMLogical: cute.Tensor | None,
        mOuterScaleK: cute.Tensor | None,
        mOuterScaleM: cute.Tensor | None,
        mRhtSign: cute.Tensor | None,
        mSeed: cute.Tensor | None,
        input_smem_layout: cute.ComposedLayout,
        output_k_smem_layout: cute.ComposedLayout | None,
        output_m_smem_layout: cute.ComposedLayout | None,
        data_k_tv_layout: cute.Layout | None,
        output_k_tv_layout: cute.Layout | None,
        M: cutlass.Int32,
        K: cutlass.Int32,
    ) -> None:
        """
        Kernel for MXFP8, MXFP4, and NVFP4 quantization across dim-K, dim-M, and dim-KM.

        MXFP8 and NVFP4 support RTNE and SR, while MXFP4 currently supports RTNE only.

        High level flow:
          0. create smem scratchpad
             a. both dim-k and dim-m use a tile_m_size * tile_k_size input-typed scratchpad,
             b. dim-k reuses the leading region of 0a for qdata (to save smem)
             c. dim-m additionally uses a packed-qdata scratchpad
          1. load input data with TMA.
             a. If SR enabled, also load the random key while input data is loading
          2. barrier to wait for (1)
          3. if do_dim_m:
             - read input data from smem and quantize in registers
             - store scale to global memory
             - store qdata to smem scratchpad from 0c
          4. if do_dim_k:
             - read input data from smem and quantize in registers
             - store scale in registers
             - syncthreads for dim-k smem input reads
             - store qdata to smem scratchpad from 0b
          5. synchronize/publish dim-m and dim-k shared-memory output writes
          6. if do_dim_k, launch the dim-k TMA qdata store
          7. if do_dim_m, launch the dim-m TMA qdata store
          8. if do_dim_k, store scales from registers to global memory

        """

        input_element_type = self.input_element_type
        tile_m_size = cutlass.const_expr(self.tile_m_size)
        tile_k_size = cutlass.const_expr(self.tile_k_size)
        needs_boundary_masking = cutlass.const_expr(self.needs_boundary_masking)
        quant_orientation = cutlass.const_expr(self.quant_orientation)
        is_stochastic_qdata_rounding = cutlass.const_expr(
            self.is_stochastic_qdata_rounding
        )
        is_square_scaling = cutlass.const_expr(self.is_square_scaling)
        scale_algo = cutlass.const_expr(self.scale_algo)
        has_dim_m_rht = cutlass.const_expr(self.has_dim_m_rht)
        scale_group_size = cutlass.const_expr(self.scale_group_size)
        qdata_dtype = self.qdata_dtype
        is_packed_fp4_qdata = cutlass.const_expr(
            qdata_dtype == cutlass.Float4E2M1FN
        )
        qdata_storage_element_type = (
            cutlass.Uint8
            if cutlass.const_expr(is_packed_fp4_qdata)
            else cutlass.Float8E4M3FN
        )
        qdata_storage_elements_per_group = (
            scale_group_size // 2
            if cutlass.const_expr(is_packed_fp4_qdata)
            else scale_group_size
        )
        qdata_storage_elements_per_32 = (
            16 if cutlass.const_expr(is_packed_fp4_qdata) else 32
        )
        qdata_k_divisor = 2 if cutlass.const_expr(is_packed_fp4_qdata) else 1
        do_dim_k = quant_orientation != _QUANT_ORIENTATION_DIM_M
        do_dim_m = quant_orientation != _QUANT_ORIENTATION_DIM_K

        if cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_K):
            assert output_k_tma_atom is not None
            assert output_k_tma_tensor is not None
            assert mScaleKLogical is not None
            assert output_k_smem_layout is not None
            assert data_k_tv_layout is not None
            assert output_k_tv_layout is not None
            if cutlass.const_expr(scale_algo == ScaleAlgo.NVFP4_FP8_E4M3):
                assert mOuterScaleK is not None
            else:
                assert mOuterScaleK is None
            assert output_m_tma_atom is None
            assert output_m_tma_tensor is None
            assert mScaleMLogical is None
            assert mOuterScaleM is None
            assert output_m_smem_layout is None
        elif cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_M):
            assert output_k_tma_atom is None
            assert output_k_tma_tensor is None
            assert mScaleKLogical is None
            assert output_k_smem_layout is None
            assert data_k_tv_layout is None
            assert output_k_tv_layout is None
            assert mOuterScaleK is None
            assert output_m_tma_atom is not None
            assert output_m_tma_tensor is not None
            assert mScaleMLogical is not None
            assert output_m_smem_layout is not None
            if cutlass.const_expr(scale_algo == ScaleAlgo.NVFP4_FP8_E4M3):
                assert mOuterScaleM is not None
            else:
                assert mOuterScaleM is None
        else:
            assert quant_orientation == _QUANT_ORIENTATION_DIM_KM
            assert output_k_tma_atom is not None
            assert output_k_tma_tensor is not None
            assert mScaleKLogical is not None
            assert output_k_smem_layout is not None
            assert data_k_tv_layout is not None
            assert output_k_tv_layout is not None
            if cutlass.const_expr(scale_algo == ScaleAlgo.NVFP4_FP8_E4M3):
                assert mOuterScaleK is not None
            else:
                assert mOuterScaleK is None
            assert output_m_tma_atom is not None
            assert output_m_tma_tensor is not None
            assert mScaleMLogical is not None
            assert output_m_smem_layout is not None
            if cutlass.const_expr(scale_algo == ScaleAlgo.NVFP4_FP8_E4M3):
                assert mOuterScaleM is not None
            else:
                assert mOuterScaleM is None

        if cutlass.const_expr(has_dim_m_rht):
            assert scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
            assert do_dim_m
            assert mRhtSign is not None
        else:
            assert mRhtSign is None
        if cutlass.const_expr(is_stochastic_qdata_rounding):
            assert mSeed is not None
        else:
            assert mSeed is None
        if cutlass.const_expr(
            is_packed_fp4_qdata and scale_algo != ScaleAlgo.NVFP4_FP8_E4M3
        ):
            assert not is_stochastic_qdata_rounding
            assert not is_square_scaling
        if cutlass.const_expr(scale_algo == ScaleAlgo.NVFP4_FP8_E4M3):
            assert is_packed_fp4_qdata
            assert input_element_type == cutlass.BFloat16
            assert not is_square_scaling

        # bookkeeping
        tidx, _, _ = cute.arch.thread_idx()
        tile_k_idx, tile_m_idx, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if cutlass.const_expr(not needs_boundary_masking):
            M = cute.assume(M, divby=128)
            K = cute.assume(K, divby=128)
        elif cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_K):
            K = cute.assume(K, divby=32)
        elif cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_M):
            M = cute.assume(M, divby=32)
            K = cute.assume(K, divby=16)
        else:
            M = cute.assume(M, divby=32)
            K = cute.assume(K, divby=32)

        # create smem scratchpad
        smem = utils.SmemAllocator()
        input_storage = smem.allocate_array(
            input_element_type, tile_m_size * tile_k_size, byte_alignment=1024
        )
        if cutlass.const_expr(do_dim_m):
            output_m_storage = smem.allocate_array(
                qdata_storage_element_type,
                tile_m_size * tile_k_size // qdata_k_divisor,
                byte_alignment=1024,
            )
        if cutlass.const_expr(has_dim_m_rht):
            sign_storage = smem.allocate_array(
                cutlass.BFloat16,
                _NVFP4_GROUP,
                byte_alignment=16,
            )
            sSigns = cute.make_tensor(
                sign_storage, cute.make_layout(_NVFP4_GROUP)
            )
        # Put the small barrier after the 1KB-aligned tile buffers to avoid an otherwise unused 1016B
        # alignment gap at the front of every CTA's shared-memory allocation.
        tma_bar_ptr = smem.allocate_array(cutlass.Int64, 1)

        # create smem barrier for input data TMA load
        if tidx == 0:
            cute.arch.mbarrier_init(tma_bar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # set up TMA machinery
        # Note:
        # * All quant orientations share one input TMA load
        # * Dim-K aliases the input buffer for qdata output, to save smem
        # * Dim-M does not alias the input and has a separate qdata smem buffer
        sInput = cute.make_tensor(
            cute.recast_ptr(
                input_storage, input_smem_layout.inner, dtype=input_element_type
            ),
            input_smem_layout.outer,
        )
        gInput = cute.local_tile(input_tma_tensor, (tile_m_size, tile_k_size), (None, None))
        tInputsInput, tInputgInput = cpasync.tma_partition(
            input_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sInput, 0, 2),
            cute.group_modes(gInput, 0, 2),
        )

        if cutlass.const_expr(do_dim_k):
            sOutputK = cute.make_tensor(
                cute.recast_ptr(
                    input_storage,
                    output_k_smem_layout.inner,
                    dtype=qdata_storage_element_type,
                ),
                output_k_smem_layout.outer,
            )
            gOutputK = cute.local_tile(
                output_k_tma_tensor,
                (tile_m_size, tile_k_size // qdata_k_divisor),
                (None, None),
            )
            tOutputsK, tOutputgK = cpasync.tma_partition(
                output_k_tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sOutputK, 0, 2),
                cute.group_modes(gOutputK, 0, 2),
            )

        if cutlass.const_expr(do_dim_m):
            sOutputM = cute.make_tensor(
                cute.recast_ptr(
                    output_m_storage,
                    output_m_smem_layout.inner,
                    dtype=qdata_storage_element_type,
                ),
                output_m_smem_layout.outer,
            )
            gOutputM = cute.local_tile(
                output_m_tma_tensor,
                (tile_k_size, tile_m_size // qdata_k_divisor),
                (None, None),
            )
            tOutputsM, tOutputgM = cpasync.tma_partition(
                output_m_tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sOutputM, 0, 2),
                cute.group_modes(gOutputM, 0, 2),
            )

        # kick off input data TMA
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    tma_bar_ptr,
                    tile_m_size * tile_k_size * input_element_type.width // 8,
                )
            cute.copy(
                input_tma_atom,
                tInputgInput[(None, tile_m_idx, tile_k_idx)],
                tInputsInput,
                tma_bar_ptr=tma_bar_ptr,
            )

        philox_k0 = None
        philox_k1 = None
        sr_counter_base = None
        if cutlass.const_expr(is_stochastic_qdata_rounding):
            # The key is tile-independent, so load it while TMA fills sInput. In dim-KM, dim-M
            # follows dim-K's M*K elements in the same Philox stream.
            philox_k0, philox_k1, sr_counter_base = _nvfp4_load_philox_key(mSeed)

        if cutlass.const_expr(has_dim_m_rht):
            if tidx < _NVFP4_GROUP:
                sSigns[tidx] = (
                    mRhtSign[tidx].to(cutlass.Float32) * cutlass.Float32(0.25)
                ).to(cutlass.BFloat16)
            cute.arch.sync_threads()

        if cutlass.const_expr(
            scale_algo == ScaleAlgo.NVFP4_FP8_E4M3 and do_dim_k
        ):
            frgOuterScaleK = cute.make_rmem_tensor(
                cute.make_layout(1), mOuterScaleK.element_type
            )
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mOuterScaleK.element_type),
                mOuterScaleK,
                frgOuterScaleK,
            )
        if cutlass.const_expr(
            scale_algo == ScaleAlgo.NVFP4_FP8_E4M3 and do_dim_m
        ):
            frgOuterScaleM = cute.make_rmem_tensor(
                cute.make_layout(1), mOuterScaleM.element_type
            )
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mOuterScaleM.element_type),
                mOuterScaleM,
                frgOuterScaleM,
            )

        # wait for input data to arrive
        cute.arch.mbarrier_wait(tma_bar_ptr, 0)

        smem_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), input_element_type, num_bits_per_copy=128
        )
        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            qdata_storage_element_type,
            num_bits_per_copy=128,
        )

        if cutlass.const_expr(do_dim_m):
            # dim_m main pass (for dim_m and dim_km)
            # * read input data from smem and quantize in registers
            # * store scale to global memory
            # * store qdata to smem (to be stored to global memory) with TMA later
            #
            # Note:
            # Each dim-M thread owns one input column and processes every 32-row group. Its transposed
            # qdata uses separate shared memory, so the optional dim-K pass can still read sInput.
            row_blocks = tile_m_size // 32
            groups_per_row_block = 32 // scale_group_size
            scale_groups_m = tile_m_size // scale_group_size
            output_row_m = tile_k_idx * tile_k_size + tidx
            scale_col_m = tile_m_idx * scale_groups_m
            rScaleM = cute.make_rmem_tensor(scale_groups_m, cutlass.Uint8)
            if cutlass.const_expr(is_stochastic_qdata_rounding):
                # M is divisible by 16, so divide before the wide multiply and form the Philox
                # counter directly instead of materializing the larger flat element index.
                global_col_m = tile_k_idx * tile_k_size + tidx
                if cutlass.const_expr(do_dim_k):
                    # In dim-KM, place dim-M after dim-K's M*K elements in the Philox stream.
                    # Folding K into the transposed output row retains a single wide multiply.
                    sr_counter_base_m = (
                        sr_counter_base
                        + (cutlass.Uint64(K) + cutlass.Uint64(global_col_m))
                        * cutlass.Uint64(M // 16)
                        + cutlass.Uint64(tile_m_idx * tile_m_size // 16)
                    )
                else:
                    sr_counter_base_m = (
                        sr_counter_base
                        + cutlass.Uint64(global_col_m) * cutlass.Uint64(M // 16)
                        + cutlass.Uint64(tile_m_idx * tile_m_size // 16)
                    )
            if cutlass.const_expr(has_dim_m_rht):
                # Keep the scaled sign vector in registers across every RHT group this thread owns.
                rSigns = cute.make_rmem_tensor(_NVFP4_GROUP, cutlass.BFloat16)
                rSignHalves = cute.tiled_divide(rSigns, (_NVFP4_DIRECT_HALF,))
                sSignHalves = cute.tiled_divide(sSigns, (_NVFP4_DIRECT_HALF,))
                for half in cutlass.range_constexpr(2):
                    cute.copy(
                        smem_load_atom,
                        sSignHalves[(None, half)],
                        rSignHalves[(None, half)],
                    )
            for row_block in cutlass.range_constexpr(row_blocks):
                rQM = cute.make_rmem_tensor(
                    qdata_storage_elements_per_32,
                    qdata_storage_element_type,
                )
                rQMGroups = cute.tiled_divide(
                    rQM, (qdata_storage_elements_per_group,)
                )
                for group in cutlass.range_constexpr(groups_per_row_block):
                    row_start = row_block * 32 + group * scale_group_size
                    if cutlass.const_expr(has_dim_m_rht):
                        vm = _nvfp4_rht_fwht_x16(
                            sInput, row_start, tidx, rSigns
                        )
                    else:
                        rInputM = cute.make_rmem_tensor(
                            scale_group_size, input_element_type
                        )
                        for value in cutlass.range_constexpr(scale_group_size):
                            rInputM[value] = sInput[(row_start + value, tidx)]
                        vm = rInputM.load().to(cutlass.Float32)
                    if cutlass.const_expr(is_stochastic_qdata_rounding):
                        sr_counter_m = sr_counter_base_m + cutlass.Uint64(
                            row_block * (32 // 16)
                            + group * (scale_group_size // 16)
                        )
                    else:
                        sr_counter_m = None
                    dim_m_outer = (
                        frgOuterScaleM[0]
                        if cutlass.const_expr(
                            scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
                        )
                        else None
                    )
                    qdata_m, scale_m = _blockscaled_quantize_group(
                        vm,
                        dim_m_outer,
                        scale_group_size,
                        qdata_dtype,
                        scale_algo,
                        False,
                        is_stochastic_qdata_rounding,
                        sr_counter_m,
                        philox_k0,
                        philox_k1,
                    )
                    rQMGroups[(None, group)].store(qdata_m)
                    rScaleM[
                        row_block * groups_per_row_block + group
                    ] = scale_m.to(cutlass.Uint8)

                rQM16 = cute.tiled_divide(rQM, (16,))
                sQM16 = cute.tiled_divide(sOutputM[(tidx, None)], (16,))
                for vec in cutlass.range_constexpr(
                    qdata_storage_elements_per_32 // 16
                ):
                    cute.copy(
                        smem_store_atom,
                        rQM16[(None, vec)],
                        sQM16[
                            (
                                None,
                                row_block
                                * (qdata_storage_elements_per_32 // 16)
                                + vec,
                            )
                        ],
                    )

            use_full_tile_m = cutlass.const_expr(
                not needs_boundary_masking
            ) or (
                (tile_m_idx < M // tile_m_size)
                & (tile_k_idx < K // tile_k_size)
            )
            if use_full_tile_m:
                _store_swizzled_scale_groups_as_uint(
                    mScaleMLogical,
                    rScaleM,
                    output_row_m,
                    scale_col_m,
                    scale_groups_m,
                )
            else:
                rScaleMPadded = cute.make_rmem_tensor(
                    scale_groups_m, cutlass.Uint8
                )
                # pad unused scale entries with zeros
                rScaleMPadded.fill(0)
                if output_row_m < K:
                    for group in cutlass.range_constexpr(scale_groups_m):
                        if scale_col_m + group < M // scale_group_size:
                            rScaleMPadded[group] = rScaleM[group]
                if output_row_m < _ceil_div(K, 128) * 128:
                    _store_swizzled_scale_groups_as_uint(
                        mScaleMLogical,
                        rScaleMPadded,
                        output_row_m,
                        scale_col_m,
                        scale_groups_m,
                    )

        if cutlass.const_expr(do_dim_k):
            # dim-k main pass (for dim_k and dim_km)
            # * read input data from smem and quantize in registers
            # * keep scale to registers
            # * store qdata in smem, overwriting the first half of input data (to save smem)
            groups_per_row_k = tile_k_size // scale_group_size
            row_owned_k = scale_group_size == 16 or (
                tile_m_size == _DIM_K_TILE_M_SIZE_128
            )
            if cutlass.const_expr(row_owned_k):
                threads_per_row_k = _MIN_CTA_THREADS_128 // tile_m_size
                iters = groups_per_row_k // threads_per_row_k
            else:
                iters = tile_m_size // 32
            tidfrgInputK = cute.composition(sInput, data_k_tv_layout)
            tidfrgOutputK = cute.composition(sOutputK, output_k_tv_layout)
            thrInputGroupsK = tidfrgInputK[(tidx, None)]
            thrOutputGroupsK = tidfrgOutputK[(tidx, None)]
            rScaleK = cute.make_rmem_tensor(iters, cutlass.Uint8)
            rQK = cute.make_rmem_tensor(
                qdata_storage_elements_per_group * iters,
                qdata_storage_element_type,
            )
            rQGroupsK = cute.tiled_divide(
                rQK, (qdata_storage_elements_per_group,)
            )
            if cutlass.const_expr(is_stochastic_qdata_rounding):
                if cutlass.const_expr(row_owned_k):
                    global_row_k = (
                        tile_m_idx * tile_m_size + tidx // threads_per_row_k
                    )
                    global_scale_col_k = (
                        tile_k_idx * groups_per_row_k
                        + (tidx % threads_per_row_k) * iters
                    )
                    sr_counter_base_k = (
                        sr_counter_base
                        + cutlass.Uint64(global_row_k) * cutlass.Uint64(K // 16)
                        + cutlass.Uint64(
                            global_scale_col_k * (scale_group_size // 16)
                        )
                    )
                else:
                    global_row_k = (
                        tile_m_idx * tile_m_size + tidx // groups_per_row_k
                    )
                    global_group_k = (
                        tile_k_idx * groups_per_row_k
                        + tidx % groups_per_row_k
                    )
                    sr_counter_base_k = (
                        sr_counter_base
                        + cutlass.Uint64(global_row_k) * cutlass.Uint64(K // 16)
                        + cutlass.Uint64(
                            global_group_k * (scale_group_size // 16)
                        )
                    )
            for it in cutlass.range_constexpr(iters):
                if cutlass.const_expr(is_stochastic_qdata_rounding):
                    if cutlass.const_expr(row_owned_k):
                        sr_counter_start_k = sr_counter_base_k + cutlass.Uint64(
                            it * (scale_group_size // 16)
                        )
                    else:
                        sr_counter_start_k = (
                            sr_counter_base_k
                            + cutlass.Uint64(it * 2) * cutlass.Uint64(K)
                        )
                else:
                    sr_counter_start_k = None
                sGroupK = thrInputGroupsK[((None, it),)]
                rInputK = cute.make_rmem_tensor(
                    scale_group_size, input_element_type
                )
                input_values_per_copy = 128 // input_element_type.width
                sGroupKVec = cute.tiled_divide(sGroupK, (input_values_per_copy,))
                rInputKVec = cute.tiled_divide(rInputK, (input_values_per_copy,))
                for vec in cutlass.range_constexpr(
                    scale_group_size // input_values_per_copy
                ):
                    cute.copy(
                        smem_load_atom,
                        sGroupKVec[(None, vec)],
                        rInputKVec[(None, vec)],
                    )
                vk = rInputK.load().to(cutlass.Float32)
                dim_k_outer = (
                    frgOuterScaleK[0]
                    if cutlass.const_expr(
                        scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
                    )
                    else None
                )
                qdata_k, scale_k = _blockscaled_quantize_group(
                    vk,
                    dim_k_outer,
                    scale_group_size,
                    qdata_dtype,
                    scale_algo,
                    is_square_scaling,
                    is_stochastic_qdata_rounding,
                    sr_counter_start_k,
                    philox_k0,
                    philox_k1,
                )
                rQGroupsK[(None, it)].store(qdata_k)
                rScaleK[it] = scale_k.to(rScaleK.element_type)

            # All input reads must finish before the aliased qK view overwrites the input tile.
            cute.arch.sync_threads()
            rQK16 = cute.tiled_divide(rQK, (16,))
            sGroupK16 = cute.tiled_divide(thrOutputGroupsK, (16,))
            for vec in cutlass.range_constexpr(
                qdata_storage_elements_per_group * iters // 16
            ):
                cute.copy(
                    smem_store_atom,
                    rQK16[(None, vec)],
                    sGroupK16[(None, vec)],
                )

        # Publish all enabled qdata writes before launching their TMA transfers.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        if cutlass.const_expr(do_dim_k):
            # kick off dim-k TMA qdata store
            if warp == 0:
                cute.copy(
                    output_k_tma_atom,
                    tOutputsK,
                    tOutputgK[(None, tile_m_idx, tile_k_idx)],
                )
        if cutlass.const_expr(do_dim_m):
            # The fused SR specialization issues its independent output transfers from separate warps.
            output_m_warp = (
                1
                if cutlass.const_expr(is_stochastic_qdata_rounding and do_dim_k)
                else 0
            )
            if warp == output_m_warp:
                cute.copy(
                    output_m_tma_atom,
                    tOutputsM,
                    tOutputgM[(None, tile_k_idx, tile_m_idx)],
                )

        if cutlass.const_expr(do_dim_k):
            # do the dim-k scale write (overlaps with qdata TMA store)
            use_full_tile_k = cutlass.const_expr(not needs_boundary_masking) or (
                (tile_m_idx < M // tile_m_size) & (tile_k_idx < K // tile_k_size)
            )
            if cutlass.const_expr(row_owned_k):
                input_row_k = (
                    tile_m_idx * tile_m_size + tidx // threads_per_row_k
                )
                scale_col_k = (
                    tile_k_idx * groups_per_row_k
                    + (tidx % threads_per_row_k) * iters
                )
                if use_full_tile_k:
                    _store_swizzled_scale_groups_as_uint(
                        mScaleKLogical,
                        rScaleK,
                        input_row_k,
                        scale_col_k,
                        iters,
                    )
                else:
                    rScaleKPadded = cute.make_rmem_tensor(iters, cutlass.Uint8)
                    rScaleKPadded.fill(0)
                    if input_row_k < M:
                        for it in cutlass.range_constexpr(iters):
                            if scale_col_k + it < K // scale_group_size:
                                rScaleKPadded[it] = rScaleK[it]
                    _store_swizzled_scale_groups_as_uint(
                        mScaleKLogical,
                        rScaleKPadded,
                        input_row_k,
                        scale_col_k,
                        iters,
                    )

                if cutlass.const_expr(needs_boundary_masking):
                    ncb_k = _ceil_div(K, scale_group_size * 4)
                    grid_n = _ceil_div(K, tile_k_size)
                    covered_groups = grid_n * groups_per_row_k
                    if covered_groups < ncb_k * 4:
                        if tile_k_idx == grid_n - 1:
                            for offset in cutlass.range_constexpr(3):
                                col = covered_groups + offset
                                if col < ncb_k * 4:
                                    mScaleKLogical[(input_row_k, col)] = cutlass.Uint8(0)
            else:
                # Short tiles distribute (row, 1x32 group) pairs across all threads. The fused 1D path
                # is group-major; square scaling is warp/block-major. Both need individual scale stores
                # because their slots are strided across rows.
                if cutlass.const_expr(is_square_scaling):
                    local_group_k = tidx // 32
                    local_row_k = tidx % 32
                else:
                    local_group_k = tidx % groups_per_row_k
                    local_row_k = tidx // groups_per_row_k
                for it in cutlass.range_constexpr(iters):
                    input_row_k = tile_m_idx * tile_m_size + local_row_k + it * 32
                    scale_col_k = (
                        tile_k_idx * groups_per_row_k + local_group_k
                    )
                    if use_full_tile_k:
                        mScaleKLogical[(input_row_k, scale_col_k)] = rScaleK[it]
                    else:
                        scale_k = cutlass.Uint8(0)
                        if input_row_k < M:
                            if scale_col_k < K // scale_group_size:
                                scale_k = rScaleK[it]
                        mScaleKLogical[(input_row_k, scale_col_k)] = scale_k

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutputK: cute.Tensor | None,
        mScaleK: cute.Tensor | None,
        mOuterScaleK: cute.Tensor | None,
        mOutputM: cute.Tensor | None,
        mScaleM: cute.Tensor | None,
        mOuterScaleM: cute.Tensor | None,
        mRhtSign: cute.Tensor | None,
        mSeed: cute.Tensor | None,
        stream: cuda.CUstream,
        M: cutlass.Int32,
        K: cutlass.Int32,
    ) -> None:
        tile_m_size = cutlass.const_expr(self.tile_m_size)
        tile_k_size = cutlass.const_expr(self.tile_k_size)
        cluster_k = cutlass.const_expr(self.cluster_k)
        needs_boundary_masking = cutlass.const_expr(self.needs_boundary_masking)
        quant_orientation = cutlass.const_expr(self.quant_orientation)
        is_stochastic_qdata_rounding = cutlass.const_expr(
            self.is_stochastic_qdata_rounding
        )
        is_square_scaling = cutlass.const_expr(self.is_square_scaling)
        scale_algo = cutlass.const_expr(self.scale_algo)
        has_dim_m_rht = cutlass.const_expr(self.has_dim_m_rht)
        scale_group_size = cutlass.const_expr(self.scale_group_size)
        qdata_dtype = self.qdata_dtype
        is_packed_fp4_qdata = cutlass.const_expr(
            qdata_dtype == cutlass.Float4E2M1FN
        )
        qdata_storage_element_type = (
            cutlass.Uint8
            if cutlass.const_expr(is_packed_fp4_qdata)
            else cutlass.Float8E4M3FN
        )
        qdata_k_divisor = 2 if cutlass.const_expr(is_packed_fp4_qdata) else 1
        qdata_storage_elements_per_group = (
            scale_group_size // 2
            if cutlass.const_expr(is_packed_fp4_qdata)
            else scale_group_size
        )

        do_dim_k = quant_orientation != _QUANT_ORIENTATION_DIM_M
        do_dim_m = quant_orientation != _QUANT_ORIENTATION_DIM_K

        if cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_K):
            assert mOutputK is not None
            assert mScaleK is not None
            assert (mOuterScaleK is not None) == (
                scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
            )
            assert mOutputM is None
            assert mScaleM is None
            assert mOuterScaleM is None
        elif cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_M):
            assert mOutputK is None
            assert mScaleK is None
            assert mOuterScaleK is None
            assert mOutputM is not None
            assert mScaleM is not None
            assert (mOuterScaleM is not None) == (
                scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
            )
        else:
            assert quant_orientation == _QUANT_ORIENTATION_DIM_KM
            assert mOutputK is not None
            assert mScaleK is not None
            assert (mOuterScaleK is not None) == (
                scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
            )
            assert mOutputM is not None
            assert mScaleM is not None
            assert (mOuterScaleM is not None) == (
                scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
            )

        if cutlass.const_expr(has_dim_m_rht):
            assert (
                scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
                and do_dim_m
                and mRhtSign is not None
            )
        else:
            assert mRhtSign is None
        if cutlass.const_expr(is_stochastic_qdata_rounding):
            assert mSeed is not None
        else:
            assert mSeed is None
        if cutlass.const_expr(is_square_scaling):
            assert quant_orientation == _QUANT_ORIENTATION_DIM_K
        if cutlass.const_expr(
            is_packed_fp4_qdata and scale_algo != ScaleAlgo.NVFP4_FP8_E4M3
        ):
            assert not is_stochastic_qdata_rounding
            assert not is_square_scaling
        if cutlass.const_expr(scale_algo == ScaleAlgo.NVFP4_FP8_E4M3):
            assert is_packed_fp4_qdata

        # M is padded to a full 128-row scale-layout block. Since every tile-M divides 128,
        # compute the CTA count without materializing a potentially overflowing padded extent.
        if cutlass.const_expr(
            scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
            and quant_orientation == _QUANT_ORIENTATION_DIM_M
        ):
            grid_m = _ceil_div(M, 64) * 64 // tile_m_size
        else:
            grid_m = _ceil_div(M, 128) * (128 // tile_m_size)
        grid_k = _ceil_div(K, tile_k_size)

        if cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_M):
            # Keep the kernel argument type uniform while retaining the original unswizzled dim-M
            # address mapping.
            input_smem_layout = cute.make_composed_layout(
                cute.make_swizzle(0, 0, 0),
                0,
                cute.make_layout((tile_m_size, tile_k_size), stride=(tile_k_size, 1)),
            )
        else:
            # Keep each 128-bit row vector intact while XORing row bits into the shared-memory bank
            # selection. This targets the dim-K phase's 16-way row-read conflicts.
            use_k_sw64 = (
                scale_algo != ScaleAlgo.NVFP4_FP8_E4M3
                and quant_orientation == _QUANT_ORIENTATION_DIM_K
                and tile_k_size == 32
            )
            input_smem_kind = (
                tcgen05.SmemLayoutAtomKind.K_SW64
                if cutlass.const_expr(use_k_sw64)
                else tcgen05.SmemLayoutAtomKind.K_SW128
            )
            input_smem_atom = tcgen05.make_smem_layout_atom(
                input_smem_kind, mInput.element_type
            )
            input_smem_layout = cute.coalesce(
                cute.tile_to_shape(input_smem_atom, (tile_m_size, tile_k_size), order=(0, 1)),
                target_profile=(1, 1),
            )

        input_tma_atom, input_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mInput,
            input_smem_layout,
            (tile_m_size, tile_k_size),
        )

        if cutlass.const_expr(do_dim_k):
            if cutlass.const_expr(scale_algo == ScaleAlgo.NVFP4_FP8_E4M3):
                if cutlass.const_expr(
                    quant_orientation == _QUANT_ORIENTATION_DIM_K
                ):
                    output_k_smem_kind = (
                        tcgen05.SmemLayoutAtomKind.K_SW64
                        if cutlass.const_expr(tile_k_size == 128)
                        else tcgen05.SmemLayoutAtomKind.K_SW32
                    )
                else:
                    output_k_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW64
                output_k_smem_atom = tcgen05.make_smem_layout_atom(
                    output_k_smem_kind, qdata_storage_element_type
                )
                output_k_smem_layout = cute.coalesce(
                    cute.tile_to_shape(
                        output_k_smem_atom,
                        (tile_m_size, tile_k_size // qdata_k_divisor),
                        order=(0, 1),
                    ),
                    target_profile=(1, 1),
                )
            elif cutlass.const_expr(is_packed_fp4_qdata):
                output_k_smem_layout = cute.make_composed_layout(
                    cute.make_swizzle(0, 0, 0),
                    0,
                    cute.make_layout(
                        (tile_m_size, tile_k_size // 2),
                        stride=(tile_k_size // 2, 1),
                    ),
                )
            else:
                if cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_K):
                    if cutlass.const_expr(tile_k_size == 32):
                        output_k_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW32
                    elif cutlass.const_expr(tile_k_size == 64):
                        output_k_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW64
                    else:
                        output_k_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW128
                else:
                    output_k_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW128
                output_k_smem_atom = tcgen05.make_smem_layout_atom(
                    output_k_smem_kind, qdata_storage_element_type
                )
                output_k_smem_layout = cute.coalesce(
                    cute.tile_to_shape(
                        output_k_smem_atom,
                        (tile_m_size, tile_k_size),
                        order=(0, 1),
                    ),
                    target_profile=(1, 1),
                )
            output_k_tma_atom, output_k_tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mOutputK,
                output_k_smem_layout,
                (tile_m_size, tile_k_size // qdata_k_divisor),
            )

            nrb_k = _ceil_div(M, 128)
            ncb_k = _ceil_div(K, scale_group_size * 4)
            scale_k_row_block_stride = cutlass.Int64(ncb_k) * 32 * 16
            scale_k_layout = cute.make_layout(
                ((32, 4, nrb_k), (4, ncb_k)),
                stride=((16, 4, scale_k_row_block_stride), (1, 32 * 16)),
            )
            mScaleKLogical = cute.make_tensor(mScaleK.iterator, scale_k_layout)

            groups_per_row_k = tile_k_size // scale_group_size
            row_owned_k = scale_group_size == 16 or (
                tile_m_size == _DIM_K_TILE_M_SIZE_128
            )
            if cutlass.const_expr(row_owned_k):
                threads_per_row_k = _MIN_CTA_THREADS_128 // tile_m_size
                groups_per_thread_k = (
                    groups_per_row_k // threads_per_row_k
                )
                data_k_tv_layout = cute.make_layout(
                    (
                        (threads_per_row_k, tile_m_size),
                        (scale_group_size, groups_per_thread_k),
                    ),
                    stride=(
                        (
                            tile_m_size
                            * scale_group_size
                            * groups_per_thread_k,
                            1,
                        ),
                        (tile_m_size, tile_m_size * scale_group_size),
                    ),
                )
                output_k_tv_layout = cute.make_layout(
                    (
                        (threads_per_row_k, tile_m_size),
                        (
                            qdata_storage_elements_per_group,
                            groups_per_thread_k,
                        ),
                    ),
                    stride=(
                        (
                            tile_m_size
                            * qdata_storage_elements_per_group
                            * groups_per_thread_k,
                            1,
                        ),
                        (
                            tile_m_size,
                            tile_m_size * qdata_storage_elements_per_group,
                        ),
                    ),
                )
            elif cutlass.const_expr(is_square_scaling):
                data_k_tv_layout = cute.make_layout(
                    ((32, groups_per_row_k), (32, 1)),
                    stride=((1, tile_m_size * 32), (tile_m_size, tile_m_size * tile_k_size)),
                )
                output_k_tv_layout = data_k_tv_layout
            else:
                rows_per_stage = _MIN_CTA_THREADS_128 // groups_per_row_k
                row_blocks = tile_m_size // rows_per_stage
                data_k_tv_layout = cute.make_layout(
                    (
                        (groups_per_row_k, rows_per_stage),
                        (scale_group_size, row_blocks),
                    ),
                    stride=((tile_m_size * 32, 1), (tile_m_size, rows_per_stage)),
                )
                if cutlass.const_expr(is_packed_fp4_qdata):
                    output_k_tv_layout = cute.make_layout(
                        (
                            (groups_per_row_k, rows_per_stage),
                            (qdata_storage_elements_per_group, row_blocks),
                        ),
                        stride=(
                            (tile_m_size * qdata_storage_elements_per_group, 1),
                            (tile_m_size, rows_per_stage),
                        ),
                    )
                else:
                    output_k_tv_layout = data_k_tv_layout
        else:
            output_k_smem_layout = None
            output_k_tma_atom = None
            output_k_tma_tensor = None
            mScaleKLogical = None
            data_k_tv_layout = None
            output_k_tv_layout = None

        if cutlass.const_expr(do_dim_m):
            if cutlass.const_expr(
                scale_algo == ScaleAlgo.NVFP4_FP8_E4M3 and tile_m_size != 32
            ):
                output_m_smem_kind = (
                    tcgen05.SmemLayoutAtomKind.K_SW64
                    if cutlass.const_expr(tile_m_size == 128)
                    else tcgen05.SmemLayoutAtomKind.K_SW32
                )
                output_m_smem_atom = tcgen05.make_smem_layout_atom(
                    output_m_smem_kind, qdata_storage_element_type
                )
                output_m_smem_layout = cute.coalesce(
                    cute.tile_to_shape(
                        output_m_smem_atom,
                        (tile_k_size, tile_m_size // qdata_k_divisor),
                        order=(0, 1),
                    ),
                    target_profile=(1, 1),
                )
            else:
                output_m_smem_layout = cute.make_composed_layout(
                    cute.make_swizzle(0, 0, 0),
                    0,
                    cute.make_layout(
                        (tile_k_size, tile_m_size // qdata_k_divisor),
                        stride=(tile_m_size // qdata_k_divisor, 1),
                    ),
                )
            output_m_tma_atom, output_m_tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mOutputM,
                output_m_smem_layout,
                (tile_k_size, tile_m_size // qdata_k_divisor),
            )

            nrb_m = _ceil_div(K, 128)
            ncb_m = _ceil_div(M, scale_group_size * 4)
            scale_m_row_block_stride = cutlass.Int64(ncb_m) * 32 * 16
            scale_m_layout = cute.make_layout(
                ((32, 4, nrb_m), (4, ncb_m)),
                stride=((16, 4, scale_m_row_block_stride), (1, 32 * 16)),
            )
            mScaleMLogical = cute.make_tensor(mScaleM.iterator, scale_m_layout)
        else:
            output_m_smem_layout = None
            output_m_tma_atom = None
            output_m_tma_tensor = None
            mScaleMLogical = None

        kernel = self.kernel(
            input_tma_atom,
            input_tma_tensor,
            output_k_tma_atom,
            output_k_tma_tensor,
            output_m_tma_atom,
            output_m_tma_tensor,
            mScaleKLogical,
            mScaleMLogical,
            mOuterScaleK,
            mOuterScaleM,
            mRhtSign,
            mSeed,
            input_smem_layout,
            output_k_smem_layout,
            output_m_smem_layout,
            data_k_tv_layout,
            output_k_tv_layout,
            M,
            K,
        )
        grid = (grid_k, grid_m, 1)
        block_threads = max(_MIN_CTA_THREADS_128, tile_k_size)
        block = (block_threads, 1, 1)
        if cutlass.const_expr(scale_algo == ScaleAlgo.NVFP4_FP8_E4M3):
            launch_cluster = (
                (cluster_k, 1, 1)
                if cutlass.const_expr(
                    quant_orientation == _QUANT_ORIENTATION_DIM_K
                    or cluster_k != 1
                )
                else None
            )
        else:
            if cutlass.const_expr(
                quant_orientation == _QUANT_ORIENTATION_DIM_KM
                and not is_stochastic_qdata_rounding
                and tile_m_size != 32
            ):
                # A degenerate cluster constrains residency for this larger two-output
                # specialization; an ordinary launch lets Blackwell keep nine CTAs resident per SM.
                launch_cluster = None
            else:
                # K-major scheduling keeps adjacent row-major input columns together.
                launch_cluster = (cluster_k, 1, 1)
        kernel.launch(grid=grid, block=block, cluster=launch_cluster, stream=stream)




def _make_dynamic_matrix_fake(dtype):
    """Match a 16-byte-aligned row-major tensor with a dynamic, 16-divisible K."""
    return cute.runtime.make_fake_tensor(
        dtype,
        (cute.sym_int(), cute.sym_int(divisibility=16)),
        stride=(cute.sym_int64(divisibility=16), 1),
        assumed_align=16,
    )


def _make_dynamic_scale_fake():
    """Match the compact, padded byte-scale allocation used by the runtime wrapper."""
    return cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (cute.sym_int(divisibility=512),),
        stride=(1,),
        assumed_align=4,
    )


def _make_static_vector_fake(dtype, size: int, assumed_align: int):
    return cute.runtime.make_fake_tensor(
        dtype,
        (size,),
        stride=(1,),
        assumed_align=assumed_align,
    )


def _blockscaled_tma_compile_log_key(
    input_dtype: torch.dtype,
    tile_m_size: int,
    tile_k_size: int,
    cluster_k: int,
    needs_boundary_masking: bool,
    quant_orientation: int,
    is_stochastic_qdata_rounding: bool,
    is_square_scaling: bool,
    qdata_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_algo: ScaleAlgo = ScaleAlgo.RCEIL_E8M0,
    has_dim_m_rht: bool = False,
) -> str:
    return (
        f"dtype={input_dtype} orientation={quant_orientation} "
        f"tile={tile_m_size}x{tile_k_size} cluster_k={cluster_k} "
        f"masking={needs_boundary_masking} sr={is_stochastic_qdata_rounding} "
        f"square={is_square_scaling} qdata_dtype={qdata_dtype} "
        f"scale_algo={scale_algo.name} dim_m_rht={has_dim_m_rht}"
    )


@instrumented_cutedsl_cache(
    "quant_cast_bench::blockscaled_tma",
    key_fn=_blockscaled_tma_compile_log_key,
)
def _compile_blockscaled_tma(
    input_dtype: torch.dtype,
    tile_m_size: int,
    tile_k_size: int,
    cluster_k: int,
    needs_boundary_masking: bool,
    quant_orientation: int,
    is_stochastic_qdata_rounding: bool,
    is_square_scaling: bool,
    qdata_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_algo: ScaleAlgo = ScaleAlgo.RCEIL_E8M0,
    has_dim_m_rht: bool = False,
):
    if qdata_dtype not in _TORCH_TO_CUTE_QDATA_DTYPE:
        raise ValueError(f"unsupported qdata dtype: {qdata_dtype}")
    input_element_type = _TORCH_TO_CUTE_DTYPE[input_dtype]
    qdata_element_type = _TORCH_TO_CUTE_QDATA_DTYPE[qdata_dtype]
    if scale_algo == ScaleAlgo.NVFP4_FP8_E4M3:
        if input_dtype != torch.bfloat16:
            raise ValueError("NVFP4 TMA supports only bfloat16 input")
        if qdata_dtype != torch.float4_e2m1fn_x2:
            raise ValueError("NVFP4 scaling requires float4_e2m1fn_x2 qdata")
        if is_square_scaling:
            raise ValueError("NVFP4 scaling does not support square scaling")
    do_dim_k = quant_orientation != _QUANT_ORIENTATION_DIM_M
    do_dim_m = quant_orientation != _QUANT_ORIENTATION_DIM_K

    operation = _BlockscaledTma(
        input_element_type,
        tile_m_size,
        tile_k_size,
        cluster_k,
        needs_boundary_masking,
        quant_orientation,
        is_stochastic_qdata_rounding,
        is_square_scaling,
        qdata_element_type,
        scale_algo,
        has_dim_m_rht,
    )

    mInput = _make_dynamic_matrix_fake(input_element_type)
    qdata_storage_element_type = (
        cutlass.Uint8
        if qdata_dtype == torch.float4_e2m1fn_x2
        else cutlass.Float8E4M3FN
    )
    mOutputK = (
        _make_dynamic_matrix_fake(qdata_storage_element_type) if do_dim_k else None
    )
    mScaleK = _make_dynamic_scale_fake() if do_dim_k else None
    is_nvfp4 = scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
    mOuterScaleK = (
        _make_static_vector_fake(cutlass.Float32, 1, assumed_align=4)
        if is_nvfp4 and do_dim_k
        else None
    )
    mOutputM = (
        _make_dynamic_matrix_fake(qdata_storage_element_type) if do_dim_m else None
    )
    mScaleM = _make_dynamic_scale_fake() if do_dim_m else None
    mOuterScaleM = (
        _make_static_vector_fake(cutlass.Float32, 1, assumed_align=4)
        if is_nvfp4 and do_dim_m
        else None
    )
    mRhtSign = (
        _make_static_vector_fake(cutlass.BFloat16, _NVFP4_GROUP, assumed_align=16)
        if is_nvfp4 and has_dim_m_rht
        else None
    )
    mSeed = (
        cute.runtime.make_fake_tensor(
            cutlass.Int64,
            (2,),
            stride=(1,),
            assumed_align=8,
        )
        if is_stochastic_qdata_rounding
        else None
    )

    return cute.compile(
        operation,
        mInput,
        mOutputK,
        mScaleK,
        mOuterScaleK,
        mOutputM,
        mScaleM,
        mOuterScaleM,
        mRhtSign,
        mSeed,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        cutlass.Int32(0),
        cutlass.Int32(0),
        options="--enable-tvm-ffi",
    )


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
    rht_sign: torch.Tensor | None = None,
):
    if quant_orientation not in ("dim_k", "dim_m", "dim_km"):
        raise ValueError(f"unsupported quant_orientation: {quant_orientation}")
    do_dim_k = quant_orientation != "dim_m"
    do_dim_m = quant_orientation != "dim_k"
    is_nvfp4 = scale_algo == ScaleAlgo.NVFP4_FP8_E4M3
    has_dim_m_rht = rht_sign is not None

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
        if rht_sign is not None:
            raise ValueError("RCEIL scaling does not use an RHT sign tensor")
    rounding_mode = str(getattr(rounding_mode, "value", rounding_mode)).lower()
    if is_nvfp4:
        assert rounding_mode in ("rtne", "stochastic"), (
            f"unsupported rounding_mode: {rounding_mode}"
        )
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
        if has_dim_m_rht:
            assert do_dim_m, "RHT requires a dim-m output"
            assert rht_sign.shape == (16,), "RHT sign input must have shape (16,)"
            assert (
                rht_sign.dtype == torch.bfloat16
                and rht_sign.device == input.device
            ), "RHT sign input must be bf16 on the input device"
            assert rht_sign.is_contiguous(), "RHT sign input must be contiguous"
        if is_stochastic_qdata_rounding:
            assert has_dim_m_rht and do_dim_m, (
                "stochastic rounding currently requires an RHT dim-m output"
            )
            assert key is not None, "stochastic rounding requires a Philox key"
            assert key.device == input.device, (
                "input and Philox key must be on the same device"
            )
            assert key.dtype == torch.uint64 and key.numel() == 2, (
                "Philox key must be uint64[2]"
            )
        else:
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

    if is_nvfp4 and quant_orientation == "dim_k":
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
        fused_rht = quant_orientation == "dim_km" and has_dim_m_rht
        if fused_rht:
            tile_m_size = 32 if M * K <= 2048 * 2048 else 64
            tile_k_size = 128
        elif M * K <= 2048 * 2048:
            tile_m_size, tile_k_size = 32, 128
        elif is_stochastic_qdata_rounding and M * K >= 8192 * 8192:
            tile_m_size, tile_k_size = 128, 128
        else:
            tile_m_size, tile_k_size = 64, 128
        grid_k = _ceil_div(K, tile_k_size)
        if fused_rht:
            cluster_k = (
                2
                if (is_stochastic_qdata_rounding or M * K <= 4096 * 4096)
                and grid_k % 2 == 0
                else 1
            )
        elif is_stochastic_qdata_rounding and M * K >= 8192 * 8192:
            cluster_k = 2 if grid_k % 2 == 0 else 1
        else:
            cluster_k = (
                2
                if has_dim_m_rht and M * K <= 2048 * 2048 and grid_k % 2 == 0
                else 1
            )
        needs_boundary_masking = M % 128 != 0 or K % 128 != 0
    elif quant_orientation == "dim_k":
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
            if input.dtype == torch.float32
            or is_stochastic_qdata_rounding
            or (is_square_scaling and M * K <= 4096 * 4096)
            else next(c for c in (16, 8, 4, 2, 1) if c <= ncb_k and grid_k % c == 0)
        )
        needs_boundary_masking = M != nrb_k * 128 or K != ncb_k * 128
    elif quant_orientation == "dim_m":
        # TODO: For MXFP4 dim-M above 2048^2, try a 64x128 tile with an ordinary
        # non-cluster launch. On B200 this improved geomean throughput by 2.20% over
        # 81 unpadded shapes and 3.08% over nine +32 padded shapes (up to 6.84%);
        # retain the current 32x128 clustered path at 2048^2, where the change lost 3.1%.
        tile_m_size, tile_k_size = (
            _DIM_M_KM_SMALL_TILE_32_128
            if M * K <= 2048 * 2048
            else (
                _DIM_M_FLOAT32_LARGE_TILE_64_128
                if input.dtype == torch.float32
                else _DIM_M_LARGE_TILE_64_256
            )
        )
        padded_M = ncb_m * 128
        padded_K = _ceil_div(K, tile_k_size) * tile_k_size
        grid_m = padded_M // tile_m_size
        grid_k = padded_K // tile_k_size
        cluster_k = (
            next(c for c in (16, 8, 4, 2, 1) if c <= grid_k and grid_k % c == 0)
            if grid_m * grid_k <= 512
            else 1
        )
        needs_boundary_masking = M != ncb_m * 128 or K != nrb_m * 128
    else:
        tile_m_size, tile_k_size = (
            _DIM_M_KM_SMALL_TILE_32_128
            if M * K <= 2048 * 2048
            else _DIM_KM_LARGE_TILE_64_128
        )
        cluster_k = 1
        needs_boundary_masking = M != ncb_m * 128 or K != nrb_m * 128

    launch_grid_k = _ceil_div(K, tile_k_size)
    if is_nvfp4 and quant_orientation == "dim_m":
        launch_grid_m = _ceil_div(M, 64) * 64 // tile_m_size
    else:
        launch_grid_m = _ceil_div(M, 128) * (128 // tile_m_size)
    if launch_grid_k > _CUDA_GRID_X_MAX or launch_grid_m > _CUDA_GRID_Y_MAX:
        raise ValueError(
            "blockscaled TMA launch grid exceeds CUDA limits: "
            f"grid=({launch_grid_k}, {launch_grid_m}, 1), "
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
        tile_m_size,
        tile_k_size,
        cluster_k,
        needs_boundary_masking,
        quant_orientation_id,
        is_stochastic_qdata_rounding,
        is_square_scaling,
        qdata_dtype,
        scale_algo,
        has_dim_m_rht,
    )
    fn(
        input,
        output_k,
        scale_k,
        outer_scale_k_arg,
        output_m,
        scale_m,
        outer_scale_m_arg,
        rht_sign,
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
    rht_sign: torch.Tensor | None = None,
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
            rht_sign=rht_sign,
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
    rht_sign: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
    rounding_mode: str = "rtne",
    **kwargs,
):
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
        key=key,
        rounding_mode=rounding_mode,
        is_square_scaling=False,
        qdata_dtype=torch.float4_e2m1fn_x2,
        scale_algo=ScaleAlgo.NVFP4_FP8_E4M3,
        outer_scale_k=outer_scale_k_arg,
        outer_scale_m=outer_scale_m_arg,
        rht_sign=rht_sign,
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


def nvfp4_dim_m_rht_swizzle_tma(input, outer_scale, rht_sign, **kwargs):
    return nvfp4_swizzle_tma(
        input,
        outer_scale,
        mode="dim_m",
        rht_sign=rht_sign,
        **kwargs,
    )


NVFP4_DIM_M_RHT_SWIZZLE_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzleDimMRHTGold, cute_fn=nvfp4_dim_m_rht_swizzle_tma
)


def nvfp4_dim_m_swizzle_rht_sr_tma(
    input, outer_scale, rht_sign, key, **kwargs
):
    return nvfp4_swizzle_tma(
        input,
        outer_scale,
        mode="dim_m",
        rht_sign=rht_sign,
        key=key,
        rounding_mode="stochastic",
        **kwargs,
    )


NVFP4_DIM_M_SWIZZLE_RHT_SR_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsDimMSwizzleRHTSRGold,
    cute_fn=nvfp4_dim_m_swizzle_rht_sr_tma,
)


def nvfp4_swizzle_dim_k_dim_m_rht_tma(
    input, outer_scale_k, outer_scale_m, rht_sign, **kwargs
):
    return nvfp4_swizzle_tma(
        input,
        outer_scale_k,
        outer_scale_m=outer_scale_m,
        mode="dim_km",
        rht_sign=rht_sign,
        **kwargs,
    )


NVFP4_SWIZZLE_DIM_K_DIM_M_RHT_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzle_DimK_DimMRHT_Gold,
    cute_fn=nvfp4_swizzle_dim_k_dim_m_rht_tma,
)


def nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_tma(
    input,
    outer_scale_k,
    outer_scale_m,
    rht_sign,
    key,
    **kwargs,
):
    return nvfp4_swizzle_tma(
        input,
        outer_scale_k,
        outer_scale_m=outer_scale_m,
        mode="dim_km",
        rht_sign=rht_sign,
        key=key,
        rounding_mode="stochastic",
        **kwargs,
    )


NVFP4_SWIZZLE_DIM_K_SR_DIM_M_RHT_SR_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzle_DimKSR_DimMRHTSR_Gold,
    cute_fn=nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_tma,
)
