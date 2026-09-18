"""CuTe DSL kernels and compilation for TMA-based block-scaled quantization."""

from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache
from torch._vendor.quack.cache import EXTRA_SOURCE_DIRS

from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma.blockscale_tma_plan import (
    ScaleAlgo,
)
from quant_cast_bench.quant_cast_cute_hand.utils import (
    _blockscaled_quantize_group,
    _ceil_div,
    _nvfp4_load_philox_key,
    _store_swizzled_scale_groups_as_uint,
)

# Include this prototype's Python sources in QuACK's persistent-cache fingerprint. This must
# happen before the process's first jit_cache lookup, when that fingerprint is memoized.
_BLOCKSCALED_TMA_SOURCE_DIR = Path(__file__).resolve().parent.parent
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
_MIN_CTA_WARPS_4 = 4
_MIN_CTA_THREADS_128 = _MIN_CTA_WARPS_4 * 32

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

        MXFP8 supports RTNE and SR, while MXFP4 and NVFP4 currently support RTNE only.

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
                scale_row_m_in_bounds = output_row_m < _ceil_div(K, 128) * 128
                if cutlass.const_expr(
                    scale_algo == ScaleAlgo.NVFP4_FP8_E4M3 and do_dim_k
                ):
                    # Dim-KM pads its CTA grid to 128 elements for the dim-K scale rows,
                    # while NVFP4's dim-M scale columns are padded only to 4x16=64.
                    scale_col_m_in_bounds = (
                        scale_col_m // 4 * (scale_group_size * 4) < M
                    )
                    scale_row_m_in_bounds = (
                        scale_row_m_in_bounds & scale_col_m_in_bounds
                    )
                if scale_row_m_in_bounds:
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
                    if cutlass.const_expr(
                        scale_algo == ScaleAlgo.NVFP4_FP8_E4M3 and do_dim_m
                    ):
                        # Dim-KM pads its K tile to 128 elements, but an NVFP4 scale
                        # allocation contains only the required 4x16=64-element blocks.
                        scale_col_k_in_bounds = (
                            scale_col_k // 4 * (scale_group_size * 4) < K
                        )
                        if scale_col_k_in_bounds:
                            _store_swizzled_scale_groups_as_uint(
                                mScaleKLogical,
                                rScaleKPadded,
                                input_row_k,
                                scale_col_k,
                                iters,
                            )
                    else:
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
        mSeed: cute.Tensor | None,
        stream: cuda.CUstream,
        M: cutlass.Int32,
        K: cutlass.Int32,
        grid_m: cutlass.Int32,
        grid_k: cutlass.Int32,
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
) -> str:
    return (
        f"dtype={input_dtype} orientation={quant_orientation} "
        f"tile={tile_m_size}x{tile_k_size} cluster_k={cluster_k} "
        f"masking={needs_boundary_masking} sr={is_stochastic_qdata_rounding} "
        f"square={is_square_scaling} qdata_dtype={qdata_dtype} "
        f"scale_algo={scale_algo.name}"
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
):
    if qdata_dtype not in _TORCH_TO_CUTE_QDATA_DTYPE:
        raise ValueError(f"unsupported qdata dtype: {qdata_dtype}")
    input_element_type = _TORCH_TO_CUTE_DTYPE[input_dtype]
    qdata_element_type = _TORCH_TO_CUTE_QDATA_DTYPE[qdata_dtype]
    if scale_algo == ScaleAlgo.NVFP4_FP8_E4M3:
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
        mSeed,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        options="--enable-tvm-ffi",
    )
