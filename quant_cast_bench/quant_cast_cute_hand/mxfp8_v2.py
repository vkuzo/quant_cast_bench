"""TMA-based MXFP8 v2 kernels and recipe definitions."""

from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

import torch

from quant_cast_bench.quant_cast_cute.recipes import (
    QuantCastCuteRecipe,
    _philox_4x32,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    Mxfp832x32SwizzleGold,
    Mxfp8DimKmSwizzleGold,
    Mxfp8DimKmSwizzleSRGold,
    Mxfp8DimMSwizzleGold,
    Mxfp8DimMSwizzleSRGold,
    Mxfp8SwizzleGold,
    Mxfp8SwizzleSRGold,
)
from quant_cast_bench.quant_cast_cute_hand.utils import (
    _ceil_div,
    _cvt_rs_satfinite_e4m3x4_f32,
    _e8m0,
    _e8m0_scale_store_as_uint,
)


_COMPILE_CACHE: dict = {}


def _compiled(key, jit_fn, *cute_args):
    fn = _COMPILE_CACHE.get(key)
    if fn is None:
        fn = cute.compile(jit_fn, *cute_args)
        _COMPILE_CACHE[key] = fn
    return fn


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


@cute.jit
def _mxfp8_v2_quantize_stochastic_x32(
    values: cute.TensorSSA,
    rcp: cutlass.Float32,
    counter_start: cutlass.Uint64,
    k0: cutlass.Uint32,
    k1: cutlass.Uint32,
) -> cute.TensorSSA:
    """Quantize one contiguous 32-value output run with two Philox counters."""
    scaled = values * rcp
    qwords = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Uint32)
    for half in cutlass.range_constexpr(2):
        ctr = counter_start + cutlass.Uint64(half)
        c0 = cutlass.Uint32(ctr & cutlass.Uint64(0xFFFFFFFF))
        c1 = cutlass.Uint32(ctr >> 32)
        zero = cutlass.Uint32(0)
        r0, r1, r2, r3 = _philox_4x32(c0, c1, zero, zero, k0, k1)
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


@cute.kernel
def mxfp8_swizzle_v2_kernel(
    input_tma_atom: cute.CopyAtom,
    input_tma_tensor: cute.Tensor,
    output_k_tma_atom: cute.CopyAtom | None,
    output_k_tma_tensor: cute.Tensor | None,
    output_m_tma_atom: cute.CopyAtom | None,
    output_m_tma_tensor: cute.Tensor | None,
    mScaleKLogical: cute.Tensor | None,
    mScaleMLogical: cute.Tensor | None,
    mSeed: cute.Tensor | None,
    input_smem_layout: cute.ComposedLayout,
    output_k_smem_layout: cute.ComposedLayout | None,
    output_m_smem_layout: cute.ComposedLayout | None,
    data_k_tv_layout: cute.Layout | None,
    input_element_type: cutlass.Constexpr,
    tile_m_size: cutlass.Constexpr,
    tile_k_size: cutlass.Constexpr,
    M: cutlass.Int32 | cutlass.Int64,
    K: cutlass.Int32 | cutlass.Int64,
    index_type: cutlass.Constexpr,
    needs_boundary_masking: cutlass.Constexpr,
    quant_orientation: cutlass.Constexpr,
    is_stochastic_qdata_rounding: cutlass.Constexpr,
    is_square_scaling: cutlass.Constexpr,
) -> None:
    """
    Kernel for mxfp8 quantization across (dim_k, dim_m, dim_km) x (RTNE, SR)

    High level flow:
      0. create smem scratchpad
         a. both dim-k and dim-m use a tile_m_size * tile_k_size input-typed scratchpad,
         b. dim-k reuses the first half of 0a for qdata (to save smem)
         c. dim-m additionally uses a tile_m_size * tile_k_size fp8 scratchpad for qdata
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
      5. syncthreads for dim-m and dim-k smem output writes
      6. if do_dim_k, kick off dim_k TMA qdata store
      7. if do_dim_m, kick off dim_m TMA qdata store
      8. if do_dim_k, store scales from registers to global memory

    """

    if cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_K):
        assert output_k_tma_atom is not None
        assert output_k_tma_tensor is not None
        assert mScaleKLogical is not None
        assert output_k_smem_layout is not None
        assert data_k_tv_layout is not None
        assert output_m_tma_atom is None
        assert output_m_tma_tensor is None
        assert mScaleMLogical is None
        assert output_m_smem_layout is None
    elif cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_M):
        assert output_k_tma_atom is None
        assert output_k_tma_tensor is None
        assert mScaleKLogical is None
        assert output_k_smem_layout is None
        assert data_k_tv_layout is None
        assert output_m_tma_atom is not None
        assert output_m_tma_tensor is not None
        assert mScaleMLogical is not None
        assert output_m_smem_layout is not None
    else:
        assert quant_orientation == _QUANT_ORIENTATION_DIM_KM
        assert output_k_tma_atom is not None
        assert output_k_tma_tensor is not None
        assert mScaleKLogical is not None
        assert output_k_smem_layout is not None
        assert data_k_tv_layout is not None
        assert output_m_tma_atom is not None
        assert output_m_tma_tensor is not None
        assert mScaleMLogical is not None
        assert output_m_smem_layout is not None

    if cutlass.const_expr(is_stochastic_qdata_rounding):
        assert mSeed is not None
    else:
        assert mSeed is None

    # bookkeeping
    tidx, _, _ = cute.arch.thread_idx()
    tile_k_idx_i32, tile_m_idx_i32, _ = cute.arch.block_idx()
    tile_k_idx = index_type(tile_k_idx_i32)
    tile_m_idx = index_type(tile_m_idx_i32)
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    do_dim_k = quant_orientation != _QUANT_ORIENTATION_DIM_M
    do_dim_m = quant_orientation != _QUANT_ORIENTATION_DIM_K
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
            cutlass.Float8E4M3FN, tile_m_size * tile_k_size, byte_alignment=1024
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
                dtype=cutlass.Float8E4M3FN,
            ),
            output_k_smem_layout.outer,
        )
        gOutputK = cute.local_tile(
            output_k_tma_tensor, (tile_m_size, tile_k_size), (None, None)
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
                dtype=cutlass.Float8E4M3FN,
            ),
            output_m_smem_layout.outer,
        )
        gOutputM = cute.local_tile(
            output_m_tma_tensor, (tile_k_size, tile_m_size), (None, None)
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

    if cutlass.const_expr(is_stochastic_qdata_rounding):
        # The Philox key is independent of the tile, so load and unpack it while TMA fills sInput.
        frgKey = cute.make_rmem_tensor(cute.make_layout(2), mSeed.element_type)
        cute.copy(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mSeed.element_type),
            mSeed,
            frgKey,
        )
        key64 = cute.recast_tensor(frgKey, dtype=cutlass.Uint64)
        k0 = cutlass.Uint32(key64[0] & cutlass.Uint64(0xFFFFFFFF))
        k1 = cutlass.Uint32(key64[0] >> 32)
        counter_base = key64[1]

    # wait for input data to arrive
    cute.arch.mbarrier_wait(tma_bar_ptr, 0)

    smem_load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), input_element_type, num_bits_per_copy=128
    )
    smem_store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=128
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
        output_row_m = tile_k_idx * tile_k_size + tidx
        scale_col_m = tile_m_idx * row_blocks
        rScaleM = cute.make_rmem_tensor(row_blocks, cutlass.Uint8)
        if cutlass.const_expr(is_stochastic_qdata_rounding):
            # TODO(future): currently both the dim_m and dim_k flat element indices
            # have range [0, M * K). This is fine for dim_m and dim_k in isolation, but
            # reuses each index twice for dim_km. We should fix this by offsetting dim_m
            # indices by M * K when both dim_m and dim_k are on. Just haven't gotten
            # to it yet, will require changes to gold and tests to maintain bitwise
            # equivalence vs kernel.
            sr_flat_base_m = (
                (tile_k_idx * tile_k_size + tidx) * M + tile_m_idx * tile_m_size
            )
        for row_block in cutlass.range_constexpr(row_blocks):
            if cutlass.const_expr(is_stochastic_qdata_rounding):
                sr_counter_start_m = counter_base + cutlass.Uint64(
                    (sr_flat_base_m + row_block * 32) // 16
                )
            rInputM = cute.make_rmem_tensor(32, cutlass.Float32)
            for row in cutlass.range_constexpr(32):
                rInputM[row] = sInput[row_block * 32 + row, tidx].to(
                    cutlass.Float32
                )
            vm = rInputM.load()
            amax_m = cute.math.absf(vm).reduce(
                cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
            )
            rcp_m, biased_m = _e8m0(amax_m)
            rQM = cute.make_rmem_tensor(32, cutlass.Float8E4M3FN)
            if cutlass.const_expr(is_stochastic_qdata_rounding):
                rQM.store(
                    _mxfp8_v2_quantize_stochastic_x32(
                        vm, rcp_m, sr_counter_start_m, k0, k1
                    )
                )
            else:
                rQM.store((vm * rcp_m).to(cutlass.Float8E4M3FN))
            rQM16 = cute.tiled_divide(rQM, (16,))
            sQM16 = cute.tiled_divide(sOutputM[(tidx, None)], (16,))
            for vec in cutlass.range_constexpr(2):
                cute.copy(
                    smem_store_atom,
                    rQM16[(None, vec)],
                    sQM16[(None, row_block * 2 + vec)],
                )
            rScaleM[row_block] = biased_m.to(rScaleM.element_type)

        use_full_tile_m = cutlass.const_expr(not needs_boundary_masking) or (
            ((tile_m_idx + 1) * tile_m_size <= M) & ((tile_k_idx + 1) * tile_k_size <= K)
        )
        if use_full_tile_m:
            _e8m0_scale_store_as_uint(
                mScaleMLogical,
                rScaleM,
                output_row_m,
                scale_col_m,
                row_blocks,
            )
        else:
            rScaleMPadded = cute.make_rmem_tensor(row_blocks, cutlass.Uint8)
            # pad unused scale entries with zeros
            rScaleMPadded.fill(0)
            if output_row_m < K:
                for row_block in cutlass.range_constexpr(row_blocks):
                    if tile_m_idx * row_blocks + row_block < M // 32:
                        rScaleMPadded[row_block] = rScaleM[row_block]
            if output_row_m < _ceil_div(K, 128) * 128:
                _e8m0_scale_store_as_uint(
                    mScaleMLogical,
                    rScaleMPadded,
                    output_row_m,
                    scale_col_m,
                    row_blocks,
                )

    if cutlass.const_expr(do_dim_k):
        # dim-k main pass (for dim_k and dim_km)
        # * read input data from smem and quantize in registers
        # * keep scale to registers
        # * store qdata in smem, overwriting the first half of input data (to save smem)
        bpr = tile_k_size // 32
        row_owned_k = tile_m_size == _DIM_K_TILE_M_SIZE_128
        iters = bpr if cutlass.const_expr(row_owned_k) else tile_m_size // 32
        tidfrgInputK = cute.composition(sInput, data_k_tv_layout)
        tidfrgOutputK = cute.composition(sOutputK, data_k_tv_layout)
        thrInputGroupsK = tidfrgInputK[(tidx, None)]
        thrOutputGroupsK = tidfrgOutputK[(tidx, None)]
        rScaleK = cute.make_rmem_tensor(iters, cutlass.Uint8)
        rQK = cute.make_rmem_tensor(
            cute.make_layout((32, iters), stride=(1, 32)),
            cutlass.Float8E4M3FN,
        )
        if cutlass.const_expr(is_stochastic_qdata_rounding):
            if cutlass.const_expr(row_owned_k):
                sr_flat_base_k = (
                    (tile_m_idx * tile_m_size + tidx) * K + tile_k_idx * tile_k_size
                )
            else:
                sr_flat_base_k = (
                    (tile_m_idx * tile_m_size + tidx // bpr) * K
                    + (tile_k_idx * bpr + tidx % bpr) * 32
                )
        for it in cutlass.range_constexpr(iters):
            if cutlass.const_expr(is_stochastic_qdata_rounding):
                if cutlass.const_expr(row_owned_k):
                    flat_start_k = sr_flat_base_k + it * 32
                else:
                    flat_start_k = sr_flat_base_k + it * 32 * K
                sr_counter_start_k = counter_base + cutlass.Uint64(
                    flat_start_k // 16
                )
            sGroupK = thrInputGroupsK[((None, it),)]
            rInputK = cute.make_rmem_tensor(32, input_element_type)
            input_values_per_copy = 128 // input_element_type.width
            sGroupKVec = cute.tiled_divide(sGroupK, (input_values_per_copy,))
            rInputKVec = cute.tiled_divide(rInputK, (input_values_per_copy,))
            for vec in cutlass.range_constexpr(32 // input_values_per_copy):
                cute.copy(
                    smem_load_atom,
                    sGroupKVec[(None, vec)],
                    rInputKVec[(None, vec)],
                )
            vk = rInputK.load().to(cutlass.Float32)
            amax_k = cute.math.absf(vk).reduce(
                cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
            )
            if cutlass.const_expr(is_square_scaling):
                # One warp owns the 32 rows of each 32-column group. Broadcast their combined
                # maximum so every row quantizes with the same 32x32-block scale.
                amax_k = cute.arch.warp_reduction_max(amax_k)
            rcp_k, biased_k = _e8m0(amax_k)
            if cutlass.const_expr(is_stochastic_qdata_rounding):
                rQK[(None, it)].store(
                    _mxfp8_v2_quantize_stochastic_x32(
                        vk, rcp_k, sr_counter_start_k, k0, k1
                    )
                )
            else:
                # Keep the original dim-K RTNE conversion unchanged in its specialization.
                rQK[(None, it)].store(
                    (vk * rcp_k).to(cutlass.Float8E4M3FN)
                )
            rScaleK[it] = biased_k.to(rScaleK.element_type)

        # All input reads must finish before the aliased qK view overwrites the input tile.
        cute.arch.sync_threads()
        for it in cutlass.range_constexpr(iters):
            rQK16 = cute.tiled_divide(rQK[(None, it)], (16,))
            sGroupK16 = cute.tiled_divide(
                thrOutputGroupsK[((None, it),)], (16,)
            )
            for vec in cutlass.range_constexpr(2):
                cute.copy(
                    smem_store_atom,
                    rQK16[(None, vec)],
                    sGroupK16[(None, vec)],
                )

    # Publish all enabled qdata writes to the async shared-memory proxy and wait for every thread
    # before launching the enabled qdata TMA store(s).
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
        output_m_warp = 1 if cutlass.const_expr(is_stochastic_qdata_rounding and do_dim_k) else 0
        if warp == output_m_warp:
            cute.copy(
                output_m_tma_atom,
                tOutputsM,
                tOutputgM[(None, tile_k_idx, tile_m_idx)],
            )

    if cutlass.const_expr(do_dim_k):
        # do the dim-k scale write (overlaps with qdata TMA store)
        use_full_tile_k = cutlass.const_expr(not needs_boundary_masking) or (
            ((tile_m_idx + 1) * tile_m_size <= M) & ((tile_k_idx + 1) * tile_k_size <= K)
        )
        if cutlass.const_expr(row_owned_k):
            input_row_k = tile_m_idx * tile_m_size + tidx
            scale_col_k = tile_k_idx * bpr
            if use_full_tile_k:
                _e8m0_scale_store_as_uint(
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
                        if tile_k_idx * bpr + it < K // 32:
                            rScaleKPadded[it] = rScaleK[it]
                _e8m0_scale_store_as_uint(
                    mScaleKLogical,
                    rScaleKPadded,
                    input_row_k,
                    scale_col_k,
                    iters,
                )

            if cutlass.const_expr(needs_boundary_masking):
                ncb_k = _ceil_div(K, 128)
                grid_n = _ceil_div(K, tile_k_size)
                covered_groups = grid_n * bpr
                if covered_groups < ncb_k * 4:
                    if tile_k_idx == grid_n - 1:
                        input_row_k = tile_m_idx * tile_m_size + tidx
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
                local_group_k = tidx % bpr
                local_row_k = tidx // bpr
            for it in cutlass.range_constexpr(iters):
                input_row_k = tile_m_idx * tile_m_size + local_row_k + it * 32
                scale_col_k = tile_k_idx * bpr + local_group_k
                if use_full_tile_k:
                    mScaleKLogical[(input_row_k, scale_col_k)] = rScaleK[it]
                else:
                    scale_k = cutlass.Uint8(0)
                    if input_row_k < M:
                        if scale_col_k < K // 32:
                            scale_k = rScaleK[it]
                    mScaleKLogical[(input_row_k, scale_col_k)] = scale_k


@cute.jit
def mxfp8_swizzle_v2_jit(
    mInput: cute.Tensor,
    mOutputK: cute.Tensor | None,
    mScaleK: cute.Tensor | None,
    mOutputM: cute.Tensor | None,
    mScaleM: cute.Tensor | None,
    mSeed: cute.Tensor | None,
    M: cutlass.Int32 | cutlass.Int64,
    K: cutlass.Int32 | cutlass.Int64,
    index_type: cutlass.Constexpr,
    tile_m_size: cutlass.Constexpr,
    tile_k_size: cutlass.Constexpr,
    cluster_k: cutlass.Constexpr,
    needs_boundary_masking: cutlass.Constexpr,
    quant_orientation: cutlass.Constexpr,
    is_stochastic_qdata_rounding: cutlass.Constexpr,
    is_square_scaling: cutlass.Constexpr,
) -> None:
    do_dim_k = quant_orientation != _QUANT_ORIENTATION_DIM_M
    do_dim_m = quant_orientation != _QUANT_ORIENTATION_DIM_K

    if cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_K):
        assert mOutputK is not None
        assert mScaleK is not None
        assert mOutputM is None
        assert mScaleM is None
    elif cutlass.const_expr(quant_orientation == _QUANT_ORIENTATION_DIM_M):
        assert mOutputK is None
        assert mScaleK is None
        assert mOutputM is not None
        assert mScaleM is not None
    else:
        assert quant_orientation == _QUANT_ORIENTATION_DIM_KM
        assert mOutputK is not None
        assert mScaleK is not None
        assert mOutputM is not None
        assert mScaleM is not None

    if cutlass.const_expr(is_stochastic_qdata_rounding):
        assert mSeed is not None
    else:
        assert mSeed is None
    if cutlass.const_expr(is_square_scaling):
        assert quant_orientation == _QUANT_ORIENTATION_DIM_K

    padded_M = _ceil_div(M, 128) * 128
    padded_K = _ceil_div(K, tile_k_size) * tile_k_size

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
        input_smem_kind = (
            tcgen05.SmemLayoutAtomKind.K_SW64
            if cutlass.const_expr(
                quant_orientation == _QUANT_ORIENTATION_DIM_K and tile_k_size == 32
            )
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
            output_k_smem_kind, cutlass.Float8E4M3FN
        )
        output_k_smem_layout = cute.coalesce(
            cute.tile_to_shape(
                output_k_smem_atom, (tile_m_size, tile_k_size), order=(0, 1)
            ),
            target_profile=(1, 1),
        )
        output_k_tma_atom, output_k_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mOutputK,
            output_k_smem_layout,
            (tile_m_size, tile_k_size),
        )

        nrb_k = _ceil_div(M, 128)
        ncb_k = _ceil_div(K, 128)
        scale_k_layout = cute.make_layout(
            ((32, 4, nrb_k), (4, ncb_k)),
            stride=((16, 4, ncb_k * 32 * 16), (1, 32 * 16)),
        )
        mScaleKLogical = cute.make_tensor(mScaleK.iterator, scale_k_layout)

        bpr = tile_k_size // 32
        if cutlass.const_expr(tile_m_size == _DIM_K_TILE_M_SIZE_128):
            data_k_tv_layout = cute.make_layout(
                ((tile_m_size,), (32, bpr)),
                stride=((1,), (tile_m_size, tile_m_size * 32)),
            )
        elif cutlass.const_expr(is_square_scaling):
            data_k_tv_layout = cute.make_layout(
                ((32, bpr), (32, 1)),
                stride=((1, tile_m_size * 32), (tile_m_size, tile_m_size * tile_k_size)),
            )
        else:
            rows_per_stage = _MIN_CTA_THREADS_128 // bpr
            row_blocks = tile_m_size // rows_per_stage
            data_k_tv_layout = cute.make_layout(
                ((bpr, rows_per_stage), (32, row_blocks)),
                stride=((tile_m_size * 32, 1), (tile_m_size, rows_per_stage)),
            )
    else:
        output_k_smem_layout = None
        output_k_tma_atom = None
        output_k_tma_tensor = None
        mScaleKLogical = None
        data_k_tv_layout = None

    if cutlass.const_expr(do_dim_m):
        output_m_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(0, 0, 0),
            0,
            cute.make_layout((tile_k_size, tile_m_size), stride=(tile_m_size, 1)),
        )
        output_m_tma_atom, output_m_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mOutputM,
            output_m_smem_layout,
            (tile_k_size, tile_m_size),
        )

        nrb_m = _ceil_div(K, 128)
        ncb_m = _ceil_div(M, 128)
        scale_m_layout = cute.make_layout(
            ((32, 4, nrb_m), (4, ncb_m)),
            stride=((16, 4, ncb_m * 32 * 16), (1, 32 * 16)),
        )
        mScaleMLogical = cute.make_tensor(mScaleM.iterator, scale_m_layout)
    else:
        output_m_smem_layout = None
        output_m_tma_atom = None
        output_m_tma_tensor = None
        mScaleMLogical = None

    kernel = mxfp8_swizzle_v2_kernel(
        input_tma_atom,
        input_tma_tensor,
        output_k_tma_atom,
        output_k_tma_tensor,
        output_m_tma_atom,
        output_m_tma_tensor,
        mScaleKLogical,
        mScaleMLogical,
        mSeed,
        input_smem_layout,
        output_k_smem_layout,
        output_m_smem_layout,
        data_k_tv_layout,
        mInput.element_type,
        tile_m_size,
        tile_k_size,
        M,
        K,
        index_type,
        needs_boundary_masking,
        quant_orientation,
        is_stochastic_qdata_rounding,
        is_square_scaling,
    )
    grid = (padded_K // tile_k_size, padded_M // tile_m_size, 1)
    block = (max(_MIN_CTA_THREADS_128, tile_k_size), 1, 1)
    if cutlass.const_expr(
        quant_orientation == _QUANT_ORIENTATION_DIM_KM and not is_stochastic_qdata_rounding and tile_m_size != 32
    ):
        # A degenerate cluster constrains residency for this larger two-output specialization;
        # an ordinary launch lets Blackwell keep nine CTAs resident per SM instead.
        launch_cluster = None
    else:
        # K-major scheduling keeps adjacent row-major input columns together.
        launch_cluster = (cluster_k, 1, 1)
    kernel.launch(grid=grid, block=block, cluster=launch_cluster)


def _mxfp8_swizzle_v2_impl(
    input: torch.Tensor,
    quant_orientation: str,
    key: torch.Tensor | None,
    rounding_mode: str,
    is_square_scaling: bool,
    **kwargs,
):
    assert quant_orientation in (
        "dim_k",
        "dim_m",
        "dim_km",
    ), f"unsupported quant_orientation: {quant_orientation}"
    assert input.dim() == 2, "unsupported"
    assert input.is_contiguous(), "unsupported"
    assert input.dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ), "v2 supports only bf16, fp16, and fp32 input"

    rounding_mode = str(getattr(rounding_mode, "value", rounding_mode)).lower()
    assert rounding_mode in (
        "rtne",
        "stochastic",
    ), f"unsupported rounding_mode: {rounding_mode}"
    is_stochastic_qdata_rounding = rounding_mode == "stochastic"
    if is_square_scaling:
        assert quant_orientation == "dim_k", "32x32 v2 currently supports only dim-k output"
        assert not is_stochastic_qdata_rounding, "32x32 v2 currently supports only RTNE"
    if is_stochastic_qdata_rounding:
        assert key is not None, "stochastic rounding requires a Philox key"
        assert key.device == input.device, "input and Philox key must be on the same device"
        assert key.dtype == torch.uint64 and key.numel() == 2, "Philox key must be uint64[2]"
    else:
        assert key is None, "RTNE rounding does not use a Philox key"

    M, K = input.shape
    assert M > 0 and K > 0, "v2 requires non-empty dimensions"
    if is_square_scaling:
        assert M % 32 == 0, "32x32 v2 requires M % 32 == 0"
    if quant_orientation != "dim_k":
        assert M % 32 == 0, "v2 dim-M requires M % 32 == 0"
        assert K % 16 == 0, "v2 dim-M requires K % 16 == 0"
        if quant_orientation == "dim_km":
            assert K % 32 == 0, "v2 dim-K requires K % 32 == 0"
    else:
        assert K % 32 == 0, "v2 requires K % 32 == 0"

    do_dim_k = quant_orientation != "dim_m"
    do_dim_m = quant_orientation != "dim_k"

    nrb_k = ncb_k = None
    if do_dim_k:
        nrb_k = _ceil_div(M, 128)
        ncb_k = _ceil_div(K // 32, 4)

    nrb_m = ncb_m = None
    if do_dim_m:
        nrb_m = _ceil_div(K, 128)
        ncb_m = _ceil_div(M // 32, 4)

    if quant_orientation == "dim_k":
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
        quant_orientation_id = _QUANT_ORIENTATION_DIM_K
        compile_key = (
            "mxfp8_swizzle_v2",
            "dim_k",
            tile_m_size,
            tile_k_size,
            cluster_k,
            needs_boundary_masking,
            rounding_mode,
            is_square_scaling,
        )
    elif quant_orientation == "dim_m":
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
        quant_orientation_id = _QUANT_ORIENTATION_DIM_M
        compile_key = (
            "mxfp8_swizzle_v2",
            "dim_m",
            tile_m_size,
            tile_k_size,
            cluster_k,
            needs_boundary_masking,
            rounding_mode,
        )
    else:
        tile_m_size, tile_k_size = (
            _DIM_M_KM_SMALL_TILE_32_128
            if M * K <= 2048 * 2048
            else _DIM_KM_LARGE_TILE_64_128
        )
        cluster_k = 1
        needs_boundary_masking = M != ncb_m * 128 or K != nrb_m * 128
        quant_orientation_id = _QUANT_ORIENTATION_DIM_KM
        compile_key = (
            "mxfp8_swizzle_v2",
            "dim_km",
            tile_m_size,
            tile_k_size,
            cluster_k,
            needs_boundary_masking,
            rounding_mode,
        )

    index_type = cutlass.Int64 if input.numel() > _INT32_MAX else cutlass.Int32
    compile_key = (*compile_key, input.dtype, index_type)

    output_m = scale_m = mOutputM = mScaleM = None
    if do_dim_m:
        output_m = torch.empty(K, M, dtype=torch.float8_e4m3fn, device=input.device)
        scale_m = torch.empty(
            nrb_m * ncb_m * 32 * 16,
            dtype=torch.uint8,
            device=input.device,
        )
        mOutputM = (
            from_dlpack(output_m, assumed_align=16)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=16)
        )
        # Only the scale length varies; its compact byte stride and 512-byte block divisibility do not.
        mScaleM = (
            from_dlpack(scale_m, assumed_align=4)
            .mark_layout_dynamic(leading_dim=0)
            .mark_compact_shape_dynamic(mode=0, divisibility=512)
        )

    output_k = scale_k = mOutputK = mScaleK = None
    if do_dim_k:
        output_k = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=input.device)
        # Every slot is written by the kernel, so zero-initialization would launch a redundant
        # memset.
        scale_k = torch.empty(
            nrb_k * ncb_k * 32 * 16,
            dtype=torch.uint8,
            device=input.device,
        )
        mOutputK = (
            from_dlpack(output_k, assumed_align=16)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=16)
        )
        # Keep the scale allocation out of the compile key while preserving packed-store alignment.
        mScaleK = (
            from_dlpack(scale_k, assumed_align=4)
            .mark_layout_dynamic(leading_dim=0)
            .mark_compact_shape_dynamic(mode=0, divisibility=512)
        )

    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-element aligned).
    mInput = (
        from_dlpack(input, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )
    mSeed = (
        from_dlpack(key.reshape(-1).view(torch.int64))
        if is_stochastic_qdata_rounding
        else None
    )

    fn = _compiled(
        compile_key,
        mxfp8_swizzle_v2_jit,
        mInput,
        mOutputK,
        mScaleK,
        mOutputM,
        mScaleM,
        mSeed,
        index_type(M),
        index_type(K),
        index_type,
        tile_m_size,
        tile_k_size,
        cluster_k,
        needs_boundary_masking,
        quant_orientation_id,
        is_stochastic_qdata_rounding,
        is_square_scaling,
    )
    fn(mInput, mOutputK, mScaleK, mOutputM, mScaleM, mSeed, M, K)

    if do_dim_m:
        scale_m = scale_m.view(nrb_m, ncb_m, 32, 16).view(torch.float8_e8m0fnu)
    if do_dim_k:
        scale_k = scale_k.view(nrb_k, ncb_k, 32, 16).view(torch.float8_e8m0fnu)

    if quant_orientation == "dim_k":
        return output_k, scale_k
    if quant_orientation == "dim_m":
        return output_m, scale_m
    return output_k, scale_k, output_m, scale_m


def mxfp8_swizzle_v2(
    input: torch.Tensor,
    quant_orientation: str = "dim_k",
    key: torch.Tensor | None = None,
    rounding_mode: str = "rtne",
    **kwargs,
):
    return _mxfp8_swizzle_v2_impl(
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
    return _mxfp8_swizzle_v2_impl(
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
