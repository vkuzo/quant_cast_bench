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


_MXS_TM, _MXS_MAX_TN, _MXS_WARPS = 128, 128, 4
_MXS_THREADS = _MXS_WARPS * 32                       # 128, one thread per tile row
_MXS_MODE_DIM_K = 0
_MXS_MODE_DIM_M = 1
_MXS_MODE_DIM_KM = 2


def _mxfp8_swizzle_v2_tile_n(M, K):
    """Choose the measured B200 K tile from the padded 128x128 CTA count."""
    num_128_tiles = _ceil_div(M, _MXS_TM) * _ceil_div(K, _MXS_MAX_TN)
    if num_128_tiles <= 64:
        tile_k_size = 32
    elif num_128_tiles <= 512:
        tile_k_size = 64
    else:
        tile_k_size = 128

    # Do not compute padding merely to reach the selected width for very narrow matrices.
    if K <= 32:
        return 32
    if K <= 64:
        return min(tile_k_size, 64)
    return tile_k_size


@cute.jit
def _mxfp8_v2_quantize_stochastic_x32(
    values, rcp, counter_start, k0, k1
):
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
    output_k_tma_atom: cute.CopyAtom,
    output_k_tma_tensor: cute.Tensor,
    output_m_tma_atom: cute.CopyAtom,
    output_m_tma_tensor: cute.Tensor,
    mScaleKLogical: cute.Tensor,
    mScaleMLogical: cute.Tensor,
    mSeed: cute.Tensor,
    input_smem_layout: cute.ComposedLayout,
    output_k_smem_layout: cute.ComposedLayout,
    output_m_smem_layout: cute.ComposedLayout,
    data_k_tv_layout: cute.Layout,
    tile_m_size: cutlass.Constexpr,
    tile_k_size: cutlass.Constexpr,
    M: cutlass.Int32,
    K: cutlass.Int32,
    ragged: cutlass.Constexpr,
    mode: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
    square_scaling: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    tile_k_idx, tile_m_idx, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    do_dim_k = mode != _MXS_MODE_DIM_M
    do_dim_m = mode != _MXS_MODE_DIM_K
    if cutlass.const_expr(not ragged):
        M = cute.assume(M, divby=128)
        K = cute.assume(K, divby=128)
    elif cutlass.const_expr(mode == _MXS_MODE_DIM_K):
        K = cute.assume(K, divby=32)
    elif cutlass.const_expr(mode == _MXS_MODE_DIM_M):
        M = cute.assume(M, divby=32)
        K = cute.assume(K, divby=16)
    else:
        M = cute.assume(M, divby=32)
        K = cute.assume(K, divby=32)
    smem = utils.SmemAllocator()
    input_storage = smem.allocate_array(
        cutlass.BFloat16, tile_m_size * tile_k_size, byte_alignment=1024
    )
    if cutlass.const_expr(do_dim_m):
        output_m_storage = smem.allocate_array(
            cutlass.Float8E4M3FN, tile_m_size * tile_k_size, byte_alignment=1024
        )
    # Put the small barrier after the 1KB-aligned tile buffers to avoid an otherwise unused 1016B
    # alignment gap at the front of every CTA's shared-memory allocation.
    tma_bar_ptr = smem.allocate_array(cutlass.Int64, 1)

    if tidx == 0:
        cute.arch.mbarrier_init(tma_bar_ptr, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # All modes share one input TMA load. Dim-K aliases the input buffer only after every enabled
    # pass has finished reading the BF16 tile.
    sInput = cute.make_tensor(
        cute.recast_ptr(
            input_storage, input_smem_layout.inner, dtype=cutlass.BFloat16
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

    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                tma_bar_ptr, tile_m_size * tile_k_size * 2
            )
        cute.copy(
            input_tma_atom,
            tInputgInput[(None, tile_m_idx, tile_k_idx)],
            tInputsInput,
            tma_bar_ptr=tma_bar_ptr,
        )
    if cutlass.const_expr(stochastic):
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
    cute.arch.mbarrier_wait(tma_bar_ptr, 0)

    smem_load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    smem_store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=128
    )

    # Each dim-M thread owns one input column and processes every 32-row group. Its transposed
    # qdata uses separate shared memory, so the optional dim-K pass can still read sInput.
    if cutlass.const_expr(do_dim_m):
        row_blocks = tile_m_size // 32
        output_row_m = tile_k_idx * tile_k_size + tidx
        scale_col_m = tile_m_idx * row_blocks
        rScaleM = cute.make_rmem_tensor(row_blocks, cutlass.Uint8)
        if cutlass.const_expr(stochastic):
            sr_flat_base_m = (
                (tile_k_idx * tile_k_size + tidx) * M + tile_m_idx * tile_m_size
            )
        for row_block in cutlass.range_constexpr(row_blocks):
            if cutlass.const_expr(stochastic):
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
            if cutlass.const_expr(stochastic):
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

        use_full_tile_m = cutlass.const_expr(not ragged) or (
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

    # This is the original row-owned v2 dim-K pass. Modes dim_k and dim_km execute the same code;
    # the ownership layout changes for shorter fused tiles so all 128 threads remain useful.
    if cutlass.const_expr(do_dim_k):
        bpr = tile_k_size // 32
        row_owned_k = tile_m_size == _MXS_TM
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
        if cutlass.const_expr(stochastic):
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
            if cutlass.const_expr(stochastic):
                if cutlass.const_expr(row_owned_k):
                    flat_start_k = sr_flat_base_k + it * 32
                else:
                    flat_start_k = sr_flat_base_k + it * 32 * K
                sr_counter_start_k = counter_base + cutlass.Uint64(
                    flat_start_k // 16
                )
            sGroupK = thrInputGroupsK[((None, it),)]
            sGroupK8 = cute.tiled_divide(sGroupK, (8,))
            rInputK = cute.make_rmem_tensor(32, cutlass.BFloat16)
            rInputK8 = cute.tiled_divide(rInputK, (8,))
            for vec in cutlass.range_constexpr(4):
                cute.copy(
                    smem_load_atom,
                    sGroupK8[(None, vec)],
                    rInputK8[(None, vec)],
                )
            vk = rInputK.load().to(cutlass.Float32)
            amax_k = cute.math.absf(vk).reduce(
                cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
            )
            if cutlass.const_expr(square_scaling):
                # One warp owns the 32 rows of each 32-column group. Broadcast their combined
                # maximum so every row quantizes with the same 32x32-block scale.
                amax_k = cute.arch.warp_reduction_max(amax_k)
            rcp_k, biased_k = _e8m0(amax_k)
            if cutlass.const_expr(stochastic):
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

        # All BF16 reads must finish before the aliased qK view overwrites the input tile.
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

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_threads()
    if cutlass.const_expr(do_dim_k):
        if warp == 0:
            cute.copy(
                output_k_tma_atom,
                tOutputsK,
                tOutputgK[(None, tile_m_idx, tile_k_idx)],
            )
    if cutlass.const_expr(do_dim_m):
        # The fused SR specialization issues its independent output transfers from separate warps.
        output_m_warp = 1 if cutlass.const_expr(stochastic and do_dim_k) else 0
        if warp == output_m_warp:
            cute.copy(
                output_m_tma_atom,
                tOutputsM,
                tOutputgM[(None, tile_k_idx, tile_m_idx)],
            )

    # Let the qdata TMA store overlap the much smaller direct scale write.
    if cutlass.const_expr(do_dim_k):
        use_full_tile_k = cutlass.const_expr(not ragged) or (
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

            if cutlass.const_expr(ragged):
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
            if cutlass.const_expr(square_scaling):
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
def mxfp8_swizzle_v2_dim_k_jit(
    mInput,
    mOutput,
    mScale,
    mSeed,
    M: cutlass.Int32,
    K: cutlass.Int32,
    tile_m_size: cutlass.Constexpr,
    tile_k_size: cutlass.Constexpr,
    cluster_k: cutlass.Constexpr,
    ragged: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
    square_scaling: cutlass.Constexpr,
):
    padded_M = _ceil_div(M, _MXS_TM) * _MXS_TM
    padded_N = _ceil_div(K, tile_k_size) * tile_k_size
    bpr = tile_k_size // 32
    iters = bpr
    # Match the swizzle width to the selected tile while keeping its logical shape unchanged.
    if cutlass.const_expr(tile_k_size == 32):
        input_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW64
        output_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW32
    elif cutlass.const_expr(tile_k_size == 64):
        input_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW128
        output_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW64
    else:
        input_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW128
        output_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW128
    input_smem_atom = tcgen05.make_smem_layout_atom(input_smem_kind, cutlass.BFloat16)
    input_smem_layout = cute.coalesce(
        cute.tile_to_shape(input_smem_atom, (tile_m_size, tile_k_size), order=(0, 1)),
        target_profile=(1, 1),
    )
    # The output phase aliases the same allocation but uses its own dtype-appropriate swizzle. Its
    # lifetime starts only after the bf16-read barrier, so the two interpretations cannot overlap.
    output_smem_atom = tcgen05.make_smem_layout_atom(output_smem_kind, cutlass.Float8E4M3FN)
    output_smem_layout = cute.coalesce(
        cute.tile_to_shape(output_smem_atom, (tile_m_size, tile_k_size), order=(0, 1)),
        target_profile=(1, 1),
    )
    if cutlass.const_expr(tile_m_size == _MXS_TM):
        # (thread, value) -> logical coordinate in the 128xN tile. Each thread owns one row.
        data_tv_layout = cute.make_layout(
            ((_MXS_THREADS,), (32, iters)),
            stride=((1,), (tile_m_size, tile_m_size * 32)),
        )
    elif cutlass.const_expr(square_scaling):
        # The 32x128 square-scale tile assigns one warp to each 32x32 block. Within a warp,
        # lanes own rows and values walk columns, so warp_reduction_max spans the whole block.
        data_tv_layout = cute.make_layout(
            ((32, bpr), (32, 1)),
            stride=((1, tile_m_size * 32), (tile_m_size, tile_m_size * tile_k_size)),
        )
    else:
        # Smaller tiles map one (row, 1x32 group) to each thread, then advance through 32-row stages.
        rows_per_stage = _MXS_THREADS // bpr
        row_blocks = tile_m_size // rows_per_stage
        data_tv_layout = cute.make_layout(
            ((bpr, rows_per_stage), (32, row_blocks)),
            stride=((tile_m_size * 32, 1), (tile_m_size, rows_per_stage)),
        )
    input_tma_atom, input_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mInput, input_smem_layout, (tile_m_size, tile_k_size))
    output_tma_atom, output_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mOutput, output_smem_layout, (tile_m_size, tile_k_size))

    nrb = _ceil_div(M, 128)
    ncb = _ceil_div(K, 128)
    # Manual scale indexing uses fewer registers, but this layout-based form is easier to follow.
    scale_layout = cute.make_layout(
        ((32, 4, nrb), (4, ncb)),
        stride=((16, 4, ncb * 32 * 16), (1, 32 * 16)),
    )
    mScaleLogical = cute.make_tensor(mScale.iterator, scale_layout)

    mxfp8_swizzle_v2_kernel(
        input_tma_atom,
        input_tma_tensor,
        output_tma_atom,
        output_tma_tensor,
        output_tma_atom,
        output_tma_tensor,
        mScaleLogical,
        mScaleLogical,
        mSeed,
        input_smem_layout,
        output_smem_layout,
        output_smem_layout,
        data_tv_layout,
        tile_m_size,
        tile_k_size,
        M,
        K,
        ragged,
        _MXS_MODE_DIM_K,
        stochastic,
        square_scaling,
    ).launch(
        # Make the fast-changing grid dimension follow contiguous columns. Clustering those CTAs
        # further preserves that locality in hardware scheduling.
        grid=(padded_N // tile_k_size, padded_M // tile_m_size, 1),
        block=(_MXS_THREADS, 1, 1),
        cluster=(cluster_k, 1, 1),
    )


def _mxfp8_swizzle_v2_impl(
    input: torch.Tensor,
    mode: str,
    key: torch.Tensor | None,
    rounding_mode: str,
    square_scaling: bool,
    **kwargs,
):
    assert mode in ("dim_k", "dim_m", "dim_km"), f"unsupported mode: {mode}"
    assert input.dim() == 2, "unsupported"
    assert input.is_contiguous(), "unsupported"
    assert input.dtype == torch.bfloat16, "v2 is bf16-only"
    rounding_mode = str(getattr(rounding_mode, "value", rounding_mode)).lower()
    assert rounding_mode in ("rtne", "stochastic"), f"unsupported rounding_mode: {rounding_mode}"
    stochastic = rounding_mode == "stochastic"
    if square_scaling:
        assert mode == "dim_k", "32x32 v2 currently supports only dim-k output"
        assert not stochastic, "32x32 v2 currently supports only RTNE"
    if stochastic:
        assert key is not None, "stochastic rounding requires a Philox key"
        assert key.device == input.device, "input and Philox key must be on the same device"
        assert key.dtype == torch.uint64 and key.numel() == 2, "Philox key must be uint64[2]"
    else:
        assert key is None, "RTNE rounding does not use a Philox key"
    M, K = input.shape
    assert M > 0 and K > 0, "v2 requires non-empty dimensions"
    if square_scaling:
        assert M % 32 == 0, "32x32 v2 requires M % 32 == 0"

    if mode != "dim_k":
        assert M % 32 == 0, "v2 dim-M requires M % 32 == 0"
        assert K % 16 == 0, "v2 dim-M requires K % 16 == 0"
        if mode == "dim_km":
            assert K % 32 == 0, "v2 dim-K requires K % 32 == 0"

        tile_m_size, tile_k_size = (
            _MXDMT_SMALL_TILE
            if M * K <= 2048 * 2048
            else (_MXDKMT_LARGE_TILE if mode == "dim_km" else _MXDMT_LARGE_TILE)
        )
        nrb_m, ncb_m = _ceil_div(K, 128), _ceil_div(M // 32, 4)
        padded_M = ncb_m * 128
        padded_N = _ceil_div(K, tile_k_size) * tile_k_size
        grid_n = padded_N // tile_k_size
        grid_m = padded_M // tile_m_size
        cluster_k = 1 if mode == "dim_km" else (
            next(c for c in (16, 8, 4, 2, 1) if c <= grid_n and grid_n % c == 0)
            if grid_m * grid_n <= 512
            else 1
        )

        output_m = torch.empty(
            K, M, dtype=torch.float8_e4m3fn, device=input.device
        )
        scale_m = torch.empty(
            nrb_m * ncb_m * 32 * 16, dtype=torch.uint8, device=input.device
        )
        mInput = (
            from_dlpack(input, assumed_align=16)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=16)
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
        if mode == "dim_km":
            nrb_k, ncb_k = _ceil_div(M, 128), _ceil_div(K // 32, 4)
            output_k = torch.empty(
                M, K, dtype=torch.float8_e4m3fn, device=input.device
            )
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
            mScaleK = (
                from_dlpack(scale_k, assumed_align=4)
                .mark_layout_dynamic(leading_dim=0)
                .mark_compact_shape_dynamic(mode=0, divisibility=512)
            )
        else:
            nrb_k, ncb_k = nrb_m, ncb_m
            output_k, scale_k = output_m, scale_m
            mOutputK = mOutputM
            mScaleK = mScaleM
        # The RTNE specialization never reads mSeed, so reuse mScaleM as a dummy argument.
        mSeed = from_dlpack(key.reshape(-1).view(torch.int64)) if stochastic else mScaleM

        mode_id = (
            _MXS_MODE_DIM_KM if mode == "dim_km" else _MXS_MODE_DIM_M
        )
        ragged = M != ncb_m * 128 or K != nrb_m * 128
        fn = _compiled(
            (
                "mxfp8_swizzle_v2", mode, tile_m_size, tile_k_size, cluster_k, ragged,
                rounding_mode,
            ),
            mxfp8_swizzle_v2_dim_m_or_km_jit,
            mInput,
            mOutputM,
            mScaleM,
            mOutputK,
            mScaleK,
            mSeed,
            M,
            K,
            tile_m_size,
            tile_k_size,
            cluster_k,
            ragged,
            mode_id,
            stochastic,
        )
        fn(mInput, mOutputM, mScaleM, mOutputK, mScaleK, mSeed, M, K)
        scale_m = scale_m.view(nrb_m, ncb_m, 32, 16).view(
            torch.float8_e8m0fnu
        )
        if mode == "dim_km":
            scale_k = scale_k.view(nrb_k, ncb_k, 32, 16).view(
                torch.float8_e8m0fnu
            )
            return output_k, scale_k, output_m, scale_m
        return output_m, scale_m

    assert K % 32 == 0, "v2 requires K % 32 == 0"
    ngc = K // 32
    nrb, ncb = _ceil_div(M, 128), _ceil_div(ngc, 4)
    # First choose the original adaptive K width. For small problems that would use K=64, rotate
    # the same-size 128x64 tile to 32x128: it keeps 128 one-group threads but gives TMA contiguous
    # rows and exposes more M-parallel CTAs.
    tile_k_size = _mxfp8_swizzle_v2_tile_n(M, K)
    if M * K <= 2048 * 2048 and tile_k_size >= 64:
        tile_m_size, tile_k_size = 32, 128
    else:
        tile_m_size = 128
    grid_n = _ceil_div(K, tile_k_size)
    # RTNE benefits from K-oriented clustering and its locality. Philox supplies enough arithmetic
    # latency hiding that independent CTAs are faster than forcing the same clustered schedule.
    cluster_k = (
        # Square scaling has enough independent CTAs at small/medium sizes that clustering only
        # constrains scheduling; large shapes retain v2's K-locality-oriented clusters.
        1 if stochastic or (square_scaling and M * K <= 4096 * 4096)
        else next(c for c in (16, 8, 4, 2, 1) if c <= ncb and grid_n % c == 0)
    )
    output = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=input.device)
    # Every slot is written by the kernel, so zero-initialization would launch a redundant memset.
    scale = torch.empty(nrb * ncb * 32 * 16, dtype=torch.uint8, device=input.device)
    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-elem aligned).
    mInput = (from_dlpack(input, assumed_align=16).mark_layout_dynamic(leading_dim=1)
              .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mOutput = (from_dlpack(output, assumed_align=16).mark_layout_dynamic(leading_dim=1)
               .mark_compact_shape_dynamic(mode=1, divisibility=16))
    # Keep the scale allocation out of the compile key while preserving packed-store alignment.
    mScale = (
        from_dlpack(scale, assumed_align=4)
        .mark_layout_dynamic(leading_dim=0)
        .mark_compact_shape_dynamic(mode=0, divisibility=512)
    )
    # The RTNE specialization never reads mSeed, so reuse mScale as a dummy argument on that path.
    mSeed = from_dlpack(key.reshape(-1).view(torch.int64)) if stochastic else mScale
    ragged = M != nrb * 128 or K != ncb * 128
    fn = _compiled(
        (
            "mxfp8_swizzle_v2", "dim_k", tile_m_size, tile_k_size, cluster_k, ragged,
            rounding_mode, square_scaling,
        ),
        mxfp8_swizzle_v2_dim_k_jit,
        mInput,
        mOutput,
        mScale,
        mSeed,
        M,
        K,
        tile_m_size,
        tile_k_size,
        cluster_k,
        ragged,
        stochastic,
        square_scaling,
    )
    fn(mInput, mOutput, mScale, mSeed, M, K)
    return output, scale.view(nrb, ncb, 32, 16).view(torch.float8_e8m0fnu)


def mxfp8_swizzle_v2(
    input: torch.Tensor,
    mode: str = "dim_k",
    key: torch.Tensor | None = None,
    rounding_mode: str = "rtne",
    **kwargs,
):
    return _mxfp8_swizzle_v2_impl(
        input,
        mode=mode,
        key=key,
        rounding_mode=rounding_mode,
        square_scaling=False,
        **kwargs,
    )


MXFP8_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleGold, cute_fn=mxfp8_swizzle_v2
)


def mxfp8_32x32_swizzle_v2(input: torch.Tensor, **kwargs):
    return _mxfp8_swizzle_v2_impl(
        input,
        mode="dim_k",
        key=None,
        rounding_mode="rtne",
        square_scaling=True,
        **kwargs,
    )


MXFP8_32X32_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp832x32SwizzleGold, cute_fn=mxfp8_32x32_swizzle_v2
)


def _mxfp8_swizzle_sr_v2(input, key, **kwargs):
    return mxfp8_swizzle_v2(
        input, mode="dim_k", key=key, rounding_mode="stochastic", **kwargs
    )


MXFP8_SWIZZLE_SR_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleSRGold, cute_fn=_mxfp8_swizzle_sr_v2
)


# ---------------------------------------------------------------------------
# The unified TMA kernel below supports dim-K, dim-M, and fused dim-KM specializations. Every mode
# uses one input TMA load; constexpr gates select the dim-K and dim-M passes, and each enabled qdata
# result is staged in shared memory for TMA output. Small dim-M/dim-KM inputs use a 32x128 tile to
# expose more CTAs. Larger dim-M uses 64x256, while larger dim-KM uses 64x128.
_MXDMT_SMALL_TILE = (32, 128)
_MXDMT_LARGE_TILE = (64, 256)
_MXDKMT_LARGE_TILE = (64, 128)


@cute.jit
def mxfp8_swizzle_v2_dim_m_or_km_jit(
    mInput: cute.Tensor,
    mOutputM: cute.Tensor,
    mScaleM: cute.Tensor,
    mOutputK: cute.Tensor,
    mScaleK: cute.Tensor,
    mSeed: cute.Tensor,
    M: cutlass.Int32,
    K: cutlass.Int32,
    tile_m_size: cutlass.Constexpr,
    tile_k_size: cutlass.Constexpr,
    cluster_k: cutlass.Constexpr,
    ragged: cutlass.Constexpr,
    mode: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
):
    padded_M = _ceil_div(M, 128) * 128
    padded_N = _ceil_div(K, tile_k_size) * tile_k_size
    if cutlass.const_expr(mode == _MXS_MODE_DIM_KM):
        # Keep each 128-bit row vector intact while XORing row bits into the shared-memory bank
        # selection. This targets the dim-K phase's 16-way row-read conflicts.
        input_smem_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        input_smem_layout = cute.coalesce(
            cute.tile_to_shape(input_smem_atom, (tile_m_size, tile_k_size), order=(0, 1)),
            target_profile=(1, 1),
        )
    else:
        # Keep the kernel argument type uniform while retaining the original unswizzled dim-M
        # address mapping.
        input_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(0, 0, 0),
            0,
            cute.make_layout((tile_m_size, tile_k_size), stride=(tile_k_size, 1)),
        )
    output_m_smem_layout = cute.make_composed_layout(
        cute.make_swizzle(0, 0, 0),
        0,
        cute.make_layout((tile_k_size, tile_m_size), stride=(tile_m_size, 1)),
    )
    input_tma_atom, input_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mInput,
        input_smem_layout,
        (tile_m_size, tile_k_size),
    )
    output_m_tma_atom, output_m_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        mOutputM,
        output_m_smem_layout,
        (tile_k_size, tile_m_size),
    )

    if cutlass.const_expr(mode == _MXS_MODE_DIM_KM):
        output_k_smem_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Float8E4M3FN
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
    else:
        # These arguments disappear with the constexpr dim-K branch.
        output_k_smem_layout = output_m_smem_layout
        output_k_tma_atom = output_m_tma_atom
        output_k_tma_tensor = output_m_tma_tensor

    nrb_m = _ceil_div(K, 128)
    ncb_m = _ceil_div(M, 128)
    scale_m_layout = cute.make_layout(
        ((32, 4, nrb_m), (4, ncb_m)),
        stride=((16, 4, ncb_m * 32 * 16), (1, 32 * 16)),
    )
    mScaleMLogical = cute.make_tensor(mScaleM.iterator, scale_m_layout)
    nrb_k = _ceil_div(M, 128)
    ncb_k = _ceil_div(K, 128)
    scale_k_layout = cute.make_layout(
        ((32, 4, nrb_k), (4, ncb_k)),
        stride=((16, 4, ncb_k * 32 * 16), (1, 32 * 16)),
    )
    mScaleKLogical = cute.make_tensor(mScaleK.iterator, scale_k_layout)

    bpr = tile_k_size // 32
    if cutlass.const_expr(tile_m_size == _MXS_TM):
        data_k_tv_layout = cute.make_layout(
            ((tile_m_size,), (32, bpr)),
            stride=((1,), (tile_m_size, tile_m_size * 32)),
        )
    else:
        # Flatten (1x32 group within a row, row within a 32-row stage) over the 128 threads,
        # then advance the value iteration through successive 32-row stages.
        row_blocks = tile_m_size // 32
        data_k_tv_layout = cute.make_layout(
            ((bpr, 32), (32, row_blocks)),
            stride=((tile_m_size * 32, 1), (tile_m_size, 32)),
        )

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
        tile_m_size,
        tile_k_size,
        M,
        K,
        ragged,
        mode,
        stochastic,
        False,
    )
    grid = (padded_N // tile_k_size, padded_M // tile_m_size, 1)
    block = (max(_MXS_THREADS, tile_k_size), 1, 1)
    if cutlass.const_expr(
        mode == _MXS_MODE_DIM_KM and not stochastic and tile_m_size != 32
    ):
        # A degenerate cluster constrains residency for this larger two-output specialization;
        # an ordinary launch lets Blackwell keep nine CTAs resident per SM instead.
        kernel.launch(grid=grid, block=block)
    else:
        # K-major scheduling keeps adjacent row-major input columns together.
        kernel.launch(
            grid=grid,
            block=block,
            cluster=(cluster_k, 1, 1),
        )


MXFP8_DIM_M_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimMSwizzleGold, cute_fn=partial(mxfp8_swizzle_v2, mode="dim_m")
)


def _mxfp8_dim_m_swizzle_sr_v2(input, key, **kwargs):
    return mxfp8_swizzle_v2(
        input, mode="dim_m", key=key, rounding_mode="stochastic", **kwargs
    )


MXFP8_DIM_M_SWIZZLE_SR_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimMSwizzleSRGold, cute_fn=_mxfp8_dim_m_swizzle_sr_v2
)


MXFP8_DIM_KM_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimKmSwizzleGold, cute_fn=partial(mxfp8_swizzle_v2, mode="dim_km")
)


def _mxfp8_dim_km_swizzle_sr_v2(input, key, **kwargs):
    return mxfp8_swizzle_v2(
        input, mode="dim_km", key=key, rounding_mode="stochastic", **kwargs
    )


MXFP8_DIM_KM_SWIZZLE_SR_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8DimKmSwizzleSRGold, cute_fn=_mxfp8_dim_km_swizzle_sr_v2
)
