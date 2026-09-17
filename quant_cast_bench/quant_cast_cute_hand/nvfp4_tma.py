"""TMA-based NVFP4 kernels and recipe definitions."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

import torch

from quant_cast_bench.quant_cast_cute.recipes import QuantCastCuteRecipe
from quant_cast_bench.quant_cast_cute_hand.utils import (
    _ceil_div,
    _compiled,
    _nvfp4_load_philox_key,
    _nvfp4_quantize_stochastic_x16,
    _nvfp4_quantize_x16,
    _nvfp4_rht_fwht_x16,
    _nvfp4_store_scale_groups,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    Nvfp4GsDimKMSwizzleGold,
    Nvfp4GsDimMSwizzleGold,
    Nvfp4GsDimMSwizzleRHTSRGold,
    Nvfp4GsSwizzle_DimK_DimMRHT_Gold,
    Nvfp4GsSwizzle_DimKSR_DimMRHTSR_Gold,
    Nvfp4GsSwizzleDimMRHTGold,
    Nvfp4GsSwizzleGold,
)


_NVFP4_GROUP = 16
_NVFP4_DIRECT_QVPT = 8
_NVFP4_DIRECT_HALF = 8
_NVFP4_TMA_THREADS = 128
_NVFP4_MODE_DIM_K = 0
_NVFP4_MODE_DIM_M = 1
_NVFP4_MODE_DIM_KM = 2


@cute.jit
def _nvfp4_tma_dim_m_column(
    sInput: cute.Tensor,
    sOutputM: cute.Tensor,
    mScaleMLogical: cute.Tensor,
    sSigns: cute.Tensor,
    outer,
    smem_load_atom: cute.CopyAtom,
    smem_store_atom: cute.CopyAtom,
    local_col,
    pair_count: cutlass.Constexpr,
    n_tile,
    m_tile,
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
    has_rht: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
    counter_base,
    k0,
    k1,
    M,
    N,
):
    """Quantize one shared-memory input column into the transposed dim-M output."""
    output_row_m = n_tile * tile_n + local_col
    output_m_pairs = cute.tiled_divide(sOutputM[(local_col, None)], (16,))
    sr_counter_start = cutlass.Uint64(0)
    if cutlass.const_expr(stochastic):
        sr_counter_start = counter_base + cutlass.Uint64(
            output_row_m * (M // _NVFP4_GROUP)
            + m_tile * (tile_m // _NVFP4_GROUP)
        )
    if cutlass.const_expr(has_rht):
        # Two LDS.128s keep the scaled signs in registers across every RHT group this thread owns.
        rSigns = cute.make_rmem_tensor(_NVFP4_GROUP, cutlass.BFloat16)
        rSignHalves = cute.tiled_divide(rSigns, (_NVFP4_DIRECT_HALF,))
        sSignHalves = cute.tiled_divide(sSigns, (_NVFP4_DIRECT_HALF,))
        for half in cutlass.range_constexpr(2):
            cute.copy(
                smem_load_atom,
                sSignHalves[(None, half)],
                rSignHalves[(None, half)],
            )
    for pair in cutlass.range_constexpr(pair_count):
        rQM = cute.make_rmem_tensor(16, cutlass.Uint8)
        rQMGroups = cute.tiled_divide(rQM, (_NVFP4_DIRECT_QVPT,))
        rScaleM = cute.make_rmem_tensor(2, cutlass.Uint8)
        for group in cutlass.range_constexpr(2):
            row_start = pair * 32 + group * _NVFP4_GROUP
            if cutlass.const_expr(has_rht):
                values = _nvfp4_rht_fwht_x16(
                    sInput, row_start, local_col, rSigns
                )
            else:
                rInputM = cute.make_rmem_tensor(
                    _NVFP4_GROUP, cutlass.BFloat16
                )
                for value in cutlass.range_constexpr(_NVFP4_GROUP):
                    rInputM[value] = sInput[(row_start + value, local_col)]
                values = rInputM.load().to(cutlass.Float32)
            if cutlass.const_expr(stochastic):
                sr_counter = sr_counter_start + cutlass.Uint64(pair * 2 + group)
                qdata_m, scale_m = _nvfp4_quantize_stochastic_x16(
                    values, outer, sr_counter, k0, k1
                )
            else:
                qdata_m, scale_m = _nvfp4_quantize_x16(values, outer)
            rQMGroups[(None, group)].store(qdata_m)
            rScaleM[group] = scale_m
        cute.copy(
            smem_store_atom,
            rQM,
            output_m_pairs[(None, pair)],
        )

        scale_col_m = m_tile * (tile_m // _NVFP4_GROUP) + pair * 2
        scale_cols_m = _ceil_div(M, 64) * 4
        if (output_row_m < _ceil_div(N, 128) * 128) & (
            scale_col_m < scale_cols_m
        ):
            if (output_row_m < N) & (
                scale_col_m + 2 <= M // _NVFP4_GROUP
            ):
                _nvfp4_store_scale_groups(
                    mScaleMLogical,
                    rScaleM,
                    output_row_m,
                    scale_col_m,
                    2,
                )
            else:
                rScaleMPadded = cute.make_rmem_tensor(2, cutlass.Uint8)
                rScaleMPadded.fill(0)
                if output_row_m < N:
                    for group in cutlass.range_constexpr(2):
                        if scale_col_m + group < M // _NVFP4_GROUP:
                            rScaleMPadded[group] = rScaleM[group]
                _nvfp4_store_scale_groups(
                    mScaleMLogical,
                    rScaleMPadded,
                    output_row_m,
                    scale_col_m,
                    2,
                )

@cute.kernel
def nvfp4_swizzle_tma_kernel(
    input_tma_atom: cute.CopyAtom,
    input_tma_tensor: cute.Tensor,
    output_k_tma_atom: cute.CopyAtom,
    output_k_tma_tensor: cute.Tensor,
    output_m_tma_atom: cute.CopyAtom,
    output_m_tma_tensor: cute.Tensor,
    mScaleKLogical: cute.Tensor,
    mScaleMLogical: cute.Tensor,
    mOuterK: cute.Tensor,
    mOuterM: cute.Tensor,
    mRhtSign: cute.Tensor,
    mSeedK: cute.Tensor,
    mSeedM: cute.Tensor,
    input_smem_layout: cute.ComposedLayout,
    output_k_smem_layout: cute.ComposedLayout,
    output_m_smem_layout: cute.ComposedLayout,
    input_k_tv_layout: cute.Layout,
    output_k_tv_layout: cute.Layout,
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
    M: cutlass.Int32,
    N: cutlass.Int32,
    mode: cutlass.Constexpr,
    has_rht: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    n_tile, m_tile, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    do_dim_k = mode != _NVFP4_MODE_DIM_M
    do_dim_m = mode != _NVFP4_MODE_DIM_K

    smem = utils.SmemAllocator()
    input_storage = smem.allocate_array(
        cutlass.BFloat16,
        tile_m * tile_n,
        byte_alignment=1024,
    )
    if cutlass.const_expr(do_dim_m):
        output_m_storage = smem.allocate_array(
            cutlass.Uint8,
            tile_m * tile_n // 2,
            byte_alignment=1024,
        )
    if cutlass.const_expr(do_dim_k and do_dim_m):
        output_k_storage = smem.allocate_array(
            cutlass.Uint8,
            tile_m * tile_n // 2,
            byte_alignment=1024,
        )
    if cutlass.const_expr(has_rht):
        sign_storage = smem.allocate_array(
            cutlass.BFloat16,
            _NVFP4_GROUP,
            byte_alignment=16,
        )
    else:
        sign_storage = smem.allocate_array(
            cutlass.BFloat16, 1, byte_alignment=16
        )
    tma_bar_ptr = smem.allocate_array(cutlass.Int64, 1)
    if tidx == 0:
        cute.arch.mbarrier_init(tma_bar_ptr, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    sInput = cute.make_tensor(
        cute.recast_ptr(
            input_storage, input_smem_layout.inner, dtype=cutlass.BFloat16
        ),
        input_smem_layout.outer,
    )
    sSigns = cute.make_tensor(sign_storage, cute.make_layout(_NVFP4_GROUP))
    gInput = cute.local_tile(
        input_tma_tensor, (tile_m, tile_n), (None, None)
    )
    tInputsInput, tInputgInput = cpasync.tma_partition(
        input_tma_atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sInput, 0, 2),
        cute.group_modes(gInput, 0, 2),
    )
    if cutlass.const_expr(do_dim_k):
        # The standalone dim-K specialization retains its low-smem alias. Fused dim-KM uses a
        # separate output allocation so each packed pair can be staged as soon as it is computed.
        output_k_ptr = (
            output_k_storage
            if cutlass.const_expr(do_dim_m)
            else input_storage
        )
        sOutputK = cute.make_tensor(
            cute.recast_ptr(
                output_k_ptr, output_k_smem_layout.inner, dtype=cutlass.Uint8
            ),
            output_k_smem_layout.outer,
        )
        gOutputK = cute.local_tile(
            output_k_tma_tensor, (tile_m, tile_n // 2), (None, None)
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
                dtype=cutlass.Uint8,
            ),
            output_m_smem_layout.outer,
        )
        gOutputM = cute.local_tile(
            output_m_tma_tensor, (tile_n, tile_m // 2), (None, None)
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
                tma_bar_ptr, tile_m * tile_n * 2
            )
        # TMA cute.copy is warp-collective; do not put it inside the single-lane elect_one block.
        cute.copy(
            input_tma_atom,
            tInputgInput[(None, m_tile, n_tile)],
            tInputsInput,
            tma_bar_ptr=tma_bar_ptr,
        )

    k0_k = cutlass.Uint32(0)
    k1_k = cutlass.Uint32(0)
    counter_base_k = cutlass.Uint64(0)
    k0_m = cutlass.Uint32(0)
    k1_m = cutlass.Uint32(0)
    counter_base_m = cutlass.Uint64(0)
    if cutlass.const_expr(stochastic):
        # Load both independent orientation keys while the input TMA is in flight. Logical flat
        # indices below make both random streams independent of the selected tile shape.
        k0_m, k1_m, counter_base_m = _nvfp4_load_philox_key(mSeedM)
        if cutlass.const_expr(do_dim_k):
            k0_k, k1_k, counter_base_k = _nvfp4_load_philox_key(mSeedK)

    if cutlass.const_expr(has_rht):
        if tidx < _NVFP4_GROUP:
            sSigns[tidx] = (
                mRhtSign[tidx].to(cutlass.Float32)
                * cutlass.Float32(0.25)
            ).to(cutlass.BFloat16)
        cute.arch.sync_threads()

    if cutlass.const_expr(do_dim_k):
        frgOuterK = cute.make_rmem_tensor(
            cute.make_layout(1), mOuterK.element_type
        )
        cute.copy(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mOuterK.element_type),
            mOuterK,
            frgOuterK,
        )
    if cutlass.const_expr(do_dim_m):
        frgOuterM = cute.make_rmem_tensor(
            cute.make_layout(1), mOuterM.element_type
        )
        cute.copy(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mOuterM.element_type),
            mOuterM,
            frgOuterM,
        )
    cute.arch.mbarrier_wait(tma_bar_ptr, 0)

    smem_load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    smem_store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=128
    )

    # Dim-M owns one original input column per thread. Pairs of adjacent 1x16 groups become one
    # aligned 16-byte row segment in the transposed packed output.
    if cutlass.const_expr(do_dim_m):
        _nvfp4_tma_dim_m_column(
            sInput,
            sOutputM,
            mScaleMLogical,
            sSigns,
            frgOuterM[0],
            smem_load_atom,
            smem_store_atom,
            tidx,
            tile_m // 32,
            n_tile,
            m_tile,
            tile_m,
            tile_n,
            has_rht,
            stochastic,
            counter_base_m,
            k0_m,
            k1_m,
            M,
            N,
        )

        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        if warp == 0:
            cute.copy(
                output_m_tma_atom,
                tOutputsM,
                tOutputgM[(None, n_tile, m_tile)],
            )

    # Keep the optimized dim-K pass unchanged in its constexpr specialization. It reads all BF16
    # before reinterpreting that shared allocation as the packed output tile.
    if cutlass.const_expr(do_dim_k):
        tidfrgInputK = cute.composition(sInput, input_k_tv_layout)
        tidfrgOutputK = cute.composition(sOutputK, output_k_tv_layout)
        thrInputGroupsK = tidfrgInputK[(tidx, None)]
        thrOutputGroupsK = tidfrgOutputK[(tidx, None)]
        threads_per_row_k = _NVFP4_TMA_THREADS // tile_m
        groups_per_thread_k = (
            tile_n // _NVFP4_GROUP
        ) // threads_per_row_k
        rScaleK = cute.make_rmem_tensor(groups_per_thread_k, cutlass.Uint8)
        thrOutputPairsK = cute.tiled_divide(thrOutputGroupsK, (16,))
        row_k = m_tile * tile_m + tidx // threads_per_row_k
        scale_col_k = (
            n_tile * (tile_n // _NVFP4_GROUP)
            + (tidx % threads_per_row_k) * groups_per_thread_k
        )
        sr_counter_start_k = cutlass.Uint64(0)
        if cutlass.const_expr(stochastic):
            sr_counter_start_k = counter_base_k + cutlass.Uint64(
                row_k * (N // _NVFP4_GROUP) + scale_col_k
            )
        if cutlass.const_expr(do_dim_m):
            # Separate fused output storage allows each pair to be staged immediately instead of
            # keeping the whole row segment live while the aliased input remains in use.
            for pair in cutlass.range_constexpr(groups_per_thread_k // 2):
                rQPairK = cute.make_rmem_tensor(16, cutlass.Uint8)
                rQPairGroupsK = cute.tiled_divide(
                    rQPairK, (_NVFP4_DIRECT_QVPT,)
                )
                for pair_group in cutlass.range_constexpr(2):
                    group = pair * 2 + pair_group
                    sGroupK = thrInputGroupsK[((None, group),)]
                    sGroupK8 = cute.tiled_divide(
                        sGroupK, (_NVFP4_DIRECT_HALF,)
                    )
                    rInputK = cute.make_rmem_tensor(
                        _NVFP4_GROUP, cutlass.BFloat16
                    )
                    rInputK8 = cute.tiled_divide(
                        rInputK, (_NVFP4_DIRECT_HALF,)
                    )
                    for half in cutlass.range_constexpr(2):
                        cute.copy(
                            smem_load_atom,
                            sGroupK8[(None, half)],
                            rInputK8[(None, half)],
                        )
                    values_k = rInputK.load().to(cutlass.Float32)
                    if cutlass.const_expr(stochastic):
                        qdata_k, scale_k = _nvfp4_quantize_stochastic_x16(
                            values_k,
                            frgOuterK[0],
                            sr_counter_start_k + cutlass.Uint64(group),
                            k0_k,
                            k1_k,
                        )
                    else:
                        qdata_k, scale_k = _nvfp4_quantize_x16(
                            values_k, frgOuterK[0]
                        )
                    rQPairGroupsK[(None, pair_group)].store(qdata_k)
                    rScaleK[group] = scale_k
                cute.copy(
                    smem_store_atom,
                    rQPairK,
                    thrOutputPairsK[(None, pair)],
                )
        else:
            rQK = cute.make_rmem_tensor(
                groups_per_thread_k * _NVFP4_DIRECT_QVPT, cutlass.Uint8
            )
            rQGroupsK = cute.tiled_divide(rQK, (_NVFP4_DIRECT_QVPT,))
            for group in cutlass.range_constexpr(groups_per_thread_k):
                sGroupK = thrInputGroupsK[((None, group),)]
                sGroupK8 = cute.tiled_divide(sGroupK, (_NVFP4_DIRECT_HALF,))
                rInputK = cute.make_rmem_tensor(_NVFP4_GROUP, cutlass.BFloat16)
                rInputK8 = cute.tiled_divide(rInputK, (_NVFP4_DIRECT_HALF,))
                for half in cutlass.range_constexpr(2):
                    cute.copy(
                        smem_load_atom,
                        sGroupK8[(None, half)],
                        rInputK8[(None, half)],
                    )
                values_k = rInputK.load().to(cutlass.Float32)
                if cutlass.const_expr(stochastic):
                    qdata_k, scale_k = _nvfp4_quantize_stochastic_x16(
                        values_k,
                        frgOuterK[0],
                        sr_counter_start_k + cutlass.Uint64(group),
                        k0_k,
                        k1_k,
                    )
                else:
                    qdata_k, scale_k = _nvfp4_quantize_x16(
                        values_k, frgOuterK[0]
                    )
                rQGroupsK[(None, group)].store(qdata_k)
                rScaleK[group] = scale_k

            cute.arch.sync_threads()
            rQPairsK = cute.tiled_divide(rQK, (16,))
            for pair in cutlass.range_constexpr(groups_per_thread_k // 2):
                cute.copy(
                    smem_store_atom,
                    rQPairsK[(None, pair)],
                    thrOutputPairsK[(None, pair)],
                )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        if warp == 0:
            cute.copy(
                output_k_tma_atom,
                tOutputsK,
                tOutputgK[(None, m_tile, n_tile)],
            )

        if scale_col_k < _ceil_div(N, 64) * 4:
            if (row_k < M) & (
                scale_col_k + groups_per_thread_k <= N // _NVFP4_GROUP
            ):
                _nvfp4_store_scale_groups(
                    mScaleKLogical,
                    rScaleK,
                    row_k,
                    scale_col_k,
                    groups_per_thread_k,
                )
            else:
                rScaleKPadded = cute.make_rmem_tensor(
                    groups_per_thread_k, cutlass.Uint8
                )
                rScaleKPadded.fill(0)
                if row_k < M:
                    for group in cutlass.range_constexpr(groups_per_thread_k):
                        if scale_col_k + group < N // _NVFP4_GROUP:
                            rScaleKPadded[group] = rScaleK[group]
                _nvfp4_store_scale_groups(
                    mScaleKLogical,
                    rScaleKPadded,
                    row_k,
                    scale_col_k,
                    groups_per_thread_k,
                )


@cute.jit
def nvfp4_swizzle_tma_jit(
    mInput: cute.Tensor,
    mOutput: cute.Tensor,
    mScale: cute.Tensor,
    mOuter: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
    cluster_n: cutlass.Constexpr,
):
    input_smem_atom = tcgen05.make_smem_layout_atom(
        tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
    )
    input_smem_layout = cute.coalesce(
        cute.tile_to_shape(
            input_smem_atom, (tile_m, tile_n), order=(0, 1)
        ),
        target_profile=(1, 1),
    )
    output_smem_kind = (
        tcgen05.SmemLayoutAtomKind.K_SW64
        if cutlass.const_expr(tile_n == 128)
        else tcgen05.SmemLayoutAtomKind.K_SW32
    )
    output_smem_atom = tcgen05.make_smem_layout_atom(
        output_smem_kind, cutlass.Uint8
    )
    output_smem_layout = cute.coalesce(
        cute.tile_to_shape(
            output_smem_atom,
            (tile_m, tile_n // 2),
            order=(0, 1),
        ),
        target_profile=(1, 1),
    )
    threads_per_row = _NVFP4_TMA_THREADS // tile_m
    groups_per_thread = (tile_n // _NVFP4_GROUP) // threads_per_row
    input_tv_layout = cute.make_layout(
        ((threads_per_row, tile_m), (_NVFP4_GROUP, groups_per_thread)),
        stride=(
            (tile_m * _NVFP4_GROUP * groups_per_thread, 1),
            (tile_m, tile_m * _NVFP4_GROUP),
        ),
    )
    output_tv_layout = cute.make_layout(
        ((threads_per_row, tile_m), (_NVFP4_DIRECT_QVPT, groups_per_thread)),
        stride=(
            (tile_m * _NVFP4_DIRECT_QVPT * groups_per_thread, 1),
            (tile_m, tile_m * _NVFP4_DIRECT_QVPT),
        ),
    )
    input_tma_atom, input_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mInput,
        input_smem_layout,
        (tile_m, tile_n),
    )
    output_tma_atom, output_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        mOutput,
        output_smem_layout,
        (tile_m, tile_n // 2),
    )
    nrb = _ceil_div(M, 128)
    ncb = _ceil_div(N, 64)
    scale_layout = cute.make_layout(
        ((32, 4, nrb), (4, ncb)),
        stride=((16, 4, ncb * 32 * 16), (1, 32 * 16)),
    )
    mScaleLogical = cute.make_tensor(mScale.iterator, scale_layout)
    padded_M = nrb * 128
    padded_N = ncb * 64
    nvfp4_swizzle_tma_kernel(
        input_tma_atom,
        input_tma_tensor,
        output_tma_atom,
        output_tma_tensor,
        output_tma_atom,
        output_tma_tensor,
        mScaleLogical,
        mScaleLogical,
        mOuter,
        mOuter,
        mInput,
        mScale,
        mScale,
        input_smem_layout,
        output_smem_layout,
        output_smem_layout,
        input_tv_layout,
        output_tv_layout,
        tile_m,
        tile_n,
        M,
        N,
        _NVFP4_MODE_DIM_K,
        False,
        False,
    ).launch(
        grid=(padded_N // tile_n,
              padded_M // tile_m, 1),
        block=(_NVFP4_TMA_THREADS, 1, 1),
        cluster=(cluster_n, 1, 1),
    )


@cute.jit
def nvfp4_swizzle_tma_m_jit(
    mInput: cute.Tensor,
    mOutputM: cute.Tensor,
    mScaleM: cute.Tensor,
    mOuterM: cute.Tensor,
    mOutputK: cute.Tensor,
    mScaleK: cute.Tensor,
    mOuterK: cute.Tensor,
    mRhtSign: cute.Tensor,
    mSeedK: cute.Tensor,
    mSeedM: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
    mode: cutlass.Constexpr,
    has_rht: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
    cluster_n: cutlass.Constexpr,
):
    do_dim_k = mode == _NVFP4_MODE_DIM_KM
    if cutlass.const_expr(do_dim_k):
        input_smem_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        input_smem_layout = cute.coalesce(
            cute.tile_to_shape(
                input_smem_atom, (tile_m, tile_n), order=(0, 1)
            ),
            target_profile=(1, 1),
        )
    else:
        input_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(0, 0, 0),
            0,
            cute.make_layout((tile_m, tile_n), stride=(tile_n, 1)),
        )

    if cutlass.const_expr(tile_m == 32):
        output_m_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(0, 0, 0),
            0,
            cute.make_layout(
                (tile_n, tile_m // 2), stride=(tile_m // 2, 1)
            ),
        )
    else:
        output_m_smem_kind = (
            tcgen05.SmemLayoutAtomKind.K_SW64
            if cutlass.const_expr(tile_m == 128)
            else tcgen05.SmemLayoutAtomKind.K_SW32
        )
        output_m_smem_atom = tcgen05.make_smem_layout_atom(
            output_m_smem_kind, cutlass.Uint8
        )
        output_m_smem_layout = cute.coalesce(
            cute.tile_to_shape(
                output_m_smem_atom,
                (tile_n, tile_m // 2),
                order=(0, 1),
            ),
            target_profile=(1, 1),
        )
    input_tma_atom, input_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mInput,
        input_smem_layout,
        (tile_m, tile_n),
    )
    output_m_tma_atom, output_m_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        mOutputM,
        output_m_smem_layout,
        (tile_n, tile_m // 2),
    )

    output_k_smem_atom = tcgen05.make_smem_layout_atom(
        tcgen05.SmemLayoutAtomKind.K_SW64, cutlass.Uint8
    )
    output_k_smem_layout = cute.coalesce(
        cute.tile_to_shape(
            output_k_smem_atom,
            (tile_m, tile_n // 2),
            order=(0, 1),
        ),
        target_profile=(1, 1),
    )
    if cutlass.const_expr(do_dim_k):
        output_k_tma_atom, output_k_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mOutputK,
            output_k_smem_layout,
            (tile_m, tile_n // 2),
        )
    else:
        output_k_tma_atom = output_m_tma_atom
        output_k_tma_tensor = output_m_tma_tensor
        output_k_smem_layout = output_m_smem_layout

    threads_per_row_k = _NVFP4_TMA_THREADS // tile_m
    groups_per_thread_k = (
        tile_n // _NVFP4_GROUP
    ) // threads_per_row_k
    input_k_tv_layout = cute.make_layout(
        (
            (threads_per_row_k, tile_m),
            (_NVFP4_GROUP, groups_per_thread_k),
        ),
        stride=(
            (tile_m * _NVFP4_GROUP * groups_per_thread_k, 1),
            (tile_m, tile_m * _NVFP4_GROUP),
        ),
    )
    output_k_tv_layout = cute.make_layout(
        (
            (threads_per_row_k, tile_m),
            (_NVFP4_DIRECT_QVPT, groups_per_thread_k),
        ),
        stride=(
            (tile_m * _NVFP4_DIRECT_QVPT * groups_per_thread_k, 1),
            (tile_m, tile_m * _NVFP4_DIRECT_QVPT),
        ),
    )

    nrb_m, ncb_m = _ceil_div(N, 128), _ceil_div(M, 64)
    scale_m_layout = cute.make_layout(
        ((32, 4, nrb_m), (4, ncb_m)),
        stride=((16, 4, ncb_m * 32 * 16), (1, 32 * 16)),
    )
    mScaleMLogical = cute.make_tensor(mScaleM.iterator, scale_m_layout)
    if cutlass.const_expr(do_dim_k):
        nrb_k, ncb_k = _ceil_div(M, 128), _ceil_div(N, 64)
        scale_k_layout = cute.make_layout(
            ((32, 4, nrb_k), (4, ncb_k)),
            stride=((16, 4, ncb_k * 32 * 16), (1, 32 * 16)),
        )
        mScaleKLogical = cute.make_tensor(mScaleK.iterator, scale_k_layout)
    else:
        mScaleKLogical = mScaleMLogical

    # The dim-M scale grid pads original M to four 1x16 groups and original N to 128 rows.
    padded_M = (
        _ceil_div(M, 128) * 128
        if cutlass.const_expr(do_dim_k)
        else _ceil_div(M, 64) * 64
    )
    padded_N = _ceil_div(N, 128) * 128
    kernel = nvfp4_swizzle_tma_kernel(
        input_tma_atom,
        input_tma_tensor,
        output_k_tma_atom,
        output_k_tma_tensor,
        output_m_tma_atom,
        output_m_tma_tensor,
        mScaleKLogical,
        mScaleMLogical,
        mOuterK,
        mOuterM,
        mRhtSign,
        mSeedK,
        mSeedM,
        input_smem_layout,
        output_k_smem_layout,
        output_m_smem_layout,
        input_k_tv_layout,
        output_k_tv_layout,
        tile_m,
        tile_n,
        M,
        N,
        mode,
        has_rht,
        stochastic,
    )
    grid = (padded_N // tile_n, padded_M // tile_m, 1)
    block = (_NVFP4_TMA_THREADS, 1, 1)
    if cutlass.const_expr(cluster_n == 1):
        kernel.launch(grid=grid, block=block)
    else:
        kernel.launch(grid=grid, block=block, cluster=(cluster_n, 1, 1))
def _validate_nvfp4_swizzle_inputs(input, outer_scale, kernel_name, k_multiple):
    assert input.dim() == 2, f"{kernel_name} requires a 2-D input"
    assert input.is_contiguous(), f"{kernel_name} requires contiguous input"
    assert input.dtype == torch.bfloat16, f"{kernel_name} is bf16-only"
    assert outer_scale.device == input.device, "input and outer scale must be on the same device"
    assert outer_scale.dtype == torch.float32 and outer_scale.numel() == 1, (
        "outer scale must be a float32 scalar"
    )
    M, N = input.shape
    assert M > 0 and N > 0, f"{kernel_name} requires non-empty dimensions"
    assert N % k_multiple == 0, f"{kernel_name} requires K % {k_multiple} == 0"
    return M, N


def _allocate_nvfp4_swizzle_outputs(input, M, N):
    nrb, ncb = _ceil_div(M, 128), _ceil_div(N, 64)
    output = torch.empty(M, N // 2, dtype=torch.uint8, device=input.device)
    scale = torch.empty(
        nrb * ncb * 32 * 16, dtype=torch.uint8, device=input.device
    )
    return output, scale, nrb, ncb
def nvfp4_swizzle_tma(
    input: torch.Tensor,
    outer_scale: torch.Tensor,
    outer_scale_m: torch.Tensor | None = None,
    mode: str = "dim_k",
    rht_sign: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
    key_m: torch.Tensor | None = None,
    rounding_mode: str = "rtne",
    **kwargs,
):
    assert mode in ("dim_k", "dim_m", "dim_km"), f"unsupported mode: {mode}"
    has_rht = rht_sign is not None
    rounding_mode = str(getattr(rounding_mode, "value", rounding_mode)).lower()
    assert rounding_mode in ("rtne", "stochastic"), (
        f"unsupported rounding_mode: {rounding_mode}"
    )
    stochastic = rounding_mode == "stochastic"
    if stochastic:
        assert has_rht and mode in ("dim_m", "dim_km"), (
            "stochastic rounding currently requires an RHT dim-m output"
        )
        assert key is not None, "stochastic rounding requires a Philox key"
        assert key.device == input.device, (
            "input and Philox key must be on the same device"
        )
        assert key.dtype == torch.uint64 and key.numel() == 2, (
            "Philox key must be uint64[2]"
        )
        if mode == "dim_km":
            assert key_m is not None, "stochastic dim-km requires a dim-M Philox key"
            assert key_m.device == input.device, (
                "input and dim-M Philox key must be on the same device"
            )
            assert key_m.dtype == torch.uint64 and key_m.numel() == 2, (
                "dim-M Philox key must be uint64[2]"
            )
        else:
            assert key_m is None, "stochastic dim-m takes one Philox key"
    else:
        assert key is None and key_m is None, (
            "RTNE rounding does not use Philox keys"
        )
    if has_rht:
        assert mode in ("dim_m", "dim_km"), (
            "RHT requires a dim-m output"
        )
        assert rht_sign.shape == (16,), "RHT sign input must have shape (16,)"
        assert rht_sign.dtype == torch.bfloat16 and rht_sign.device == input.device, (
            "RHT sign input must be bf16 on the input device"
        )
        assert rht_sign.is_contiguous(), "RHT sign input must be contiguous"
    # Packed TMA output rows must have a 16-byte stride: dim-K therefore requires K%32, while
    # dim-M requires M%32. The input descriptor additionally needs its BF16 row stride aligned.
    input_k_multiple = 32 if mode in ("dim_k", "dim_km") else 16
    M, N = _validate_nvfp4_swizzle_inputs(
        input, outer_scale, "nvfp4_swizzle_tma", input_k_multiple
    )
    if mode == "dim_k":
        assert outer_scale_m is None, "dim-k takes one outer scale"

    if mode != "dim_k":
        assert M % 32 == 0, "nvfp4 dim-M TMA requires M % 32 == 0"
        if mode == "dim_km":
            assert outer_scale_m is not None, "dim-km requires a dim-M outer scale"
            assert outer_scale_m.device == input.device, (
                "input and dim-M outer scale must be on the same device"
            )
            assert outer_scale_m.dtype == torch.float32 and outer_scale_m.numel() == 1, (
                "dim-M outer scale must be a float32 scalar"
            )
        else:
            assert outer_scale_m is None, "dim-m takes one outer scale"
            outer_scale_m = outer_scale

    if mode != "dim_k":
        output_m, scale_m, nrb_m, ncb_m = _allocate_nvfp4_swizzle_outputs(
            input, N, M
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
        mScaleM = (
            from_dlpack(scale_m, assumed_align=4)
            .mark_layout_dynamic(leading_dim=0)
            .mark_compact_shape_dynamic(mode=0, divisibility=512)
        )
        mOuterM = from_dlpack(outer_scale_m.reshape(1))
        mRhtSign = (
            from_dlpack(rht_sign, assumed_align=16)
            if rht_sign is not None
            else mInput
        )
        # Compile-time RTNE branches never read these dummy tensors. Dim-M-only SR treats `key` as
        # its primary stream; fused SR uses independent primary dim-K and secondary dim-M streams.
        if stochastic:
            if mode == "dim_km":
                mSeedK = from_dlpack(key.reshape(-1).view(torch.int64))
                mSeedM = from_dlpack(key_m.reshape(-1).view(torch.int64))
            else:
                mSeedK = mScaleM
                mSeedM = from_dlpack(key.reshape(-1).view(torch.int64))
        else:
            mSeedK = mScaleM
            mSeedM = mScaleM

        if mode == "dim_km":
            output_k, scale_k, nrb_k, ncb_k = _allocate_nvfp4_swizzle_outputs(
                input, M, N
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
            mOuterK = from_dlpack(outer_scale.reshape(1))
            mode_id = _NVFP4_MODE_DIM_KM
        else:
            output_k, scale_k, nrb_k, ncb_k = output_m, scale_m, nrb_m, ncb_m
            mOutputK, mScaleK, mOuterK = mOutputM, mScaleM, mOuterM
            mode_id = _NVFP4_MODE_DIM_M

        fused_rht = mode == "dim_km" and has_rht
        if fused_rht:
            tile_m = 32 if M * N <= 2048 * 2048 else 64
            tile_n = 128
        elif M * N <= 2048 * 2048:
            tile_m, tile_n = 32, 128
        elif stochastic and M * N >= 8192 * 8192:
            tile_m, tile_n = 128, 128
        else:
            tile_m, tile_n = 64, 128
        # Fused SR benefits from pairing adjacent CTAs at both tile sizes; fused RTNE does so only
        # through 4K. Standalone RTNE clusters only its smallest tile, while standalone SR also
        # pairs large tiles.
        if fused_rht:
            cluster_n = (
                2
                if (stochastic or M * N <= 4096 * 4096)
                and _ceil_div(N, tile_n) % 2 == 0
                else 1
            )
        elif stochastic and M * N >= 8192 * 8192:
            cluster_n = 2 if _ceil_div(N, tile_n) % 2 == 0 else 1
        else:
            cluster_n = (
                2
                if has_rht
                and M * N <= 2048 * 2048
                and (_ceil_div(N, tile_n) % 2 == 0)
                else 1
            )
        fn = _compiled(
            (
                "nvfp4_swizzle_tma",
                mode,
                has_rht,
                stochastic,
                tile_m,
                tile_n,
                cluster_n,
            ),
            nvfp4_swizzle_tma_m_jit,
            mInput,
            mOutputM,
            mScaleM,
            mOuterM,
            mOutputK,
            mScaleK,
            mOuterK,
            mRhtSign,
            mSeedK,
            mSeedM,
            M,
            N,
            tile_m,
            tile_n,
            mode_id,
            has_rht,
            stochastic,
            cluster_n,
        )
        fn(
            mInput,
            mOutputM,
            mScaleM,
            mOuterM,
            mOutputK,
            mScaleK,
            mOuterK,
            mRhtSign,
            mSeedK,
            mSeedM,
            M,
            N,
        )
        output_m = output_m.view(torch.float4_e2m1fn_x2)
        scale_m = scale_m.view(nrb_m, ncb_m, 32, 16).view(
            torch.float8_e4m3fn
        )
        if mode == "dim_km":
            return (
                output_k.view(torch.float4_e2m1fn_x2),
                scale_k.view(nrb_k, ncb_k, 32, 16).view(torch.float8_e4m3fn),
                output_m,
                scale_m,
            )
        return output_m, scale_m

    output, scale, nrb, ncb = _allocate_nvfp4_swizzle_outputs(input, M, N)
    mInput = (
        from_dlpack(input, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )
    mOutput = (
        from_dlpack(output, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )
    mScale = (
        from_dlpack(scale, assumed_align=4)
        .mark_layout_dynamic(leading_dim=0)
        .mark_compact_shape_dynamic(mode=0, divisibility=512)
    )
    mOuter = from_dlpack(outer_scale.reshape(1))
    # A rotated 32x128 tile exposes more M-parallel CTAs for small problems. Larger problems use
    # one thread per row; 128 columns then amortizes TMA/barrier overhead once enough work exists.
    if M * N <= 2048 * 2048 and ncb % 2 == 0:
        tile_m, tile_n = 32, 128
    else:
        tile_m = 128
        tile_n = 128 if M * N >= 4096 * 4096 and ncb % 2 == 0 else 64
    cluster_n = 1
    fn = _compiled(
        ("nvfp4_swizzle_tma", tile_m, tile_n, cluster_n),
        nvfp4_swizzle_tma_jit,
        mInput,
        mOutput,
        mScale,
        mOuter,
        M,
        N,
        tile_m,
        tile_n,
        cluster_n,
    )
    fn(mInput, mOutput, mScale, mOuter, M, N)
    return (
        output.view(torch.float4_e2m1fn_x2),
        scale.view(nrb, ncb, 32, 16).view(torch.float8_e4m3fn),
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
    key_k,
    key_m,
    **kwargs,
):
    return nvfp4_swizzle_tma(
        input,
        outer_scale_k,
        outer_scale_m=outer_scale_m,
        mode="dim_km",
        rht_sign=rht_sign,
        key=key_k,
        key_m=key_m,
        rounding_mode="stochastic",
        **kwargs,
    )


NVFP4_SWIZZLE_DIM_K_SR_DIM_M_RHT_SR_TMA = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzle_DimKSR_DimMRHTSR_Gold,
    cute_fn=nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_tma,
)
