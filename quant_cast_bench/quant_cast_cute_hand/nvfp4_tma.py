"""TMA-based NVFP4 kernels and recipe definitions."""

from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache
from torch._vendor.quack.cache import EXTRA_SOURCE_DIRS

from quant_cast_bench.quant_cast_cute.recipes import QuantCastCuteRecipe
from quant_cast_bench.quant_cast_cute_hand.utils import (
    _ceil_div,
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


_NVFP4_TMA_SOURCE_DIR = Path(__file__).resolve().parent
if _NVFP4_TMA_SOURCE_DIR not in EXTRA_SOURCE_DIRS:
    EXTRA_SOURCE_DIRS.append(_NVFP4_TMA_SOURCE_DIR)


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


class _Nvfp4SwizzleTma:
    """Compile-time NVFP4 configuration with a CuTe launcher and device kernel."""

    def __init__(
        self,
        tile_m_size: int,
        tile_k_size: int,
        cluster_k: int,
        quant_orientation: int,
        has_rht: bool,
        is_stochastic_qdata_rounding: bool,
    ) -> None:
        self.tile_m_size = tile_m_size
        self.tile_k_size = tile_k_size
        self.cluster_k = cluster_k
        self.quant_orientation = quant_orientation
        self.has_rht = has_rht
        self.is_stochastic_qdata_rounding = is_stochastic_qdata_rounding

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
        mOuterK: cute.Tensor | None,
        mOuterM: cute.Tensor | None,
        mRhtSign: cute.Tensor | None,
        mSeedK: cute.Tensor | None,
        mSeedM: cute.Tensor | None,
        input_smem_layout: cute.ComposedLayout,
        output_k_smem_layout: cute.ComposedLayout | None,
        output_m_smem_layout: cute.ComposedLayout | None,
        input_k_tv_layout: cute.Layout | None,
        output_k_tv_layout: cute.Layout | None,
        M: cutlass.Int32,
        N: cutlass.Int32,
    ) -> None:
        tile_m = cutlass.const_expr(self.tile_m_size)
        tile_n = cutlass.const_expr(self.tile_k_size)
        mode = cutlass.const_expr(self.quant_orientation)
        has_rht = cutlass.const_expr(self.has_rht)
        stochastic = cutlass.const_expr(self.is_stochastic_qdata_rounding)

        if cutlass.const_expr(mode == _NVFP4_MODE_DIM_K):
            assert output_k_tma_atom is not None
            assert output_k_tma_tensor is not None
            assert mScaleKLogical is not None
            assert mOuterK is not None
            assert output_k_smem_layout is not None
            assert input_k_tv_layout is not None
            assert output_k_tv_layout is not None
            assert output_m_tma_atom is None
            assert output_m_tma_tensor is None
            assert mScaleMLogical is None
            assert mOuterM is None
            assert output_m_smem_layout is None
        elif cutlass.const_expr(mode == _NVFP4_MODE_DIM_M):
            assert output_k_tma_atom is None
            assert output_k_tma_tensor is None
            assert mScaleKLogical is None
            assert mOuterK is None
            assert output_k_smem_layout is None
            assert input_k_tv_layout is None
            assert output_k_tv_layout is None
            assert output_m_tma_atom is not None
            assert output_m_tma_tensor is not None
            assert mScaleMLogical is not None
            assert mOuterM is not None
            assert output_m_smem_layout is not None
        else:
            assert mode == _NVFP4_MODE_DIM_KM
            assert output_k_tma_atom is not None
            assert output_k_tma_tensor is not None
            assert mScaleKLogical is not None
            assert mOuterK is not None
            assert output_k_smem_layout is not None
            assert input_k_tv_layout is not None
            assert output_k_tv_layout is not None
            assert output_m_tma_atom is not None
            assert output_m_tma_tensor is not None
            assert mScaleMLogical is not None
            assert mOuterM is not None
            assert output_m_smem_layout is not None

        if cutlass.const_expr(has_rht):
            assert mRhtSign is not None
        else:
            assert mRhtSign is None
        if cutlass.const_expr(stochastic):
            assert mode != _NVFP4_MODE_DIM_K
            assert mSeedM is not None
            if cutlass.const_expr(mode == _NVFP4_MODE_DIM_KM):
                assert mSeedK is not None
            else:
                assert mSeedK is None
        else:
            assert mSeedK is None
            assert mSeedM is None

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
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutputK: cute.Tensor | None,
        mScaleK: cute.Tensor | None,
        mOuterK: cute.Tensor | None,
        mOutputM: cute.Tensor | None,
        mScaleM: cute.Tensor | None,
        mOuterM: cute.Tensor | None,
        mRhtSign: cute.Tensor | None,
        mSeedK: cute.Tensor | None,
        mSeedM: cute.Tensor | None,
        stream: cuda.CUstream,
        M: cutlass.Int32,
        N: cutlass.Int32,
    ) -> None:
        tile_m = cutlass.const_expr(self.tile_m_size)
        tile_n = cutlass.const_expr(self.tile_k_size)
        cluster_n = cutlass.const_expr(self.cluster_k)
        mode = cutlass.const_expr(self.quant_orientation)
        has_rht = cutlass.const_expr(self.has_rht)
        stochastic = cutlass.const_expr(self.is_stochastic_qdata_rounding)
        do_dim_k = mode != _NVFP4_MODE_DIM_M
        do_dim_m = mode != _NVFP4_MODE_DIM_K

        if cutlass.const_expr(mode == _NVFP4_MODE_DIM_K):
            assert mOutputK is not None
            assert mScaleK is not None
            assert mOuterK is not None
            assert mOutputM is None
            assert mScaleM is None
            assert mOuterM is None
        elif cutlass.const_expr(mode == _NVFP4_MODE_DIM_M):
            assert mOutputK is None
            assert mScaleK is None
            assert mOuterK is None
            assert mOutputM is not None
            assert mScaleM is not None
            assert mOuterM is not None
        else:
            assert mode == _NVFP4_MODE_DIM_KM
            assert mOutputK is not None
            assert mScaleK is not None
            assert mOuterK is not None
            assert mOutputM is not None
            assert mScaleM is not None
            assert mOuterM is not None

        if cutlass.const_expr(has_rht):
            assert mRhtSign is not None
        else:
            assert mRhtSign is None
        if cutlass.const_expr(stochastic):
            assert mSeedM is not None
            if cutlass.const_expr(do_dim_k):
                assert mSeedK is not None
            else:
                assert mSeedK is None
        else:
            assert mSeedK is None
            assert mSeedM is None

        if cutlass.const_expr(mode == _NVFP4_MODE_DIM_M):
            input_smem_layout = cute.make_composed_layout(
                cute.make_swizzle(0, 0, 0),
                0,
                cute.make_layout((tile_m, tile_n), stride=(tile_n, 1)),
            )
        else:
            input_smem_atom = tcgen05.make_smem_layout_atom(
                tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
            )
            input_smem_layout = cute.coalesce(
                cute.tile_to_shape(
                    input_smem_atom, (tile_m, tile_n), order=(0, 1)
                ),
                target_profile=(1, 1),
            )

        input_tma_atom, input_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mInput,
            input_smem_layout,
            (tile_m, tile_n),
        )

        if cutlass.const_expr(do_dim_k):
            if cutlass.const_expr(mode == _NVFP4_MODE_DIM_K):
                output_k_smem_kind = (
                    tcgen05.SmemLayoutAtomKind.K_SW64
                    if cutlass.const_expr(tile_n == 128)
                    else tcgen05.SmemLayoutAtomKind.K_SW32
                )
            else:
                output_k_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW64
            output_k_smem_atom = tcgen05.make_smem_layout_atom(
                output_k_smem_kind, cutlass.Uint8
            )
            output_k_smem_layout = cute.coalesce(
                cute.tile_to_shape(
                    output_k_smem_atom,
                    (tile_m, tile_n // 2),
                    order=(0, 1),
                ),
                target_profile=(1, 1),
            )
            output_k_tma_atom, output_k_tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mOutputK,
                output_k_smem_layout,
                (tile_m, tile_n // 2),
            )

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

            nrb_k, ncb_k = _ceil_div(M, 128), _ceil_div(N, 64)
            scale_k_layout = cute.make_layout(
                ((32, 4, nrb_k), (4, ncb_k)),
                stride=((16, 4, ncb_k * 32 * 16), (1, 32 * 16)),
            )
            mScaleKLogical = cute.make_tensor(mScaleK.iterator, scale_k_layout)
        else:
            output_k_tma_atom = None
            output_k_tma_tensor = None
            output_k_smem_layout = None
            input_k_tv_layout = None
            output_k_tv_layout = None
            mScaleKLogical = None

        if cutlass.const_expr(do_dim_m):
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
            output_m_tma_atom, output_m_tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mOutputM,
                output_m_smem_layout,
                (tile_n, tile_m // 2),
            )

            nrb_m, ncb_m = _ceil_div(N, 128), _ceil_div(M, 64)
            scale_m_layout = cute.make_layout(
                ((32, 4, nrb_m), (4, ncb_m)),
                stride=((16, 4, ncb_m * 32 * 16), (1, 32 * 16)),
            )
            mScaleMLogical = cute.make_tensor(mScaleM.iterator, scale_m_layout)
        else:
            output_m_tma_atom = None
            output_m_tma_tensor = None
            output_m_smem_layout = None
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
            M,
            N,
        )

        if cutlass.const_expr(mode == _NVFP4_MODE_DIM_K):
            padded_m = _ceil_div(M, 128) * 128
            padded_n = _ceil_div(N, 64) * 64
            launch_cluster = (cluster_n, 1, 1)
        else:
            padded_m = (
                _ceil_div(M, 128) * 128
                if cutlass.const_expr(do_dim_k)
                else _ceil_div(M, 64) * 64
            )
            padded_n = _ceil_div(N, 128) * 128
            launch_cluster = (
                None
                if cutlass.const_expr(cluster_n == 1)
                else (cluster_n, 1, 1)
            )
        grid = (padded_n // tile_n, padded_m // tile_m, 1)
        block = (_NVFP4_TMA_THREADS, 1, 1)
        kernel.launch(
            grid=grid,
            block=block,
            cluster=launch_cluster,
            stream=stream,
        )


def _make_dynamic_matrix_fake(dtype):
    """Match a 16-byte-aligned row-major tensor with a dynamic, 16-divisible K."""
    return cute.runtime.make_fake_tensor(
        dtype,
        (cute.sym_int(), cute.sym_int(divisibility=16)),
        stride=(cute.sym_int64(divisibility=16), 1),
        assumed_align=16,
    )


def _make_dynamic_scale_fake():
    """Match the compact, padded E4M3 scale-byte allocation used at runtime."""
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


def _nvfp4_tma_compile_log_key(
    tile_m_size: int,
    tile_k_size: int,
    cluster_k: int,
    quant_orientation: int,
    has_rht: bool,
    is_stochastic_qdata_rounding: bool,
) -> str:
    return (
        f"orientation={quant_orientation} tile={tile_m_size}x{tile_k_size} "
        f"cluster_k={cluster_k} rht={has_rht} "
        f"sr={is_stochastic_qdata_rounding}"
    )


@instrumented_cutedsl_cache(
    "quant_cast_bench::nvfp4_swizzle_tma",
    key_fn=_nvfp4_tma_compile_log_key,
)
def _compile_nvfp4_swizzle_tma(
    tile_m_size: int,
    tile_k_size: int,
    cluster_k: int,
    quant_orientation: int,
    has_rht: bool,
    is_stochastic_qdata_rounding: bool,
):
    do_dim_k = quant_orientation != _NVFP4_MODE_DIM_M
    do_dim_m = quant_orientation != _NVFP4_MODE_DIM_K
    operation = _Nvfp4SwizzleTma(
        tile_m_size,
        tile_k_size,
        cluster_k,
        quant_orientation,
        has_rht,
        is_stochastic_qdata_rounding,
    )

    mInput = _make_dynamic_matrix_fake(cutlass.BFloat16)
    mOutputK = _make_dynamic_matrix_fake(cutlass.Uint8) if do_dim_k else None
    mScaleK = _make_dynamic_scale_fake() if do_dim_k else None
    mOuterK = (
        _make_static_vector_fake(cutlass.Float32, 1, assumed_align=4)
        if do_dim_k
        else None
    )
    mOutputM = _make_dynamic_matrix_fake(cutlass.Uint8) if do_dim_m else None
    mScaleM = _make_dynamic_scale_fake() if do_dim_m else None
    mOuterM = (
        _make_static_vector_fake(cutlass.Float32, 1, assumed_align=4)
        if do_dim_m
        else None
    )
    mRhtSign = (
        _make_static_vector_fake(cutlass.BFloat16, _NVFP4_GROUP, assumed_align=16)
        if has_rht
        else None
    )
    mSeedK = (
        _make_static_vector_fake(cutlass.Int64, 2, assumed_align=8)
        if is_stochastic_qdata_rounding and do_dim_k
        else None
    )
    mSeedM = (
        _make_static_vector_fake(cutlass.Int64, 2, assumed_align=8)
        if is_stochastic_qdata_rounding
        else None
    )

    return cute.compile(
        operation,
        mInput,
        mOutputK,
        mScaleK,
        mOuterK,
        mOutputM,
        mScaleM,
        mOuterM,
        mRhtSign,
        mSeedK,
        mSeedM,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        cutlass.Int32(0),
        cutlass.Int32(0),
        options="--enable-tvm-ffi",
    )


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


def _nvfp4_swizzle_tma_impl_on_current_device(
    input: torch.Tensor,
    outer_scale: torch.Tensor,
    outer_scale_m: torch.Tensor | None,
    mode: str,
    rht_sign: torch.Tensor | None,
    key: torch.Tensor | None,
    key_m: torch.Tensor | None,
    rounding_mode: str,
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

    do_dim_k = mode != "dim_m"
    do_dim_m = mode != "dim_k"

    output_m = scale_m = None
    nrb_m = ncb_m = None
    if do_dim_m:
        output_m, scale_m, nrb_m, ncb_m = _allocate_nvfp4_swizzle_outputs(
            input, N, M
        )

    output_k = scale_k = None
    nrb_k = ncb_k = None
    if do_dim_k:
        output_k, scale_k, nrb_k, ncb_k = _allocate_nvfp4_swizzle_outputs(
            input, M, N
        )

    if mode == "dim_k":
        tile_m, tile_n = (
            (32, 128)
            if M * N <= 2048 * 2048 and ncb_k % 2 == 0
            else (
                128,
                128 if M * N >= 4096 * 4096 and ncb_k % 2 == 0 else 64,
            )
        )
        cluster_n = 1
        mode_id = _NVFP4_MODE_DIM_K
    else:
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
        mode_id = (
            _NVFP4_MODE_DIM_KM
            if mode == "dim_km"
            else _NVFP4_MODE_DIM_M
        )

    outer_k = outer_scale.reshape(1) if do_dim_k else None
    outer_m = outer_scale_m.reshape(1) if do_dim_m else None
    seed_k = (
        key.reshape(-1).view(torch.int64)
        if stochastic and do_dim_k
        else None
    )
    seed_m_key = key_m if do_dim_k else key
    seed_m = (
        seed_m_key.reshape(-1).view(torch.int64)
        if stochastic
        else None
    )

    fn = _compile_nvfp4_swizzle_tma(
        tile_m,
        tile_n,
        cluster_n,
        mode_id,
        has_rht,
        stochastic,
    )
    fn(
        input,
        output_k,
        scale_k,
        outer_k,
        output_m,
        scale_m,
        outer_m,
        rht_sign,
        seed_k,
        seed_m,
        M,
        N,
    )

    if do_dim_k:
        output_k = output_k.view(torch.float4_e2m1fn_x2)
        scale_k = scale_k.view(nrb_k, ncb_k, 32, 16).view(torch.float8_e4m3fn)
    if do_dim_m:
        output_m = output_m.view(torch.float4_e2m1fn_x2)
        scale_m = scale_m.view(nrb_m, ncb_m, 32, 16).view(torch.float8_e4m3fn)

    if mode == "dim_k":
        return output_k, scale_k
    if mode == "dim_m":
        return output_m, scale_m
    return output_k, scale_k, output_m, scale_m


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
    device = input.get_device()

    def launch_on_current_device():
        return _nvfp4_swizzle_tma_impl_on_current_device(
            input,
            outer_scale,
            outer_scale_m,
            mode,
            rht_sign,
            key,
            key_m,
            rounding_mode,
            **kwargs,
        )

    if device == torch.cuda.current_device():
        return launch_on_current_device()
    with torch.cuda.device(device):
        return launch_on_current_device()


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
