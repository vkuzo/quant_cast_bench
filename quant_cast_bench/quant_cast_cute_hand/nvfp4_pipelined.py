"""Persistent TMA/UMMA NVFP4 RHT kernels and recipe definitions."""

from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache
from torch._vendor.quack.cache import EXTRA_SOURCE_DIRS

from quant_cast_bench.quant_cast_cute.recipes import QuantCastCuteRecipe
from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma import (
    _NVFP4_DIRECT_HALF,
    _NVFP4_GROUP,
)
from quant_cast_bench.quant_cast_cute_hand.utils import (
    _allocate_nvfp4_swizzle_outputs,
    _ceil_div,
    _nvfp4_load_philox_key,
    _nvfp4_quantize_fast_groups,
    _store_swizzled_scale_groups_as_uint,
    _validate_nvfp4_swizzle_inputs,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    Nvfp4GsSwizzle_DimK_DimMRHT_Gold,
    Nvfp4GsSwizzle_DimKSR_DimMRHTSR_Gold,
)


_NVFP4_PIPELINED_SOURCE_DIR = Path(__file__).resolve().parent
if _NVFP4_PIPELINED_SOURCE_DIR not in EXTRA_SOURCE_DIRS:
    EXTRA_SOURCE_DIRS.append(_NVFP4_PIPELINED_SOURCE_DIR)


_NVFP4_UMMA_INPUT_STAGES = 4
_NVFP4_UMMA_ACC_STAGES = 2
_NVFP4_UMMA_THREADS = 512


@cute.struct
class _Nvfp4UmmaPipelineStorage:
    input_mbars: cute.struct.MemRange[
        cutlass.Int64, _NVFP4_UMMA_INPUT_STAGES * 2
    ]
    acc_mbars: cute.struct.MemRange[
        cutlass.Int64, _NVFP4_UMMA_ACC_STAGES * 2
    ]
    tmem_dealloc_bar: cutlass.Int64
    tmem_ptr: cutlass.Int32


class _Nvfp4UmmaPipelinedRht:
    """Persistent Blackwell TMA/UMMA kernel modeled after TransformerEngine's RHT fusion."""

    def __init__(self, stochastic: bool, tiles_per_cta: int):
        self.stochastic = stochastic
        self.tiles_per_cta = tiles_per_cta
        self.acc_stages = 1 if stochastic else 2
        self.mma_tiler = (128, 16, 128)
        self.cta_tile_shape_mnk = (128, 16, 16)
        self.input_tile = (128, 128)
        self.cluster_shape_mn = (1, 1)
        self.epi_cols = 64 if stochastic else 16
        self.epi_tile = (128, self.epi_cols)
        self.epilogue_warps = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.row_first_warp = 8
        self.row_warps = 8
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.Float32
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        self.use_2cta_instrs = False
        self.use_tma_store = False
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def _make_mma(self):
        return utils.sm100.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            utils.LayoutEnum.COL_MAJOR.mma_major_mode(),
            utils.LayoutEnum.COL_MAJOR.mma_major_mode(),
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutputK: cute.Tensor,
        mScaleK: cute.Tensor,
        mOuterScaleK: cute.Tensor,
        mOutputM: cute.Tensor,
        mScaleM: cute.Tensor,
        mOuterScaleM: cute.Tensor,
        mRhtSign: cute.Tensor,
        mSeed: cute.Tensor | None,
        stream: cuda.CUstream,
        M: cutlass.Int32,
        N: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(self.stochastic):
            assert mSeed is not None
        else:
            assert mSeed is None

        # A is x.T logically: UMMA's M dimension walks original columns and K walks rows.
        mA = cute.make_tensor(
            mInput.iterator,
            cute.make_layout((N, M, 1), stride=(1, N, M * N)),
        )
        tiled_mma = self._make_mma()
        cluster_layout = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_layout = utils.sm100.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            cutlass.BFloat16,
            _NVFP4_UMMA_INPUT_STAGES,
        )
        b_layout = utils.sm100.make_smem_layout_b(
            tiled_mma, (128, 16, 16), cutlass.BFloat16, 1
        )
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        tma_a, tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mA,
            a_one,
            self.mma_tiler,
            tiled_mma,
            cluster_layout.shape,
        )
        num_tma_bytes = cute.size_in_bytes(cutlass.BFloat16, a_one)
        # One accumulator stage owns all eight 128x16 RHT blocks from an input tile. This mirrors
        # TE's epilogue unroll: one producer/consumer handshake covers a full 128x128 result.
        acc_shape = tiled_mma.partition_shape_C(self.input_tile)
        acc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.acc_stages)
        )
        num_tmem_cols = utils.get_num_tmem_alloc_cols(acc_fake, arch="sm_100")

        nrb_k, ncb_k = _ceil_div(M, 128), _ceil_div(N, 64)
        scale_k_layout = cute.make_layout(
            ((32, 4, nrb_k), (4, ncb_k)),
            stride=((16, 4, ncb_k * 32 * 16), (1, 32 * 16)),
        )
        mScaleKLogical = cute.make_tensor(mScaleK.iterator, scale_k_layout)
        nrb_m, ncb_m = _ceil_div(N, 128), _ceil_div(M, 64)
        scale_m_layout = cute.make_layout(
            ((32, 4, nrb_m), (4, ncb_m)),
            stride=((16, 4, ncb_m * 32 * 16), (1, 32 * 16)),
        )
        mScaleMLogical = cute.make_tensor(mScaleM.iterator, scale_m_layout)

        self.kernel(
            tiled_mma,
            tma_a,
            tensor_a,
            mInput,
            mOutputK,
            mScaleKLogical,
            mOuterScaleK,
            mOutputM,
            mScaleMLogical,
            mOuterScaleM,
            mRhtSign,
            mSeed,
            cluster_layout,
            a_layout,
            b_layout,
            num_tma_bytes,
            num_tmem_cols,
            M,
            N,
        ).launch(
            grid=(
                (N // self.input_tile[1]) // self.tiles_per_cta,
                M // self.input_tile[0],
                1,
            ),
            block=(_NVFP4_UMMA_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        mInput: cute.Tensor,
        mOutputK: cute.Tensor,
        mScaleKLogical: cute.Tensor,
        mOuterScaleK: cute.Tensor,
        mOutputM: cute.Tensor,
        mScaleMLogical: cute.Tensor,
        mOuterScaleM: cute.Tensor,
        mRhtSign: cute.Tensor,
        mSeed: cute.Tensor | None,
        cluster_layout: cute.Layout,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        num_tma_bytes: cutlass.Constexpr,
        num_tmem_cols: cutlass.Constexpr,
        M: cutlass.Int32,
        N: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(self.stochastic):
            assert mSeed is not None
        else:
            assert mSeed is None

        tidx, _, _ = cute.arch.thread_idx()
        n_chunk, m_tile, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_tile_base = n_chunk * self.tiles_per_cta

        # The SR epilogue's x64 TMEM load exceeds ptxas' register target if these roles
        # reconverge after setmaxnreg, so keep its register changes inside the exclusive chain.
        if cutlass.const_expr(not self.stochastic):
            if (warp >= 4) & (warp < 8):
                cute.arch.setmaxregister_decrease(32)
            if warp < 4:
                cute.arch.setmaxregister_increase(192)
            if warp >= 8:
                cute.arch.setmaxregister_increase(136)

        smem = utils.SmemAllocator()
        storage = smem.allocate(_Nvfp4UmmaPipelineStorage)
        input_pipeline = pipeline.PipelineTmaMultiConsumersAsync.create(
            barrier_storage=storage.input_mbars.data_ptr(),
            num_stages=_NVFP4_UMMA_INPUT_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            ),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.row_warps
            ),
            tx_count=num_tma_bytes,
            cta_layout_vmnk=cluster_layout,
            defer_sync=True,
            force_deprecated_per_lane_signaling=False,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbars.data_ptr(),
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.epilogue_warps
            ),
            cta_layout_vmnk=cluster_layout,
            defer_sync=True,
        )
        alloc_bar = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=(self.epilogue_warps + 1) * 32,
        )
        dealloc_bar = pipeline.NamedBarrier(
            barrier_id=self.tmem_dealloc_sync_bar_id,
            num_threads=self.epilogue_warps * 32,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_ptr.ptr,
            barrier_for_retrieve=alloc_bar,
            allocator_warp_id=0,
            is_two_cta=False,
        )
        sA = smem.allocate_tensor(
            cutlass.BFloat16,
            a_layout.outer,
            byte_alignment=128,
            swizzle=a_layout.inner,
        )
        sB = smem.allocate_tensor(
            cutlass.BFloat16,
            b_layout.outer,
            byte_alignment=128,
            swizzle=b_layout.inner,
        )

        # Materialize the tiny dense RHT matrix once per CTA. The UMMA B coordinate order is
        # (output, reduction), hence H[n,k] is written as sign[k] * H[k,n] / 4.
        if tidx < _NVFP4_GROUP * _NVFP4_GROUP:
            k = tidx % _NVFP4_GROUP
            n = tidx // _NVFP4_GROUP
            scaled_sign = (
                mRhtSign[k].to(cutlass.Float32) * cutlass.Float32(0.25)
            )
            masked = k & n
            masked = masked ^ (masked >> 2)
            masked = masked ^ (masked >> 1)
            factor = (
                cutlass.Float32(1.0)
                if (masked & 1) == 0
                else cutlass.Float32(-1.0)
            )
            sB[((n, k), 0, 0, 0)] = (
                scaled_sign * factor
            ).to(cutlass.BFloat16)
        pipeline_init_arrive(cluster_shape_mn=cluster_layout, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cluster_layout)
        cute.arch.sync_threads()

        thr_mma = tiled_mma.get_slice(0)
        gA = cute.local_tile(
            mA,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        tCgA = thr_mma.partition_A(gA)
        tAsA, tAgA = cpasync.tma_partition(
            tma_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.input_tile)
        tCtAccFake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.acc_stages)
        )
        single_acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAccSingleFake = tiled_mma.make_fragment_C(single_acc_shape)
        if warp == self.tma_warp:
            if cutlass.const_expr(self.stochastic):
                cute.arch.setmaxregister_decrease(32)
            input_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _NVFP4_UMMA_INPUT_STAGES,
            )
            for tile_offset in cutlass.range(
                self.tiles_per_cta, unroll=1
            ):
                input_pipeline.producer_acquire(input_producer_state)
                cute.copy(
                    tma_a,
                    tAgA[(None, n_tile_base + tile_offset, m_tile, 0)],
                    tAsA[(None, input_producer_state.index)],
                    tma_bar_ptr=input_pipeline.producer_get_barrier(
                        input_producer_state
                    ),
                )
                input_producer_state.advance()
            input_pipeline.producer_tail(input_producer_state)

        elif warp == self.mma_warp:
            if cutlass.const_expr(self.stochastic):
                cute.arch.setmaxregister_decrease(32)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAccFake.layout)
            input_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _NVFP4_UMMA_INPUT_STAGES,
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.acc_stages,
            )
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile_offset in cutlass.range(
                self.tiles_per_cta, unroll=1
            ):
                input_pipeline.consumer_wait(input_consumer_state)
                acc_pipeline.producer_acquire(acc_producer_state)
                for group in cutlass.range_constexpr(8):
                    tCtAccGroup = cute.make_tensor(
                        tmem_ptr
                        + group * 16
                        + acc_producer_state.index * 128,
                        tCtAccSingleFake.layout,
                    )
                    cute.gemm(
                        tiled_mma,
                        tCtAccGroup,
                        tCrA[(None, None, group, input_consumer_state.index)],
                        tCrB[(None, None, 0, 0)],
                        tCtAccGroup,
                    )
                acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                input_pipeline.consumer_release(
                    input_consumer_state, pipeline.PipelineOp.TCGen05Mma
                )
                input_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        elif (warp >= self.row_first_warp) & (
            warp < self.row_first_warp + self.row_warps
        ):
            if cutlass.const_expr(self.stochastic):
                cute.arch.setmaxregister_increase(136)
            frgOuterScaleK = cute.make_rmem_tensor(1, mOuterScaleK.element_type)
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), mOuterScaleK.element_type
                ),
                mOuterScaleK,
                frgOuterScaleK,
            )
            k0_k = cutlass.Uint32(0)
            k1_k = cutlass.Uint32(0)
            counter_base_k = cutlass.Uint64(0)
            if cutlass.const_expr(self.stochastic):
                k0_k, k1_k, counter_base_k = _nvfp4_load_philox_key(mSeed)
            load128 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.BFloat16,
                num_bits_per_copy=128,
            )
            store256 = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(),
                cutlass.Uint8,
                num_bits_per_copy=256,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            local_thread = tidx - self.row_first_warp * 32
            local_row = local_thread // 2
            local_group = (local_thread % 2) * 4
            input_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _NVFP4_UMMA_INPUT_STAGES,
            )
            for tile_offset in cutlass.range(
                self.tiles_per_cta, unroll=1
            ):
                input_pipeline.consumer_wait(input_consumer_state)
                sAFlat = cute.group_modes(sA, 0, 3)
                sTile = cute.tiled_divide(
                    sAFlat[(None, input_consumer_state.index)], (128,)
                )
                # Copy the complete owned fragment before releasing the shared stage, allowing the
                # producer to overlap the next TMA while row quantization runs from registers.
                rInputK = cute.make_rmem_tensor(64, cutlass.BFloat16)
                rInputK8 = cute.tiled_divide(
                    rInputK, (_NVFP4_DIRECT_HALF,)
                )
                sInputK8 = cute.tiled_divide(
                    sTile[(None, local_row)], (_NVFP4_DIRECT_HALF,)
                )
                first_vec = local_group * 2
                for vec in cutlass.range_constexpr(8):
                    cute.copy(
                        load128,
                        sInputK8[(None, first_vec + vec)],
                        rInputK8[(None, vec)],
                    )
                cute.arch.fence_view_async_shared()
                input_pipeline.consumer_release(
                    input_consumer_state, pipeline.PipelineOp.AsyncThread
                )
                input_consumer_state.advance()

                n_tile = n_tile_base + tile_offset
                row_k = m_tile * 128 + local_row
                scale_col_k = n_tile * 8 + local_group
                rValuesK = cute.make_rmem_tensor(64, cutlass.Float32)
                rValuesK.store(rInputK.load().to(cutlass.Float32))
                rQK = cute.make_rmem_tensor(32, cutlass.Uint8)
                rScaleK = cute.make_rmem_tensor(4, cutlass.Uint8)
                counter_start_k = counter_base_k + cutlass.Uint64(
                    row_k * (N // _NVFP4_GROUP) + scale_col_k
                )
                _nvfp4_quantize_fast_groups(
                    rValuesK,
                    rQK,
                    rScaleK,
                    frgOuterScaleK[0],
                    counter_start_k,
                    k0_k,
                    k1_k,
                    4,
                    self.stochastic,
                )
                output_offset_k = cute.assume(
                    row_k * (N // 2) + scale_col_k * 8, divby=32
                )
                cute.copy(
                    store256,
                    rQK,
                    cute.make_tensor(
                        mOutputK.iterator + output_offset_k,
                        cute.make_layout(32),
                    ),
                )
                _store_swizzled_scale_groups_as_uint(
                    mScaleKLogical,
                    rScaleK,
                    row_k,
                    scale_col_k,
                    4,
                )

        elif warp < self.epilogue_warps:
            if cutlass.const_expr(self.stochastic):
                cute.arch.setmaxregister_increase(192)
            tmem.allocate(num_tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAccFake.layout)
            cIdentity = cute.make_identity_tensor((N, M, 1))
            gC = cute.local_tile(
                cIdentity,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            tCgC = thr_mma.partition_C(gC)
            tCtAccEpi = utils.gemm.sm100.transform_partitioned_tensor_layout(
                tCtAcc
            )
            tCgCEpi = utils.gemm.sm100.transform_partitioned_tensor_layout(tCgC)
            tiled_t2r, tTR_tAcc, tTR_rAcc = (
                utils.gemm.sm100.epilogue_tmem_copy_and_partition(
                    self,
                    tidx,
                    tCtAccEpi,
                    tCgCEpi,
                    self.epi_tile,
                    False,
                )
            )
            frgOuterScaleM = cute.make_rmem_tensor(1, mOuterScaleM.element_type)
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), mOuterScaleM.element_type
                ),
                mOuterScaleM,
                frgOuterScaleM,
            )
            k0_m = cutlass.Uint32(0)
            k1_m = cutlass.Uint32(0)
            counter_base_m = cutlass.Uint64(0)
            if cutlass.const_expr(self.stochastic):
                k0_m, k1_m, counter_base_m = _nvfp4_load_philox_key(mSeed)
            store256 = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(),
                cutlass.Uint8,
                num_bits_per_copy=256,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.acc_stages,
            )
            for tile_offset in cutlass.range(
                self.tiles_per_cta, unroll=1
            ):
                n_tile = n_tile_base + tile_offset
                output_row_m = n_tile * self.input_tile[1] + tidx
                acc_pipeline.consumer_wait(acc_consumer_state)
                tAccStage = tTR_tAcc[
                    (
                        None,
                        None,
                        None,
                        None,
                        None,
                        acc_consumer_state.index,
                    )
                ]
                tAccStage = cute.group_modes(
                    tAccStage, 3, cute.rank(tAccStage)
                )
                rValuesM = cute.make_rmem_tensor(128, cutlass.Float32)
                rValueChunksM = cute.tiled_divide(
                    rValuesM, (self.epi_cols,)
                )
                for chunk in cutlass.range_constexpr(
                    128 // self.epi_cols
                ):
                    cute.copy(
                        tiled_t2r,
                        tAccStage[(None, None, None, chunk)],
                        tTR_rAcc,
                    )
                    rValueChunksM[(None, chunk)].store(tTR_rAcc.load())
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                scale_col_m = m_tile * 8
                rQM = cute.make_rmem_tensor(64, cutlass.Uint8)
                rScaleM = cute.make_rmem_tensor(8, cutlass.Uint8)
                # Dim-M follows dim-K's M*N elements in the shared Philox stream. Each counter
                # supplies the random words for 16 FP4 values.
                counter_start_m = (
                    counter_base_m
                    + cutlass.Uint64(N) * cutlass.Uint64(M // _NVFP4_GROUP)
                    + cutlass.Uint64(
                        output_row_m * (M // _NVFP4_GROUP) + scale_col_m
                    )
                )
                _nvfp4_quantize_fast_groups(
                    rValuesM,
                    rQM,
                    rScaleM,
                    frgOuterScaleM[0],
                    counter_start_m,
                    k0_m,
                    k1_m,
                    8,
                    self.stochastic,
                )
                output_offset_m = cute.assume(
                    output_row_m * (M // 2) + scale_col_m * 8,
                    divby=32,
                )
                rQChunksM = cute.tiled_divide(rQM, (32,))
                for chunk in cutlass.range_constexpr(2):
                    cute.copy(
                        store256,
                        rQChunksM[(None, chunk)],
                        cute.make_tensor(
                            mOutputM.iterator
                            + output_offset_m
                            + chunk * 32,
                            cute.make_layout(32),
                        ),
                    )
                _store_swizzled_scale_groups_as_uint(
                    mScaleMLogical,
                    rScaleM,
                    output_row_m,
                    scale_col_m,
                    8,
                )
            dealloc_bar.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        else:
            if cutlass.const_expr(self.stochastic):
                cute.arch.setmaxregister_decrease(32)


def _make_dynamic_matrix_fake(
    dtype,
    *,
    compact_dim_divisibility: int,
    assumed_align: int,
):
    """Match a row-major matrix whose two dimensions are multiples of 128."""
    return cute.runtime.make_fake_tensor(
        dtype,
        (
            cute.sym_int(divisibility=128),
            cute.sym_int(divisibility=compact_dim_divisibility),
        ),
        stride=(
            cute.sym_int64(divisibility=compact_dim_divisibility),
            1,
        ),
        assumed_align=assumed_align,
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


def _nvfp4_pipelined_compile_log_key(
    stochastic: bool,
    tiles_per_cta: int,
) -> str:
    return f"sr={stochastic} tiles_per_cta={tiles_per_cta}"


@instrumented_cutedsl_cache(
    "quant_cast_bench::nvfp4_rht_pipelined",
    key_fn=_nvfp4_pipelined_compile_log_key,
)
def _compile_nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
    stochastic: bool,
    tiles_per_cta: int,
):
    operation = _Nvfp4UmmaPipelinedRht(stochastic, tiles_per_cta)
    mInput = _make_dynamic_matrix_fake(
        cutlass.BFloat16,
        compact_dim_divisibility=128,
        assumed_align=16,
    )
    mOutputK = _make_dynamic_matrix_fake(
        cutlass.Uint8,
        compact_dim_divisibility=64,
        assumed_align=32,
    )
    mScaleK = _make_dynamic_scale_fake()
    mOuterScaleK = _make_static_vector_fake(cutlass.Float32, 1, assumed_align=4)
    mOutputM = _make_dynamic_matrix_fake(
        cutlass.Uint8,
        compact_dim_divisibility=64,
        assumed_align=32,
    )
    mScaleM = _make_dynamic_scale_fake()
    mOuterScaleM = _make_static_vector_fake(cutlass.Float32, 1, assumed_align=4)
    mRhtSign = _make_static_vector_fake(
        cutlass.BFloat16,
        _NVFP4_GROUP,
        assumed_align=16,
    )
    mSeed = (
        _make_static_vector_fake(cutlass.Int64, 2, assumed_align=8)
        if stochastic
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


def _nvfp4_swizzle_dim_k_dim_m_rht_pipelined_on_current_device(
    input,
    outer_scale_k,
    outer_scale_m,
    rht_sign,
    key=None,
    *,
    stochastic,
):
    M, N = _validate_nvfp4_swizzle_inputs(
        input, outer_scale_k, "nvfp4 RHT pipelined", 32
    )
    assert M % 128 == 0 and N % 128 == 0, (
        "nvfp4 RHT pipelined requires M % 128 == 0 and K % 128 == 0"
    )
    assert outer_scale_m.device == input.device
    assert outer_scale_m.dtype == torch.float32 and outer_scale_m.numel() == 1
    assert rht_sign.shape == (16,)
    assert rht_sign.dtype == torch.bfloat16 and rht_sign.device == input.device
    assert rht_sign.is_contiguous()
    if stochastic:
        assert key is not None, "stochastic rounding requires a key"
        assert key.device == input.device
        assert key.dtype == torch.uint64 and key.numel() == 2
    else:
        assert key is None

    output_k, scale_k, nrb_k, ncb_k = _allocate_nvfp4_swizzle_outputs(
        input, M, N
    )
    output_m, scale_m, nrb_m, ncb_m = _allocate_nvfp4_swizzle_outputs(
        input, N, M
    )
    tiles_m = M // 128
    tiles_n = N // 128
    # Amortize the fixed RHT/TMEM setup while retaining enough CTAs to smooth producer/consumer
    # imbalance. Parallel RHT setup makes 16 tiles/CTA optimal for both RTNE and SR.
    max_tiles_per_cta = 16
    tiles_per_cta = next(
        candidate
        for candidate in (32, 16, 8, 4, 2, 1)
        if candidate <= max_tiles_per_cta
        and tiles_n % candidate == 0
        and (candidate == 1 or tiles_m * tiles_n // candidate >= 128)
    )
    seed = key.reshape(-1).view(torch.int64) if stochastic else None
    fn = _compile_nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
        stochastic,
        tiles_per_cta,
    )
    fn(
        input,
        output_k,
        scale_k,
        outer_scale_k.reshape(1),
        output_m,
        scale_m,
        outer_scale_m.reshape(1),
        rht_sign,
        seed,
        M,
        N,
    )
    return (
        output_k.view(torch.float4_e2m1fn_x2),
        scale_k.view(nrb_k, ncb_k, 32, 16).view(torch.float8_e4m3fn),
        output_m.view(torch.float4_e2m1fn_x2),
        scale_m.view(nrb_m, ncb_m, 32, 16).view(torch.float8_e4m3fn),
    )


def _nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
    input,
    outer_scale_k,
    outer_scale_m,
    rht_sign,
    key=None,
    *,
    stochastic,
):
    device = input.get_device()

    def launch_on_current_device():
        return _nvfp4_swizzle_dim_k_dim_m_rht_pipelined_on_current_device(
            input,
            outer_scale_k,
            outer_scale_m,
            rht_sign,
            key,
            stochastic=stochastic,
        )

    if device == torch.cuda.current_device():
        return launch_on_current_device()
    with torch.cuda.device(device):
        return launch_on_current_device()


def nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
    input, outer_scale_k, outer_scale_m, rht_sign, **kwargs
):
    return _nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
        input,
        outer_scale_k,
        outer_scale_m,
        rht_sign,
        stochastic=False,
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
    return _nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
        input,
        outer_scale_k,
        outer_scale_m,
        rht_sign,
        key,
        stochastic=True,
    )


NVFP4_SWIZZLE_DIM_K_SR_DIM_M_RHT_SR_PIPELINED = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzle_DimKSR_DimMRHTSR_Gold,
    cute_fn=nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined,
)
