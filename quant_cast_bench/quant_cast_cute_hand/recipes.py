# handwritten cute recipes, tracking learning CuTeDSL
#
# Started from FP8_DEEPSEEK_1X128, copied verbatim from quant_cast_cute/recipes.py; this is the
# playground where we iterate on it.

import os
from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync  # TMA copies
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

import torch

# Gate debug output (host trace-time `print` + device `cute.printf`) behind an env var, read once
# at import. Gate with `cutlass.const_expr(_DEBUG)` inside kernels so that when off the tracer takes
# neither branch -- the printf ops are never emitted (no dead ops, no values kept live). Run with
# `CUTE_DEBUG=1 python -m ...` to enable.
_DEBUG = os.environ.get("CUTE_DEBUG", "0") == "1"

from quant_cast_bench.quant_cast_cute.recipes import (
    QuantCastCuteRecipe,
    _philox_4x32,
)
from quant_cast_bench.quant_cast_cute_hand.mxfp8_v2 import (
    MXFP8_32X32_SWIZZLE_V2,
    MXFP8_DIM_KM_SWIZZLE_SR_V2,
    MXFP8_DIM_KM_SWIZZLE_V2,
    MXFP8_DIM_M_SWIZZLE_SR_V2,
    MXFP8_DIM_M_SWIZZLE_V2,
    MXFP8_SWIZZLE_SR_V2,
    MXFP8_SWIZZLE_V2,
)
from quant_cast_bench.quant_cast_cute_hand.nvfp4_pipelined import (
    NVFP4_SWIZZLE_DIM_K_DIM_M_RHT_PIPELINED,
    NVFP4_SWIZZLE_DIM_K_SR_DIM_M_RHT_SR_PIPELINED,
    nvfp4_swizzle_dim_k_dim_m_rht_pipelined,
    nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined,
)
from quant_cast_bench.quant_cast_cute_hand.nvfp4_tma import (
    NVFP4_DIM_KM_SWIZZLE_TMA,
    NVFP4_DIM_M_RHT_SWIZZLE_TMA,
    NVFP4_DIM_M_SWIZZLE_RHT_SR_TMA,
    NVFP4_DIM_M_SWIZZLE_TMA,
    NVFP4_SWIZZLE_DIM_K_DIM_M_RHT_TMA,
    NVFP4_SWIZZLE_DIM_K_SR_DIM_M_RHT_SR_TMA,
    NVFP4_SWIZZLE_TMA,
    _NVFP4_MODE_DIM_K,
    _NVFP4_MODE_DIM_KM,
    _NVFP4_MODE_DIM_M,
    _NVFP4_TMA_THREADS,
    _allocate_nvfp4_swizzle_outputs,
    _nvfp4_tma_dim_m_column,
    _validate_nvfp4_swizzle_inputs,
    nvfp4_dim_km_swizzle_tma,
    nvfp4_dim_m_rht_swizzle_tma,
    nvfp4_dim_m_swizzle_rht_sr_tma,
    nvfp4_dim_m_swizzle_tma,
    nvfp4_swizzle_dim_k_dim_m_rht_tma,
    nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_tma,
    nvfp4_swizzle_tma,
)
from quant_cast_bench.quant_cast_cute_hand.utils import (
    COMPILE_CACHE,
    _ceil_div,
    _cvt_e4m3x2_to_f32x2,
    _cvt_rn_satfinite_e4m3x2_f32_packed,
    _cvt_rs_satfinite_e2m1x4_f32_x8,
    _compiled,
    _cvt_rs_satfinite_e4m3x4_f32,
    _e8m0,
    _e8m0_scale_store_as_uint,
    _nvfp4_load_philox_key,
    _nvfp4_quantize_fast_groups,
    _nvfp4_quantize_stochastic_x16,
    _nvfp4_quantize_x16,
    _nvfp4_rht_fwht_x16,
    _nvfp4_scale_e4m3_fast_x2,
    _nvfp4_store_scale_groups,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    Deepseek1x128Gold, Deepseek1x128DimMGold, Mxfp8SwizzleGold, Mxfp8SwizzleSRGold,
    Nvfp4GsSwizzleGold,
)

@cute.kernel
def add_v0_kernel(input: cute.Tensor, num: cutlass.Float32, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()  # thread index in block (0 to bdim-1)
    bidx, _, _ = cute.arch.block_idx()  # block index in grid (0 to grid_dim -1)
    bdim, _, _ = cute.arch.block_dim()  # threads per block


    # global thread_id
    global_tidx = bidx * bdim + tidx

    # if cutlass.dynamic_expr(global_tidx == 0):
    #     cute.printf("hello from global thread %d", global_tidx)

    # global element index for this thread
    m, n = input.shape
    ni = global_tidx % n
    mi = global_tidx // n

    # skip out of bounds
    if global_tidx < m * n:

        # elementwise add 1
        input_val = input[mi, ni]
        # if cutlass.dynamic_expr(tidx == 1):
        #     cute.printf("m %d n %d", m, n)
        #     cute.printf("global_tidx %d, mi %d, ni %d, val %f", global_tidx, mi, ni, input_val)
        output_val = input_val + num
        output[mi, ni] = output_val


@cute.jit
def add_v0_jit(input: cute.Tensor, num: float, output: cute.Tensor):
    kernel = add_v0_kernel(input, num, output)

    # naive - each thread does one element
    m, n = input.shape
    numel = m * n
    num_threads = numel

    # H100 numbers on 16384x16384 tensor
    # num_threads_per_block -> pct_peak
    # 1 -> 0.4%
    # 32 -> 12.7%
    # 128 -> 50.7%
    # 256 -> 66.6%
    # 512 -> 61.6%
    # 1024 -> 55.6%
    #
    # Rise (1 -> 256): an SM hides memory latency with resident warps (max 64 warps / 2048 threads
    # on H100). More threads/block => more resident warps => more loads in flight => closer to peak.
    # 256 (8 warps) is enough to nearly max out latency hiding.
    #
    # Plateau+dip (256 -> 1024): 256/512/1024 all hit full occupancy (2048 divides evenly), so past
    # 256 there is NO extra latency hiding to gain -- bigger blocks just repack the same 2048 threads
    # into fewer resident blocks per SM (8 -> 4 -> 2). Fewer resident blocks costs a few points via:
    #   (a) drain/refill bubbles -- blocks launched together retire together, so with only 2 big
    #       blocks the SM briefly idles at wave boundaries; 8 small blocks stagger and stay busy;
    #   (b) burstier traffic -- 32 warps of one block march load/wait/store in lockstep (bursts with
    #       gaps), whereas many small blocks desync and interleave into a smoother memory stream;
    #   (c) coarser wave quantization -- fewer/larger blocks waste proportionally more SM-time in the
    #       final partial wave.
    # Confirm with ncu (achieved occupancy ~flat, DRAM throughput drops). The ~66% ceiling itself is
    # the 1-elem/thread float32 load (1 outstanding load/thread); vectorizing (128-bit loads) raises
    # the ceiling -- block size only moves you along it.
    num_threads_per_block = 256

    num_blocks = _ceil_div(numel, num_threads_per_block)

    kernel.launch(
        # grid - number of thread blocks (CTAs) per launch
        grid=(num_blocks, 1, 1),
        # block - number of threads per thread_block. Each block runs on one SM,
        # and threads execute in warps of 32.
        block=(num_threads_per_block, 1, 1),
    )


def add_v0(input: torch.Tensor, num: float):
    # elementwise add, each thread handles one element

    output = torch.empty_like(input) 
    input_cute = from_dlpack(input)
    output_cute = from_dlpack(output)
    add_v0_jit(input_cute, num, output_cute)
    return output


@cute.kernel
def add_v1_kernel(gA: cute.Tensor, num: cutlass.Float32, gB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()  # thread index in block (0 to bdim-1)
    bidx, _, _ = cute.arch.block_idx()  # block index in grid (0 to grid_dim -1)
    bdim, _, _ = cute.arch.block_dim()  # threads per block


    thread_idx = bidx * bdim + tidx


    # map thread index to logical index of input tensor, in unit of vector
    m, n = gA.shape[1]
    num_threads = cute.size(gA, mode=1)
    # if tidx == 0 and bidx == 0:
    #     cute.printf("thread 0 block 0")
    #     cute.printf("m %d n %d", m, n)
    #     cute.printf("num_threads %d", num_threads)

    # only calculate for in bounds tiles
    if thread_idx < num_threads:

        ni = thread_idx % n
        mi = thread_idx // n

        # map logical index to physical address via tensor layout
        a_val = gA[(None, (mi, ni))].load()
        if tidx == 0 and bidx == 0:
            # tensor<ptr<f32, gmem> o ((1,4)):((0,1))>
            # print(f"sliced gA = {gA[(None, (mi, ni))]}")
            # tensor_value<vector<4xf32> o ((1, 4),)>
            # print(a_val)
            pass

        gB[(None, (mi, ni))] = a_val + num



@cute.jit
def add_v1_jit(mA: cute.Tensor, num: float, mB: cute.Tensor):

    m, n = mA.shape
    numel = m * n
    num_threads = numel

    # 256 is a reasonable default
    num_threads_per_block = 256

    # tensor<ptr<f32, gmem> o (2,64):(64,1)>
    # print("mA", mA)

    # we are in fp32, so 4 elements per thread to get 128 bit load/store
    el_per_thread = 4
    gA = cute.zipped_divide(mA, (1, el_per_thread))
    gB = cute.zipped_divide(mB, (1, el_per_thread))
    # mA - "m=source matrix A, in global memory"
    # gA - "g=global memory of matrix A, tiled by convention"

    # gA tensor<ptr<f32, gmem> o ((1,4),(2,16)):((0,1),(64,4))>
    #                               |------ mode0--|
    #                                     |--------mode1--|
    #                             shape0,shape1:strides0,strides1
    # print("gA", gA)
    # print("gB", gB)

    # print(f'numel: {numel}, num_threads: {cute.size(gA, mode=[1])}, el_per_thread: {cute.size(gA, mode=[0])}')

    num_blocks = _ceil_div(cute.size(gA, mode=[1]), num_threads_per_block)
    add_v1_kernel(gA, num, gB).launch(
        # grid - number of thread blocks (CTAs) per launch
        grid=(num_blocks, 1, 1),
        # block - number of threads per thread_block. Each block runs on one SM,
        # and threads execute in warps of 32.
        block=(num_threads_per_block, 1, 1),
    )


def add_v1(input: torch.Tensor, num: float):
    # v1 - each thread does a 128-bit load and store
    # for fp32, that's 4 elements per thread

    # for now, no ragged shapes
    assert len(input.shape) == 2, "unsupported"
    assert input.shape[-1] % 4 == 0, "unsupported"

    output = torch.empty_like(input) 
    input_cute = from_dlpack(input, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)
    add_v1_jit(input_cute, num, output_cute)
    return output



@cute.kernel
def add_v2_kernel(
    gA: cute.Tensor, 
    num: cutlass.Float32, 
    gB: cute.Tensor, 
    gIdA: cute.Tensor,
    tv_layout: cute.Layout,
    orig_shape: cute.Shape,
):
    # prints in this function are for input M, N == 4, 1024

    tidx, _, _ = cute.arch.thread_idx()  # thread index in block (0 to bdim-1)
    bidx, _, _ = cute.arch.block_idx()  # block index in grid (0 to grid_dim -1)
    # bdim, _, _ = cute.arch.block_dim()  # threads per block

    # slice for thread-block level view
    # note: "select everything in the 1d tile, for tile index bidx"
    blk_coord = ((None,), bidx)

    # logical coord -> address
    blkA = gA[blk_coord]  # (1024,) -> physical address
    blkB = gB[blk_coord]
    blkIdA = gIdA[blk_coord]

    # compose for thread-index & value-index to physical mapping
    # blockA: (1024,) logical index in tile -> physical address
    # tv_layout: (tid, vid) -> (1024,) logical index in tile
    # Note: composition(blkA, tv_layout) is blkA(tv_layout(input)), NOT tv_layout(blkA(input))
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgIdA = cute.composition(blkIdA, tv_layout)

    if False and (tidx == 0 and bidx == 0):
        # raw_ptr(0x00007f12d9e00000: f32, gmem, align<16>) o ((1024),(4)):((1),(1024))
        cute.printf("gA {}", gA)
        # ((_),0)
        cute.printf("blk_coord {}", blk_coord)
        # raw_ptr(0x00007f12d9e00000: f32, gmem, align<16>) o (1024):(1)
        cute.printf("blkA {}", blkA)

        # print("Composed with TV layout:")
        # tidfrgA: tensor<ptr<f32, gmem, align<16>> o (256,4):(4,1)>
        # print(f"  tidfrgA: {tidfrgA}")
        pass


    # slice for thread-level view
    thr_coord = (tidx, None)

    # mask out of bounds
    thrCrd = tidfrgIdA[thr_coord]
    if cute.elem_less(thrCrd[0], orig_shape):

        # slice for threads: vid -> address
        thrA = tidfrgA[thr_coord]
        thrB = tidfrgB[thr_coord]
        thrB[None] = thrA.load() + num


@cute.jit
def add_v2_jit(mA: cute.Tensor, num: float, mB: cute.Tensor):
    # 256 is a reasonable default
    num_threads_per_block = 256

    # build the tv layout

    # 1. thread arrangement over the tile -> (256):(1)
    # shape (256,): the block's 256 threads laid out as 256 cols
    # order (0,) means 0th column is fastest
    thr_layout = cute.make_ordered_layout((num_threads_per_block,), order=(0,))
    # (256):(1)
    # print('thr_layout', thr_layout)
    assert cute.size(thr_layout) == num_threads_per_block

    # 2. each thread's private chunk, expressed in elements
    # Note: this will have to change for different dtypes.
    # Can modify to bytes (as tutorials do) to stay independent of dtype.
    val_layout = cute.make_ordered_layout((4,), order=(0,))
    # (4):(1)
    # print('val_layout', val_layout)

    # fuse thread-arrangement x per-thread-values into the TV layout + tiler
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # (1024,)
    # - note: (1024,) = (256*4,), i.e. product of thr_layout.shape and val_layout.shape
    # i.e. the MxN region of the source that one CTA (256 threads x 4 values each) covers
    # print('tiler_mn', tiler_mn)

    # (256,4):(4,1) == (thread, value) -> offset into the (1024,) tile
    #   mode0 (256):(4) = THREAD (size 256): a linear tid splits to (tid%256);
    #   mode1 4:1 = VALUE (size 4): a thread's 4 values step by 1
    # print('tv_layout', tv_layout)

    # tensor<ptr<f32, gmem, align<16>> o (128):(1)>
    # print('mA', mA)

    # identity version of the original tensor
    mIdA = cute.make_identity_tensor(mA.shape)

    # ((TileM,), (RestM,))
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gIdA = cute.zipped_divide(mIdA, tiler_mn)


    # tensor<ptr<f32, gmem, align<16>> o ((1024),(1)):((1),(0))>
    # mode0 (1024):(1) = one tile, mode 1 (1):(0) - grid of tiles
    # print('gA', gA)

    # identity tensor for out of bounds check
    # print('gIdA', gIdA)

    add_v2_kernel(gA, num, gB, gIdA, tv_layout, mA.shape).launch(
        # grid - number of thread blocks (CTAs) per launch
        grid=(cute.size(gA, mode=[1]), 1, 1),
        # block - number of threads per thread_block
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
    )


def add_v2(input: torch.Tensor, num: float):
    # v2 - same as v1, but using tv layout
    assert len(input.shape) == 2, "unsupported"
    assert input.shape[-1] % 4 == 0, "unsupported"
    assert input.is_contiguous(), "unsupported"
    M, N = input.shape
    input = input.view(-1)
    output = torch.empty_like(input) 
    input_cute = from_dlpack(input, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)
    add_v2_jit(input_cute, num, output_cute)
    return output.view(M, N)


@cute.kernel
def transpose_v0_kernel(
    gA: cute.Tensor, 
    A_tv_layout: cute.Layout,
    gB: cute.Tensor, 
    B_tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()  # thread index in block (0 to bdim-1)
    bidx, bidy, _ = cute.arch.block_idx()  # block index in grid (0 to grid_dim -1)
    # bdim, _, _ = cute.arch.block_dim()  # threads per block

    # select everything in the 2d tile, for tile index (bidx, bidy)
    blk_coord_A = ((None, None), (bidx, bidy))
    # logical coord -> address
    blkA = gA[blk_coord_A]

    # output - swap block indices to transpose the tiles
    blk_coord_B = ((None, None), (bidy, bidx))
    blkB = gB[blk_coord_B]

    tidfrgA = cute.composition(blkA, A_tv_layout)
    tidfrgB = cute.composition(blkB, B_tv_layout)

    # slice for thread-level view
    thr_coord = (tidx, None)

    # do the transpose, the in-tile value transpose is handled with
    # the layout
    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrB[None] = thrA.load()

    if cutlass.const_expr(_DEBUG):
        # entire tensor
        # tensor<ptr<bf16, gmem, align<16>> o ((1,2048),(2,2)):((0,1),(4096,2048))>
        print('gA', gA)
        # the current block
        # tensor<ptr<bf16, gmem, align<16>> o (1,2048):(0,1)>
        print('blkA', blkA)
        # the current block, in tv-layout
        # tensor<ptr<bf16, gmem, align<16>> o (256,8):(8,1)>
        print('tidfrgA', tidfrgA)
        # the current thread's data
        # tensor<ptr<bf16, gmem, align<16>> o (8):(1)>
        print('thrA', thrA)


@cute.jit
def transpose_v0_jit(mA: cute.Tensor, mB: cute.Tensor):
    # 256 is a reasonable default
    num_threads_per_block = 256

    # for now, a simple 2d layout

    # thr: (128,2):(2,1)
    thrA_layout = cute.make_ordered_layout((128, num_threads_per_block // 128,), order=(1, 0))
    # val: (1,8):(0,1)
    valA_layout = cute.make_ordered_layout((1, 8,), order=(1, 0))
    # (128,16)  ((2,128),8):((1024,1),128)
    tilerA_mn, A_tv_layout = cute.make_layout_tv(thrA_layout, valA_layout)
    # ((TileM,), (RestM,))
    gA = cute.zipped_divide(mA, tilerA_mn)

    # layout of B is transpose of layout of A
    # thr: (2,128):(1,2)
    thrB_layout = cute.make_ordered_layout((num_threads_per_block // 128, 128), order=(0, 1))
    # val: (8,1):(1,0)
    valB_layout = cute.make_ordered_layout((8, 1), order=(0, 1))
    # (16,128) (256,8):(8,1)
    tilerB_mn, B_tv_layout = cute.make_layout_tv(thrB_layout, valB_layout)
    gB = cute.zipped_divide(mB, tilerB_mn)

    if cutlass.const_expr(_DEBUG):
        print('thrA_layout', thrA_layout)
        print('valA_layout', valA_layout)
        print('tilerA_mn', tilerA_mn)
        print('A_tv_layout', A_tv_layout)
        print('thrB_layout', thrB_layout)
        print('valB_layout', valB_layout)
        print('tilerB_mn', tilerB_mn)
        print('B_tv_layout', B_tv_layout)
        print('gA', gA)
        print('gB', gB)

    transpose_v0_kernel(gA, A_tv_layout, gB, B_tv_layout).launch(
        # grid - number of thread blocks (CTAs) per launch
        grid=(cute.size(gA, mode=[1, 0]), cute.size(gA, mode=[1, 1]), 1),
        # block - threads per block instance
        block=(cute.size(A_tv_layout, mode=[0]), 1, 1),
    )

def transpose_v0(input: torch.Tensor):
    assert len(input.shape) == 2, "unsupported"
    assert input.is_contiguous(), "unsupported"
    M, N = input.shape
    assert M % 128 == 0, "unsupported"
    assert N % 16 == 0, "unsupported"
    output = torch.empty(N, M, dtype=input.dtype, device=input.device)
    input_cute = from_dlpack(input, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)
    transpose_v0_jit(input_cute, output_cute)
    return output


@cute.kernel
def transpose_v1_kernel(
    gA: cute.Tensor, 
    A_tv_layout: cute.Layout,
    gB: cute.Tensor, 
    B_tv_layout: cute.Layout,
    sScratch_layout: cute.Layout,
    sScratchT_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()  # thread index in block (0 to bdim-1)
    bidx, bidy, _ = cute.arch.block_idx()  # block index in grid (0 to grid_dim -1)
    # bdim, _, _ = cute.arch.block_dim()  # threads per block

    # slice for thread-level view
    thr_coord = (tidx, None)

    # create shared memory scratchpad
    # raw smem allocation: cosize() = footprint in elements (2048 here), 16 is alignment hint
    sScratch_ptr = cute.arch.alloc_smem(cutlass.BFloat16, cute.cosize(sScratch_layout), 16)
    sScratch = cute.make_tensor(sScratch_ptr, sScratch_layout)

    # select everything in the 2d tile, for tile index (bidx, bidy)
    blk_coord_A = ((None, None), (bidx, bidy))
    # logical coord -> address
    blkA = gA[blk_coord_A]
    tidfrgA = cute.composition(blkA, A_tv_layout)
    thrA = tidfrgA[thr_coord]

    # write to scratchpad
    tidfrgScratch = cute.composition(sScratch, A_tv_layout)
    thrScratch = tidfrgScratch[thr_coord]
    thrScratch[None] = thrA.load()

    # sync this CTA's threads
    cute.arch.sync_threads()

    # end of phase 1, start of phase 2

    # transpose the scratch pad
    sScratchT = cute.make_tensor(sScratch_ptr, sScratchT_layout)

    # select this thread's scratchpad region
    tidfrgScratchT = cute.composition(sScratchT, B_tv_layout)
    thrScratchT = tidfrgScratchT[thr_coord]

    # select the write region, with transposed block indices
    blk_coord_B = ((None, None), (bidy, bidx))
    blkB = gB[blk_coord_B]
    tidfrgB = cute.composition(blkB, B_tv_layout)
    thrB = tidfrgB[thr_coord]

    # do the write
    thrB[None] = thrScratchT.load()

    if cutlass.const_expr(_DEBUG):
        # entire tensor
        # tensor<ptr<bf16, gmem, align<16>> o ((1,2048),(2,2)):((0,1),(4096,2048))>
        print('gA', gA)
        # the current block
        # tensor<ptr<bf16, gmem, align<16>> o (1,2048):(0,1)>
        print('blkA', blkA)
        # the current block, in tv-layout
        # tensor<ptr<bf16, gmem, align<16>> o (256,8):(8,1)>
        print('tidfrgA', tidfrgA)
        # the current thread's data
        # tensor<ptr<bf16, gmem, align<16>> o (8):(1)>
        print('thrA', thrA)
        
        print('sScratch_layout', sScratch_layout)
        print('sScratch', sScratch)
        print('sScratchT', sScratchT)
        print('thrScratch', thrScratch)
        if tidx == 0 and bidx == 0 and bidy == 0:
            cute.printf('thrScratch {} {} {} {}', thrScratch[0], thrScratch[1], thrScratch[2], thrScratch[3])


@cute.jit
def transpose_v1_jit(mA: cute.Tensor, mB: cute.Tensor):
    # 256 is a reasonable default
    num_threads_per_block = 256

    # for now, a simple 2d layout

    # thr: (16,16):(16,1)
    thrA_layout = cute.make_ordered_layout((16, num_threads_per_block // 16,), order=(1, 0))
    # val: (1,8):(0,1)
    valA_layout = cute.make_ordered_layout((1, 8,), order=(1, 0))
    # (16,128)  ((16,16),8):((128,1),16)
    tilerA_mn, A_tv_layout = cute.make_layout_tv(thrA_layout, valA_layout)
    # ((TileM,), (RestM,))
    gA = cute.zipped_divide(mA, tilerA_mn)

    # layout of the shared memory scratchpad
    sScratch_layout = cute.make_ordered_layout(tilerA_mn, order=(1, 0))
    sScratchT_layout = cute.make_ordered_layout((tilerA_mn[1], tilerA_mn[0]), order=(0, 1))

    # now, thinking through the read-write of stage 2
    # scratchpad shape: (16*1, 16*8) = (16, 128) = (TM, TN)
    # scratchpad write for phase 1: thr (16,16):(16,1), val (1,8):(0,1)
    # transpose scratchpad (TM, TN) -> (TN, TM)
    # scratchpad read for phase 2: = (TN, TM) = (128, 16)
    #   each thread handles 8 values
    #   thr (128,2):(2,1), val (1,8):(0,1)
    thrB_layout = cute.make_ordered_layout((128, num_threads_per_block // 128), order=(1, 0))
    valB_layout = cute.make_ordered_layout((1, 8), order=(1, 0))
    tilerB_mn, B_tv_layout = cute.make_layout_tv(thrB_layout, valB_layout)
    gB = cute.zipped_divide(mB, tilerB_mn)

    if cutlass.const_expr(_DEBUG):
        print('thrA_layout', thrA_layout)
        print('valA_layout', valA_layout)
        print('tilerA_mn', tilerA_mn, type(tilerA_mn))
        print('A_tv_layout', A_tv_layout, type(A_tv_layout))
        print('thrB_layout', thrB_layout)
        print('valB_layout', valB_layout)
        print('tilerB_mn', tilerB_mn)
        print('B_tv_layout', B_tv_layout)
        print('sScratch_layout', sScratch_layout)
        print('sScratchT_layout', sScratchT_layout)
        print('gA', gA)
        print('gB', gB)

    transpose_v1_kernel(
        gA, A_tv_layout, gB, B_tv_layout, sScratch_layout, sScratchT_layout
    ).launch(
        # grid - number of thread blocks (CTAs) per launch
        grid=(cute.size(gA, mode=[1, 0]), cute.size(gA, mode=[1, 1]), 1),
        # block - threads per block instance
        block=(cute.size(A_tv_layout, mode=[0]), 1, 1),
    )

def transpose_v1(input: torch.Tensor):
    # Phase 2 reads the smem scratchpad down columns (the transpose) -> classic bank conflicts
    # (~2.17M shared-load conflicts). A Swizzle<3,3,5> over the scratchpad kills them (~30x), diff:
    #   https://gist.github.com/vkuzo/487b4f2ede42be8167638f13182d235c
    # Padding (row pitch W+pad) did NOT work: the read has two collisions -- (1) an inter-group
    # collision that padding only *shifts* by (4*pad) mod 32 banks (so pad=8 was a null shift; pad=4
    # only halved 4-way -> 2-way), and (2) two bf16 elements sharing one 32-bit bank word, which no
    # pad can separate (it acts at >=4-byte granularity). pad=1 fixed the shift but its odd pitch
    # misaligned the 128-bit store -> ~4.5M store conflicts. Swizzle sidesteps all of this.
    # We did NOT include the swizzle: it is not a win for our 16x128 tile (v1 is DRAM-bandwidth-bound
    # there, so the conflicts hide behind gmem), only helping tall-skinny (conflict-bound) tiles.
    # A tile sweep at 16384x16384 (noted for future reference):
    #   tile (TM x TN)   swizzle OFF   swizzle ON      d
    #   8 x 256          29.7%         29.9%          +0.2
    #   16 x 128         85.3%         85.4%          ~0    <- our tile
    #   32 x 64          84.6%         84.9%          +0.3
    #   64 x 32          66.3%         74.1%          +7.8
    #   128 x 16         48.4%         72.9%          +24.5
    #   256 x 8          26.2%         62.6%          +36.4
    assert len(input.shape) == 2, "unsupported"
    assert input.is_contiguous(), "unsupported"
    M, N = input.shape
    assert M % 16 == 0, "unsupported"
    assert N % 128 == 0, "unsupported"
    output = torch.empty(N, M, dtype=input.dtype, device=input.device)
    input_cute = from_dlpack(input, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)
    transpose_v1_jit(input_cute, output_cute)
    return output


@cute.kernel
def fp8_deepseek_1x128_kernel(
    gInput: cute.Tensor,
    gOutput: cute.Tensor,
    gScale: cute.Tensor,
    gId: cute.Tensor,
    input_tv_layout: cute.Layout,
    scale_tv_layout: cute.Layout,
    orig_shape: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()  # thread index in block (0 to bdim-1)
    bidx, _, _ = cute.arch.block_idx()  # block index in grid (0 to grid_dim -1)
    # bdim, _, _ = cute.arch.block_dim()  # threads per block

    # slice for thread-block level view
    # note: "select everything in the 1d tile, for tile index bidx"
    blk_coord = ((None,), bidx)

    # logical coord -> address
    blkInput = gInput[blk_coord]  # (1024,) -> physical address
    blkOutput = gOutput[blk_coord]
    blkId = gId[blk_coord]
    blkScale = gScale[blk_coord]

    # compose for thread-index & value-index to physical mapping
    # blkInput: (1024,) logical index in tile -> physical address
    # input_tv_layout: (tid, vid) -> (1024,) logical index in tile
    # Note: composition(blkInput, input_tv_layout) is blkInput(input_tv_layout(input)), NOT input_tv_layout(blkInput(input))
    tidfrgInput = cute.composition(blkInput, input_tv_layout)
    tidfrgOutput = cute.composition(blkOutput, input_tv_layout)
    tidfrgId = cute.composition(blkId, input_tv_layout)
    tidfrgScale = cute.composition(blkScale, scale_tv_layout)

    # slice for thread-level view
    thr_coord = (tidx, None)

    # mask out of bounds
    thrId = tidfrgId[thr_coord]
    if cute.elem_less(thrId[0], orig_shape):

        # reference:
        #
        # def deepseek_1x128_f(x, **kwargs):
        #     fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        #     *lead, last = x.shape
        #     x_b = x.reshape(*lead, last // 128, 128)
        #     amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float32)
        #     scale = (amax / fp8_max).to(torch.float32)  # forward scale
        #     qdata = (x_b.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
        #     return qdata.reshape(*lead, last), scale.squeeze(-1)
        #
        # Each thread owns 8 elements, 128 // 8 = 16, so every 16 threads
        # calculate a 1x128 block

        # Load the fragment, slice for threads: vid -> address
        thrInput = tidfrgInput[thr_coord].load()

        # docs: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_math.html#cutlass.cute.math.abs
        thrA_abs = cute.math.absf(thrInput)
        # thread-local max with clamp
        thrA_amax8 = thrA_abs.reduce(cute.ReductionOp.MAX, init_val=1e-12, reduction_profile=1)
        # intra-thread max
        thrA_amax128 = cute.arch.warp_reduction_max(thrA_amax8, threads_in_group=16)
        # convert bf16 -> fp32
        thrA_amax128_fp32 = cutlass.Float32(thrA_amax128)
        # calculate scale
        scale = (thrA_amax128_fp32 * cutlass.Float32(1.0 / 448.0))
        # calculate qdata
        qdata = (thrInput.to(cutlass.Float32) * (1.0 / scale)).to(cutlass.Float8E4M3FN)

        if cutlass.const_expr(_DEBUG):
            print('thrA_amax8 static type', thrA_amax8.type)
            print('thrA_amax128_fp32 static type', thrA_amax128_fp32.dtype)
            print('scale static type', scale.dtype)
            if tidx == 0 and bidx == 0:
                cute.printf(
                    "thrInput {}, {}, {}, {}, {}, {}, {}, {}",
                    thrInput[0], thrInput[1], thrInput[2], thrInput[3],
                    thrInput[4], thrInput[5], thrInput[6], thrInput[7],
                )
                cute.printf(
                    "thrA_abs {}, {}, {}, {}, {}, {}, {}, {}",
                    thrA_abs[0], thrA_abs[1], thrA_abs[2], thrA_abs[3],
                    thrA_abs[4], thrA_abs[5], thrA_abs[6], thrA_abs[7],
                )
                cute.printf("thrA_amax8 {}", thrA_amax8)
                cute.printf("thrA_amax128 {}", thrA_amax128)
                cute.printf("scale {}", scale)
                cute.printf(
                    "qdata {}, {}, {}, {}, {}, {}, {}, {}",
                    qdata[0], qdata[1], qdata[2], qdata[3],
                    qdata[4], qdata[5], qdata[6], qdata[7],
                )

        # store qdata
        thrOutput = tidfrgOutput[thr_coord]
        thrOutput[None] = qdata
        # every 16'th thread stores scale
        if tidx % 16 == 0:
            tidfrgScale[tidx] = scale


@cute.jit
def fp8_deepseek_1x128_jit(mInput: cute.Tensor, mOutput: cute.Tensor, mScale: cute.Tensor):
    # 256 is a reasonable default
    num_threads_per_block = 256

    # build the tv layout
    # thr: (256):(1)
    thr_layout = cute.make_ordered_layout((num_threads_per_block,), order=(0,))
    # val: (8):(1)
    val_layout = cute.make_ordered_layout((8,), order=(0,))
    # (2048,)  (256,8):(8,1)
    tiler_mn, input_tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    # ((TileM,), (RestM,))
    gInput = cute.zipped_divide(mInput, tiler_mn)
    gOutput = cute.zipped_divide(mOutput, tiler_mn)
    mId = cute.make_identity_tensor(mInput.shape)
    gId = cute.zipped_divide(mId, tiler_mn)

    # scale tv layout
    # mode 0 - every 16 threads all write to the same location (only one writes)
    # mode 1 - scales are one element apart
    scale_tv_layout = cute.make_layout((16, 16), stride=(0, 1))   # (16,16):(0,1)
    # scale_tiler_mn is the number of scale slots written to by a CTA
    scale_tiler_mn = (cute.cosize(scale_tv_layout),)  # (16,)
    gScale = cute.zipped_divide(mScale, scale_tiler_mn)

    if cutlass.const_expr(_DEBUG):
        print('scale_tv_layout', scale_tv_layout)
        print('scale_tiler_mn', scale_tiler_mn)
        print('mScale', mScale)
        print('gScale', gScale)

    fp8_deepseek_1x128_kernel(
        gInput, gOutput, gScale, gId, input_tv_layout, scale_tv_layout, mInput.shape
    ).launch(
        grid=(cute.size(gInput, mode=[1]), 1, 1),
        block=(cute.size(input_tv_layout, mode=[0]), 1, 1),
    )

def fp8_deepseek_1x128(input: torch.Tensor, **kwargs):
    assert len(input.shape) == 2, "unsupported"
    assert input.shape[-1] % 128 == 0, "unsupported"
    assert input.is_contiguous(), "unsupported"
    M, N = input.shape
    input = input.view(-1)
    output = torch.empty(input.shape, dtype=torch.float8_e4m3fn, device=input.device) 
    scale = torch.empty(M * (N // 128), dtype=torch.float32, device=input.device)
    input_cute = from_dlpack(input, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)
    scale_cute = from_dlpack(scale, assumed_align=16)
    fp8_deepseek_1x128_jit(input_cute, output_cute, scale_cute)
    return output.view(M, N), scale.view(M, N // 128)


FP8_DEEPSEEK_1X128 = QuantCastCuteRecipe.from_gold(
    Deepseek1x128Gold, cute_fn=fp8_deepseek_1x128
)

@cute.kernel
def fp8_deepseek_1x128_dim_m_kernel(
    gInput: cute.Tensor,
    gOutput: cute.Tensor,
    gScale: cute.Tensor,
    Input_tv_layout: cute.Layout,
    Output_tv_layout: cute.Layout,
    Scale_tv_layout: cute.Layout,
    sScratch_layout: cute.typing.ComposedLayout,
    sScratchT_layout: cute.typing.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()  # thread index in block (0 to bdim-1)
    bidx, bidy, _ = cute.arch.block_idx()  # block index in grid (0 to grid_dim -1)
    # bdim, _, _ = cute.arch.block_dim()  # threads per block

    # slice for thread-level view
    thr_coord = (tidx, None)

    # create shared memory scratchpad
    # raw smem allocation: cosize() = footprint in elements (2048 here), 16 is alignment hint
    sScratch_ptr = cute.arch.alloc_smem(cutlass.BFloat16, cute.cosize(sScratch_layout), 16)
    sScratch = cute.make_tensor(sScratch_ptr, sScratch_layout)

    # select everything in the 2d tile, for tile index (bidx, bidy)
    blk_coord_Input = ((None, None), (bidx, bidy))
    # logical coord -> address
    blkInput = gInput[blk_coord_Input]
    tidfrgInput = cute.composition(blkInput, Input_tv_layout)
    thrInput = tidfrgInput[thr_coord]

    # write to scratchpad
    tidfrgScratch = cute.composition(sScratch, Input_tv_layout)
    thrScratch = tidfrgScratch[thr_coord]
    thrScratch[None] = thrInput.load()

    # sync this CTA's threads
    cute.arch.sync_threads()

    # end of phase 1, start of phase 2

    # transpose the scratch pad
    sScratchT = cute.make_tensor(sScratch_ptr, sScratchT_layout)

    # select this thread's scratchpad region
    tidfrgScratchT = cute.composition(sScratchT, Output_tv_layout)
    thrScratchT = tidfrgScratchT[thr_coord].load()

    # do the fp8 deepseek 1x128 calculation
    thrA_abs = cute.math.absf(thrScratchT)
    thrA_amax8 = thrA_abs.reduce(cute.ReductionOp.MAX, init_val=1e-12, reduction_profile=1)
    thrA_amax128 = cute.arch.warp_reduction_max(thrA_amax8, threads_in_group=16)
    thrA_amax128_fp32 = cutlass.Float32(thrA_amax128)
    scale = (thrA_amax128_fp32 * cutlass.Float32(1.0 / 448.0))
    qdata = (thrScratchT.to(cutlass.Float32) * (1.0 / scale)).to(cutlass.Float8E4M3FN)

    # select the write region, with transposed block indices
    blk_coord_Output = ((None, None), (bidy, bidx))

    blkOutput = gOutput[blk_coord_Output]
    tidfrgOutput = cute.composition(blkOutput, Output_tv_layout)
    thrOutput = tidfrgOutput[thr_coord]

    blkScale = gScale[blk_coord_Output]
    tidfrgScale = cute.composition(blkScale, Scale_tv_layout)

    # store qdata
    thrOutput[None] = qdata

    # every 16'th thread stores scale
    if tidx % 16 == 0:
        tidfrgScale[tidx] = scale

    if cutlass.const_expr(_DEBUG):
        pass


@cute.jit
def fp8_deepseek_1x128_dim_m_jit(mInput: cute.Tensor, mOutput: cute.Tensor, mScale: cute.Tensor):
    # 256 is a reasonable default
    num_threads_per_block = 256

    # NEW

    # * input is bf16, so 8 values per thread for a 128 bit load
    # * quant_block size is 128, 128 / 8 = 16, so we need 16 threads to cover a quant_block

    # input tv-layout
    # first element must be at least 128 in the line below to cover quant block size 128 along dim-m
    thrInput_layout = cute.make_ordered_layout((128, num_threads_per_block // 128), order=(1, 0))
    valInput_layout = cute.make_ordered_layout((1, 8), order=(1, 0))
    tilerInput_mn, Input_tv_layout = cute.make_layout_tv(thrInput_layout, valInput_layout)
    gInput = cute.zipped_divide(mInput, tilerInput_mn)

    # layout of the shared memory scratchpad.
    # Phase 2 reads this scratchpad transposed (down columns) -> 16-way bank conflicts that pin the
    # L1/TEX pipe at ~93% while DRAM sits at ~34% (ncu). We swizzle the physical buffer to scatter
    # those column reads across banks. The SAME swizzle wraps both the write-view (sScratch) and the
    # transposed read-view (sScratchT): a swizzle is a bijection on the offset, and both views compute
    # the same base offset for a given logical element, so the transpose aliasing is preserved
    # bit-exactly. Swizzle<3,3,5> protects the low 3 bits (the 128-bit vector store stays intact) and
    # XORs the colliding row bits (at stride 128 = bit 7) into the bank bits. See transpose_v1.
    smem_swizzle = cute.make_swizzle(3, 3, 5)
    sScratch_layout = cute.make_composed_layout(
        smem_swizzle, 0, cute.make_ordered_layout(tilerInput_mn, order=(1, 0))
    )
    sScratchT_layout = cute.make_composed_layout(
        smem_swizzle, 0, cute.make_ordered_layout((tilerInput_mn[1], tilerInput_mn[0]), order=(0, 1))
    )

    # now, thinking through the read-write of stage 2
    # scratchad shape: (128,16):(16,1) = (128, 16) = (TM, TN)
    # scratchT shape: (16,128)
    #   each thread handles 8 values
    #   thr (16,16):(16,1), val (1,8):(0,1)
    thrOutput_layout = cute.make_ordered_layout((16, num_threads_per_block // 16), order=(1, 0))
    valOutput_layout = cute.make_ordered_layout((1, 8), order=(1, 0))
    tilerOutput_mn, Output_tv_layout = cute.make_layout_tv(thrOutput_layout, valOutput_layout)
    gOutput = cute.zipped_divide(mOutput, tilerOutput_mn)

    # scale tv layout
    # mode 0 - every 16 threads all write to the same location (only one writes)
    # mode 1 - scales are one element apart
    Scale_tv_layout = cute.make_layout((16, 16), stride=(0, 1))   # (16,16):(0,1)
    # scale_tiler_mn is the number of scale slots written to by a CTA
    Scale_tiler_mn = (cute.cosize(Scale_tv_layout), 1)  # (16,1)
    gScale = cute.zipped_divide(mScale, Scale_tiler_mn)

    fp8_deepseek_1x128_dim_m_kernel(
        gInput, gOutput, gScale, Input_tv_layout, Output_tv_layout, Scale_tv_layout,
        sScratch_layout, sScratchT_layout
    ).launch(
        # grid - number of thread blocks (CTAs) per launch
        grid=(cute.size(gInput, mode=[1, 0]), cute.size(gInput, mode=[1, 1]), 1),
        # block - threads per block instance
        block=(cute.size(Input_tv_layout, mode=[0]), 1, 1),
    )

    # TODO implement the rest


    if cutlass.const_expr(_DEBUG):
        print('mInput', mInput)
        print('mOutput', mOutput)
        print('thrInput_layout', thrInput_layout)
        print('valInput_layout', valInput_layout)
        print('tilerInput_mn', tilerInput_mn)
        print('Input_tv_layout', Input_tv_layout)
        print('gInput', gInput)
        print('sScratch_layout', sScratch_layout)
        print('sScratchT_layout', sScratchT_layout)
        print('thrOutput_layout', thrOutput_layout)
        print('valOutput_layout', valOutput_layout)
        print('tilerOutput_mn', tilerOutput_mn)
        print('Output_tv_layout', Output_tv_layout)
        print('gOutput', gOutput)
        print('mScale', mScale)
        print('Scale_tv_layout', Scale_tv_layout)
        print('Scale_tiler_mn', Scale_tiler_mn)
        print('gScale', gScale)


def fp8_deepseek_1x128_dim_m(input: torch.Tensor, **kwargs):
    assert len(input.shape) == 2, "unsupported"
    assert input.shape[0] % 128 == 0, "unsupported"
    assert input.shape[1] % 16 == 0, "unsupported"
    assert input.is_contiguous(), "unsupported"
    M, N = input.shape
    output = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=input.device) 
    scale = torch.empty(N, M // 128, dtype=torch.float32, device=input.device)
    input_cute = from_dlpack(input, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)
    scale_cute = from_dlpack(scale, assumed_align=16)
    fp8_deepseek_1x128_dim_m_jit(input_cute, output_cute, scale_cute)
    return output, scale

FP8_DEEPSEEK_1X128_DIM_M = QuantCastCuteRecipe.from_gold(
    Deepseek1x128DimMGold, cute_fn=fp8_deepseek_1x128_dim_m
)

# ---------------------------------------------------------------------------
# v2: direct copy-paste of quant_cast_cute/recipes.py::fp8_deepseek_1x128_dim_m_v2 and its
# implementation, self-contained here (no code reused from the original module). We will make it more
# readable later; for now it is a faithful copy so we can iterate on it in this playground.
#
# deepseek fp8 1x128 dim-M: reduce 128-row blocks down M, transposed outputs (N, M) / (N, M//128).
# The direct analog of `mxfp8_dim_m` (warp-specialized TMA), with a 128-row block (not 32) and
# an fp32 amax/448 scale (not an e8m0 byte). Feeding a transposed x.t() to the scalar kernel makes
# every load uncoalesced (~7% peak); instead:
#   - TMA G2S loads a (TM, TN) row-major tile of x into smem (sInput), TM a multiple of 128;
#   - the (TM/128 x TN) scale-groups are split across threads; each owns one (128-row block, col),
#     scans its 128 rows down a column of sInput for the amax (scalar accumulate -> low registers,
#     keeps occupancy high), computes scale = max(amax,1e-12)/448, then re-reads the column to
#     quantize and write the 128 fp8 values as a CONTIGUOUS run into sOutput laid out (TN, TM) -- the
#     transpose happens in the register->smem write (no col-major TMA, which the DSL can't drive);
#   - TMA S2G stores sOutput into the (TN, TM) tile of the row-major (N, M) output at (n_tile, m_tile).
# The fp32 scale is scattered straight to gmem scales (N, M//128). Barrier follows the same
# arrive-and-expect-tx pattern (single arrival, warp-0 gated) required for a multi-warp block.
# ---------------------------------------------------------------------------
_DSM_TM, _DSM_TN, _DSM_WARPS = 128, 128, 4         # tuned on B200 @ 16384 (needs M%TM==0, N%TN==0)
_DSM_THREADS = _DSM_WARPS * 32  # 128
_DSM_RB = _DSM_TM // 128                            # 1 128-row block per tile
_DSM_CHUNKS = 128 // 32                              # 4 32-wide chunks per 128-row block (vectorize)
_DSM_GROUPS = _DSM_TN * _DSM_RB                     # 128 (col, row-block) scale groups
_DSM_ITERS = (_DSM_GROUPS + _DSM_THREADS - 1) // _DSM_THREADS  # 1 iter
_DSM_IN_BYTES = _DSM_TM * _DSM_TN * 2               # 128*128*2 bf16 tile bytes for the TMA expect-tx


@cute.struct
class _DeepseekDimMSmem:
    tma_bar: cute.struct.MemRange[cutlass.Int64, 1]
    sInput: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, _DSM_TM * _DSM_TN], 1024]
    sOutput: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, _DSM_TM * _DSM_TN], 1024]


@cute.kernel
def fp8_deepseek_1x128_dim_m_v2_kernel(
    input_tma_atom: cute.CopyAtom,
    input_tma_tensor: cute.Tensor,
    output_tma_atom: cute.CopyAtom,
    output_tma_tensor: cute.Tensor,
    mScale: cute.Tensor,
    input_smem_layout: cute.Layout,
    output_smem_layout: cute.Layout,
    M: cutlass.Int64,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()   # bidx = m_tile, bidy = n_tile

    # compiler hint that warp_idx does not change across a warp
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # allocator over this CTA's smem region
    smem = utils.SmemAllocator()
    # allocate the struct in smem
    st = smem.allocate(_DeepseekDimMSmem)
    # get the barrier pointer
    tma_bar_ptr = st.tma_bar.data_ptr()
    if tidx == 0:
        # initializes the mbarrier so that it takes 1 arrival to complete the current phase
        # that arrival comes from mbarrier_arrive_and_expect_tx below; the TMA copy separately
        # signals completion via the expected transaction-byte count (_DSM_IN_BYTES)
        cute.arch.mbarrier_init(tma_bar_ptr, 1)

    # publish the mbarrier write above before any mbarrier op
    cute.arch.mbarrier_init_fence()
    # sync threads
    cute.arch.sync_threads()

    # TMA smem staging buffers (TMA will copy into sInput, and later out of sOutput)
    sInput = st.sInput.get_tensor(input_smem_layout)            # (TM, TN) row-major
    sOutput = st.sOutput.get_tensor(output_smem_layout)         # (TN, TM) row-major (transposed)
    # TMA gmem tile views, for now for entire tensor (not yet for current block)
    gInput = cute.local_tile(input_tma_tensor, (_DSM_TM, _DSM_TN), (None, None))
    gOutput = cute.local_tile(output_tma_tensor, (_DSM_TN, _DSM_TM), (None, None))

    # produce the final partitioned views that cute.copy actually consumes — 
    # one pair for the input (G2S) copy, one for the output (S2G) copy.
    # Note: tma_partition returns (smem-side, gmem-side) views; the tXsX/tXgX form is t + copy-op + mem +
    #   operand (so the operand word appears twice, mirroring the tAsA/tAgA convention).
    tInputsInput, tInputgInput = cpasync.tma_partition(
        input_tma_atom,  # TMA descriptor
        0,   # cta coord
        cute.make_layout(1),  # 1 CTA per transfer
        cute.group_modes(sInput, 0, 2),  # smem tensor, modes 0..2 grouped into one
        cute.group_modes(gInput, 0, 2),  # gmem tensor, modes 0..2 grouped into one
    )
    tOutputsOutput, tOutputgOutput = cpasync.tma_partition(
        output_tma_atom, 0, cute.make_layout(1),
        cute.group_modes(sOutput, 0, 2), cute.group_modes(gOutput, 0, 2))

    # one warp inits the copy
    if warp == 0:
        # one thread arms the barrier for the copy
        with cute.arch.elect_one():
            # arrive: contributes the 1 expected arrival that mbarrier_init(tma_bar_ptr, 1) is waiting for
            # expect_tx: expect _DSM_IN_BYTES bytes to be delivered before this phase completes.
            # Note: TMA will decrement remaining bytes as data arrives
            cute.arch.mbarrier_arrive_and_expect_tx(tma_bar_ptr, _DSM_IN_BYTES)
        # Fires the actual G2S bulk copy
        # Note: internally the "only use one elected thread" info is in the TMA descriptor, so no-op for any non-elected thread
        cute.copy(input_tma_atom, tInputgInput[(None, bidx, bidy)], tInputsInput, tma_bar_ptr=tma_bar_ptr)

    # All block here until the barrier's phase flips.
    # After this line returns, sInput is guaranteed fully populated
    cute.arch.mbarrier_wait(tma_bar_ptr, 0)

    # these are for manual indexing into scale, would go away
    # if scale was using a proper layout
    m0 = bidx * _DSM_TM  # bidx * 128
    n0 = bidy * _DSM_TN  # bidy * 128
    mblk = M // 128

    # this loops over thread-groups in columns of a 128x128 tile. Since
    # right now we are using 128 threads per CTA, number of iterations is 1.
    # If we had 64 threads / CTA, it would be 2, etc.
    for it in cutlass.range_constexpr(_DSM_ITERS):

        g = tidx + it * _DSM_THREADS
        if g < _DSM_GROUPS:
            # this indexing should be rewritten with layouts
            col = g % _DSM_TN
            rb = g // _DSM_TN
            r0 = rb * 128

            # pass 1: amax over the 128 rows down this column, in 4 chunks of 32 (vector reduce,
            # only 32 f32 live at a time -> low registers, high occupancy).
            amax = cutlass.Float32(0.0)
            # loops 4 times over 128 rows in a column, processing 32 elements each time
            # reason: keep register usage low
            for c in cutlass.range_constexpr(_DSM_CHUNKS):
                rInput = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                for r in cutlass.range_constexpr(32):
                    rInput[r] = sInput[r0 + c * 32 + r, col].to(cutlass.Float32)
                v = rInput.load()
                amax = cutlass.max(amax, cute.where(v < 0, -v, v).reduce(
                    cute.ReductionOp.MAX, cutlass.Float32(0.0), 0))
            scale = cutlass.max(amax, cutlass.Float32(1e-12)) * (1.0 / 448.0)  # recip-mul: gold's
            inv = 1.0 / scale                        # `/ 448.0` (tensor/py-scalar) lowers to *(1/448)
            # pass 2: re-read (cheap smem), quantize each chunk (vectorized f32->fp8), transpose in
            # the register->smem write (contiguous run into sOutput).
            for c in cutlass.range_constexpr(_DSM_CHUNKS):
                rInput = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
                for r in cutlass.range_constexpr(32):
                    rInput[r] = sInput[r0 + c * 32 + r, col].to(cutlass.Float32)
                rOutput = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
                rOutput.store((rInput.load() * inv).to(cutlass.Float8E4M3FN))
                for r in cutlass.range_constexpr(32):
                    sOutput[col, r0 + c * 32 + r] = rOutput[r]
            # this should be a layout write
            mScale[(n0 + col) * mblk + (m0 // 128 + rb)] = scale.to(mScale.element_type)

    # make the register->smem writes to sOutput visible to the async (TMA) proxy...
    cute.arch.fence_proxy("async.shared", space="cta")
    # ...and wait for all threads to finish writing sOutput before the store-TMA reads it
    cute.arch.sync_threads()
    # kick off TMA qdata write
    if warp == 0:
        cute.copy(output_tma_atom, tOutputsOutput, tOutputgOutput[(None, bidy, bidx)])  # y tile (n_tile, m_tile)


@cute.jit
def fp8_deepseek_1x128_dim_m_v2_jit(mInput, mOutput, mScale, M: cutlass.Constexpr):
    # _DSM_TM, DSM_TN = 128, 128

    # (128,128):(128,1)
    input_smem_layout = cute.make_layout((_DSM_TM, _DSM_TN), stride=(_DSM_TN, 1))
    # (128,128):(128,1)
    output_smem_layout = cute.make_layout((_DSM_TN, _DSM_TM), stride=(_DSM_TM, 1))

    # input_tma_atom Copy Atom
    # ThrID:         1:0                 - a single thread participates, shape 1 stride 0
    # TV Layout Src: (1,16384):(0,1)     - 1 single thread owns all the 128*128 src elements
    # TV Layout Dst: (1,16384):(0,1)     - 1 single thread owns all the 128*128 dst elements
    # Value type:    bf16
    # input_tma_tensor tensor<(0,0) o (?,?{div=16}):(1@1,1@0)>
    #   <engine o layout>
    #   (0,0) — the "engine" (iterator) is a coordinate (0,0), not a raw pointer. 
    #     TMA addresses data by coordinates into the descriptor, so this tensor 
    #     carries a base coord instead of a memory address. When you later do
    #     local_tile(input_tma_tensor, (128,128), (bidx, bidy)), you're indexing 
    #     in that coordinate space.
    #   (?,?{div=16}) — the shape: two dynamic modes (M and N), both ? (runtime values, 
    #     because we called .mark_layout_dynamic(...)). The {div=16} on the 
    #     second mode (N) is the divisibility guarantee from
    #     .mark_compact_shape_dynamic(mode=1, divisibility=16) — 
    #     it promises N is a multiple of 16, which TMA needs for alignment.
    #   :(1@1,1@0) — the strides in scaled-basis notation. 
    #     k@j means "stride k along coordinate axis j". 
    #     So mode 0 (M) → 1@1 (advances descriptor axis 1), 
    #     mode 1 (N) → 1@0 (advances descriptor axis 0). The axis indices are
    #     swapped relative to the logical (M, N) listing because we marked 
    #     leading_dim=1: N is the contiguous dimension, and a TMA descriptor 
    #     puts the contiguous dim on axis 0. So N↔axis0, M↔axis1 — exactly what the basis strides
    #     encode. It's the coordinate-space way of saying "row-major with N innermost."
    # atom - the transfer op description
    # tma_tensor - the data being addressed
    input_tma_atom, input_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),  # op kind: bulk-tensor Global→Shared
        mInput,  
        input_smem_layout,  # (128,128):(128,1) — the smem tile it lands in
        (_DSM_TM, _DSM_TN),  # (128, 128) — the box/tile shape TMA moves
    )
    # output_tma_atom Copy Atom
    # ThrID:         1:0
    # TV Layout Src: (1,16384):(0,1)
    # TV Layout Dst: (1,16384):(0,1)
    # Value type:    f8E4M3FN
    # output_tma_tensor tensor<(0,0) o (?,?{div=16}):(1@1,1@0)>
    output_tma_atom, output_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), 
        mOutput, 
        output_smem_layout, 
        (_DSM_TN, _DSM_TM),
    )
    M2, N2 = mInput.shape

    fp8_deepseek_1x128_dim_m_v2_kernel(
        input_tma_atom, input_tma_tensor, output_tma_atom, output_tma_tensor, mScale,
        input_smem_layout, output_smem_layout, cutlass.Int64(M),
    ).launch(
        # (ceil_div(M, 128), ceil_div(N, 128), 1)
        grid=(_ceil_div(M2, _DSM_TM), _ceil_div(N2, _DSM_TN), 1),
        # (4*32, 1, 1) = (128, 1, 1)
        block=(_DSM_THREADS, 1, 1),
    )

    if cutlass.const_expr(_DEBUG):
        print('input_smem_layout', input_smem_layout)
        print('output_smem_layout', output_smem_layout)
        print('input_tma_atom', input_tma_atom)
        print('input_tma_tensor', input_tma_tensor)
        print('output_tma_atom', output_tma_atom)
        print('output_tma_tensor', output_tma_tensor)


def fp8_deepseek_1x128_dim_m_v2(input, **kwargs):
    assert input.is_contiguous() and input.dim() == 2
    M, N = input.shape
    assert M % _DSM_TM == 0 and N % _DSM_TN == 0, \
        f"deepseek_1x128_dim_m cute kernel needs M%{_DSM_TM}==0 and N%{_DSM_TN}==0"
    output = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=input.device)  # transposed row-major output
    scale = torch.empty(N, M // 128, dtype=torch.float32, device=input.device)
    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-elem aligned).
    mInput = (from_dlpack(input, assumed_align=16).mark_layout_dynamic(leading_dim=1)
              .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mOutput = (from_dlpack(output, assumed_align=16).mark_layout_dynamic(leading_dim=1)
               .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mScale = from_dlpack(scale.reshape(-1)).mark_layout_dynamic()
    fn = _compiled(("deepseek_1x128_dim_m_v2", M, N), fp8_deepseek_1x128_dim_m_v2_jit, mInput, mOutput, mScale, M)
    fn(mInput, mOutput, mScale)
    return output, scale

FP8_DEEPSEEK_1X128_DIM_M_V2 = QuantCastCuteRecipe.from_gold(
    Deepseek1x128DimMGold, cute_fn=fp8_deepseek_1x128_dim_m_v2
)


# ---------------------------------------------------------------------------
# mxfp8_swizzle: FP8_DEEPSEEK_1X128 with mxfp8 numerics. Same launch grid (256 threads x 8 vals =
# a 2048-element tile over the flattened input), so the only real changes are the block size
# (1x32 instead of 1x128 -> 4 threads cooperate instead of 16), the scale format (e8m0 RCEIL byte
# instead of an fp32 amax/448), and the scale store (a swizzled scatter instead of a contiguous
# TV-layout write).
# ---------------------------------------------------------------------------


# swizzle offset (device): pre-swizzle scale position (row, col) -> flat offset into the 4D
# (nrb, ncb, 32, 16) block grid, i.e. ((row//128 * ncb + col//4) * 32 + row%128%32) * 16
# + ((row%128)//32 * 4 + col%4). Copied from quant_cast_cute/recipes.py `_swizzle_flat`; an exact
# port of the gold `_to_blocked_4d` index math. `col` is the 32-group (block-column) index.
@cute.jit
def _swizzle_flat(row, col, ncb: cutlass.Int32):
    br = row // 128
    r128 = row % 128
    a = r128 // 32
    b = r128 % 32
    bc = col // 4
    c4 = col % 4
    return ((br * ncb + bc) * 32 + b) * 16 + (a * 4 + c4)


@cute.kernel
def mxfp8_swizzle_kernel(
    gInput: cute.Tensor,
    gOutput: cute.Tensor,
    mScale: cute.Tensor,
    gId: cute.Tensor,
    input_tv_layout: cute.Layout,
    orig_shape: cute.Shape,
    gpr: cutlass.Constexpr,  # 32-element groups per row == N // 32
    ncb: cutlass.Constexpr,  # swizzle column-blocks == (N // 32) // 4
):
    tidx, _, _ = cute.arch.thread_idx()  # thread index in block (0 to bdim-1)
    bidx, _, _ = cute.arch.block_idx()  # block index in grid (0 to grid_dim -1)

    # slice for thread-block level view
    # note: "select everything in the 1d tile, for tile index bidx"
    blk_coord = ((None,), bidx)

    # logical coord -> address
    blkInput = gInput[blk_coord]
    blkOutput = gOutput[blk_coord]
    blkId = gId[blk_coord]

    # compose for thread-index & value-index to physical mapping
    tidfrgInput = cute.composition(blkInput, input_tv_layout)
    tidfrgOutput = cute.composition(blkOutput, input_tv_layout)
    tidfrgId = cute.composition(blkId, input_tv_layout)

    # slice for thread-level view
    thr_coord = (tidx, None)

    # mask out of bounds
    thrId = tidfrgId[thr_coord]
    if cute.elem_less(thrId[0], orig_shape):

        # reference:
        #
        # def mxfp8_f(x, **kwargs):
        #     *lead, last = x.shape
        #     x_b = x.reshape(*lead, last // 32, 32)
        #     amax = x_b.abs().amax(dim=-1, keepdim=True)   # NOTE: no clamp, unlike deepseek
        #     scale_e8m0 = _amax_to_e8m0_rceil(amax)
        #     qdata = (x_b.to(torch.float32)
        #              * _e8m0_scale_to_reciprocal_fp32(scale_e8m0)).to(torch.float8_e4m3fn)
        #     return qdata.reshape(*lead, last), scale_e8m0.squeeze(-1)
        # def mxfp8_swizzle_f(x, **kwargs):
        #     qdata, scale_e8m0 = mxfp8_f(x)
        #     return qdata, _to_blocked_4d(scale_e8m0)
        #
        # Each thread owns 8 elements, 32 // 8 = 4, so every 4 threads
        # calculate a 1x32 block

        # Load the fragment, slice for threads: vid -> address
        thrInput = tidfrgInput[thr_coord].load()

        # Cast to fp32 here, for 2 reasons:
        # 1. cute.math.absf is fp32 
        # 2. warp reduction uses fp32 registers
        thrInput = thrInput.to(cutlass.Float32)

        thrA_abs = cute.math.absf(thrInput)
        # thread-local max. init_val is 0.0, NOT deepseek's 1e-12: mxfp8 does not clamp amax, and
        # an all-zero block must produce biased == 0 (1e-12 would give descale 2.2e-15 -> ~78).
        thrA_amax8 = thrA_abs.reduce(cute.ReductionOp.MAX, init_val=0.0, reduction_profile=1)
        # intra-thread max. 4, not 16: warp_reduction_max is a butterfly shuffle clamped to the
        # warp, so it reduces within ALIGNED 4-lane groups -- and val_layout (8):(1) gives each
        # thread a contiguous 8-run, so lanes 4k..4k+3 own exactly one 1x32 block.
        thrA_amax32 = cute.arch.warp_reduction_max(thrA_amax8, threads_in_group=4)
        # convert bf16 -> fp32
        # thrA_amax32_fp32 = cutlass.Float32(thrA_amax32)
        thrA_amax32_fp32 = thrA_amax32
        # calculate the e8m0 scale byte and its fp32 reciprocal pow2 factor
        rcp, biased = _e8m0(thrA_amax32_fp32)
        # calculate qdata -- multiply by the reciprocal, matching torchao's _to_mx_rceil
        qdata = (thrInput.to(cutlass.Float32) * rcp).to(cutlass.Float8E4M3FN)

        if cutlass.const_expr(_DEBUG):
            print('thrA_amax8 static type', thrA_amax8.type)
            print('thrA_amax32_fp32 static type', thrA_amax32_fp32.dtype)
            print('rcp static type', rcp.dtype)
            if tidx == 0 and bidx == 0:
                cute.printf(
                    "thrInput {}, {}, {}, {}, {}, {}, {}, {}",
                    thrInput[0], thrInput[1], thrInput[2], thrInput[3],
                    thrInput[4], thrInput[5], thrInput[6], thrInput[7],
                )
                cute.printf("thrA_amax8 {}", thrA_amax8)
                cute.printf("thrA_amax32 {}", thrA_amax32)
                cute.printf("biased {}, rcp {}", biased, rcp)
                cute.printf(
                    "qdata {}, {}, {}, {}, {}, {}, {}, {}",
                    qdata[0], qdata[1], qdata[2], qdata[3],
                    qdata[4], qdata[5], qdata[6], qdata[7],
                )

        # store qdata
        thrOutput = tidfrgOutput[thr_coord]
        thrOutput[None] = qdata
        # every 4'th thread stores scale. Unlike deepseek we cannot use a scale TV layout +
        # zipped_divide: that works only because deepseek's scale is dense row-major in the same
        # order as the input tile, whereas the swizzled destination is scattered. So compute the
        # global 1x32 block index directly and scatter. (Deriving it from bidx/tidx rather than
        # thrId[0] keeps it a plain integer -- a rank-1 identity tensor yields a coordinate.)
        if tidx % 4 == 0:
            tile = cute.size(input_tv_layout)  # 2048 flat elements per CTA (static)
            vpt = cute.size(input_tv_layout, mode=[1])  # 8 elements per thread (static)
            gblk = (bidx * tile + tidx * vpt) // 32  # global 1x32 block index
            row, col = gblk // gpr, gblk % gpr  # pre-swizzle scale coords
            mScale[_swizzle_flat(row, col, ncb)] = biased.to(mScale.element_type)


@cute.jit
def mxfp8_swizzle_jit(
    mInput: cute.Tensor,
    mOutput: cute.Tensor,
    mScale: cute.Tensor,
    gpr: cutlass.Constexpr,
    ncb: cutlass.Constexpr,
):
    # 256 is a reasonable default
    num_threads_per_block = 256

    # build the tv layout
    # thr: (256):(1)
    thr_layout = cute.make_ordered_layout((num_threads_per_block,), order=(0,))
    # val: (8):(1)
    val_layout = cute.make_ordered_layout((8,), order=(0,))
    # (2048,)  (256,8):(8,1)
    tiler_mn, input_tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    # ((TileM,), (RestM,))
    gInput = cute.zipped_divide(mInput, tiler_mn)
    gOutput = cute.zipped_divide(mOutput, tiler_mn)
    mId = cute.make_identity_tensor(mInput.shape)
    gId = cute.zipped_divide(mId, tiler_mn)

    # no scale tv layout here -- the swizzled scale is scattered, see the kernel's scale store.

    if cutlass.const_expr(_DEBUG):
        print('mInput', mInput)
        print('gInput', gInput)
        print('mOutput', mOutput)
        print('gpr', gpr)
        print('ncb', ncb)
        print('mScale', mScale)

    mxfp8_swizzle_kernel(
        gInput, gOutput, mScale, gId, input_tv_layout, mInput.shape, gpr, ncb
    ).launch(
        grid=(cute.size(gInput, mode=[1]), 1, 1),
        block=(cute.size(input_tv_layout, mode=[0]), 1, 1),
    )

def mxfp8_swizzle(input: torch.Tensor, **kwargs):
    assert len(input.shape) == 2, "unsupported"
    assert input.shape[-1] % 32 == 0, "unsupported"
    assert input.is_contiguous(), "unsupported"
    M, N = input.shape
    ngc = N // 32  # 32-element groups per row
    # Whole 128x4 swizzle atoms only, so every slot of the scale grid is written and there is no
    # padding to reason about. This also implies N % 128 == 0, hence (M*N) % 2048 == 0, so every
    # 2048-element tile is fully in bounds -- which matters because warp_reduction_max below uses
    # a full-warp mask inside the out-of-bounds guard.
    assert M % 128 == 0 and ngc % 4 == 0, "unsupported"
    nrb, ncb = M // 128, ngc // 4
    input = input.view(-1)
    output = torch.empty(input.shape, dtype=torch.float8_e4m3fn, device=input.device)
    scale = torch.zeros(nrb * ncb * 32 * 16, dtype=torch.uint8, device=input.device)
    input_cute = from_dlpack(input, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)
    scale_cute = from_dlpack(scale)
    mxfp8_swizzle_jit(input_cute, output_cute, scale_cute, ngc, ncb)
    return output.view(M, N), scale.view(nrb, ncb, 32, 16).view(torch.float8_e8m0fnu)


MXFP8_SWIZZLE = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleGold, cute_fn=mxfp8_swizzle
)


# ---------------------------------------------------------------------------
# mxfp8_swizzle_v3: 2-D 32x128 tile + 16-elem/thread aligned WORD load. This started as v1's
# numerics on a 2-D tile with an 8-elem LDG.128 (isolating the tiling change; that regressed to
# 55.6% -- see the memory note), then adopted the other perf lever: 16 elems/thread via an explicit
# uint32-word load path.
#
# Why the word load and not a plain 16-wide fragment .load(): at 16 bf16 (=32B=2xLDG.128) the TV
# fragment .load() FAILS to vectorize -- it serializes to 1 sector/element (268M vs 16.8M sectors,
# L1TEX pipe saturates). The word load sidesteps this by viewing the input as uint32 (mWords),
# issuing 128-bit CopyUniversal atoms into 4 u32 registers, and unpacking bf16x2 -> f32x2 in
# registers. The host must mark the row stride divisible by 4 words so the compiler proves 128-bit
# alignment.
#
# Tile = 32 rows x 128 cols. At 16 elems/thread that is 256 threads/CTA. A tile row of 128 cols is
# 8 threads = 4 blocks (1x32) x 2 lanes. So warp_reduction_max(threads_in_group=2) -- ONE shuffle
# stage (vs v1's 4-lane / two-stage), the other saving. Every 2nd lane stores the scale.
#
# The qdata STORE stays v3's TV-composition store (thrOutput[None] = qdata): 16 fp8 = 16B = one
# STG.128, mapped to contiguous columns, so it coalesces without a packed-uint32 store path.
#
# Grid is (N/128, M/32) -- a CTA's tile origin is a plain 2-D coordinate, so the scale scatter
# computes (row, col) directly.
_MXS3_TM, _MXS3_TN = 32, 128          # 2-D tile (rows x cols); needs M%32==0, N%128==0
_MXS3_VPT = 16                         # elements per thread (2xLDG.128 for bf16, via word load)
_MXS3_TPR = _MXS3_TN // _MXS3_VPT      # 8 threads per tile row
_MXS3_THREADS = _MXS3_TM * _MXS3_TPR   # 256 threads/CTA
_MXS3_LANES = 32 // _MXS3_VPT          # 2 lanes per 1x32 block


@dsl_user_op
def _bf16x2_to_f32x2(word, *, loc=None, ip=None):
    # Unpack a uint32 holding two bf16 lanes into (lo_f32, hi_f32). bf16 -> f32 is lossless: shift
    # each bf16 into the high 16 bits of a 32-bit word and reinterpret as fp32.
    lo_bits = cutlass.Int32((word & cutlass.Uint32(0xFFFF)) << cutlass.Uint32(16))
    hi_bits = cutlass.Int32(word & cutlass.Uint32(0xFFFF0000))
    lo = cutlass.Float32(llvm.bitcast(T.f32(), lo_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    hi = cutlass.Float32(llvm.bitcast(T.f32(), hi_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return lo, hi


@cute.jit
def _load_bf16x16_as_f32(mWords: cute.Tensor, row, col_start):
    # Load 16 contiguous bf16 from row `row`, starting at column `col_start`, as 16 f32 -- via two
    # 128-bit uint32 copy atoms (8 bf16 each) + register unpack. `mWords` is the input reinterpreted
    # as uint32 with a divisibility-4 row stride, so 128-bit copies are provably aligned.
    values = cute.make_rmem_tensor(16, cutlass.Float32)
    src_words = cute.tiled_divide(mWords[row, None], (4,))  # ((4,), n_word_groups)
    dst = cute.make_rmem_tensor(4, cutlass.Uint32)
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=128
    )
    # 16 bf16 = 8 uint32 words = two 4-word (128-bit) copies. word column = col_start // 2.
    word_col0 = col_start // 2
    for grp in cutlass.range_constexpr(2):
        cute.copy(copy_atom, src_words[None, (word_col0 + grp * 4) // 4], dst)
        for w in cutlass.range_constexpr(4):
            lo, hi = _bf16x2_to_f32x2(dst[w])
            values[grp * 8 + w * 2 + 0] = lo
            values[grp * 8 + w * 2 + 1] = hi
    return values


@cute.kernel
def mxfp8_swizzle_v3_kernel(
    mWords: cute.Tensor,   # 2-D (M, N//2) uint32 view of the bf16 input (div-4 row stride)
    gOutput: cute.Tensor,
    mScale: cute.Tensor,
    input_tv_layout: cute.Layout,
    ncb: cutlass.Constexpr,  # swizzle column-blocks == (N // 32) // 4
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()   # bidx = n-tile (cols), bidy = m-tile (rows)

    # this thread's (row, col_start) in global coords -- 16 contiguous cols starting at col_start.
    # M%128==0 and N%128==0 => every 32x128 tile is fully in bounds, so no per-thread guard is needed
    # (the warp_reduction_max below also assumes a full aligned group, like v1).
    row = bidy * _MXS3_TM + tidx // _MXS3_TPR
    col_start = bidx * _MXS3_TN + (tidx % _MXS3_TPR) * _MXS3_VPT

    # 16-elem aligned word load (2x LDG.128), unpacked to f32 in registers.
    rInput = _load_bf16x16_as_f32(mWords, row, col_start)
    thrInput = rInput.load()

    thrA_abs = cute.math.absf(thrInput)
    # no clamp (mxfp8), init_val 0.0 so an all-zero block -> biased 0.
    thrA_amax16 = thrA_abs.reduce(cute.ReductionOp.MAX, init_val=0.0, reduction_profile=1)
    # 2 lanes per 1x32 block -> ONE shuffle stage over aligned 2-lane groups.
    thrA_amax32 = cute.arch.warp_reduction_max(thrA_amax16, threads_in_group=_MXS3_LANES)
    rcp, biased = _e8m0(thrA_amax32)
    qdata = (thrInput * rcp).to(cutlass.Float8E4M3FN)

    # store qdata via the TV composition (16 fp8 = one STG.128, contiguous columns).
    blkOutput = gOutput[((None, None), (bidy, bidx))]
    tidfrgOutput = cute.composition(blkOutput, input_tv_layout)
    tidfrgOutput[(tidx, None)] = qdata

    # every 2nd lane owns the leading 16 elems of a 1x32 block -> writes its scale.
    if tidx % _MXS3_LANES == 0:
        col = col_start // 32   # global 1x32 block col
        mScale[_swizzle_flat(row, col, ncb)] = biased.to(mScale.element_type)


@cute.jit
def mxfp8_swizzle_v3_jit(
    mWords: cute.Tensor,   # 2-D (M, N//2) uint32
    mOutput: cute.Tensor,  # 2-D (M, N) fp8
    mScale: cute.Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ncb: cutlass.Constexpr,
):
    tiler_mn = (_MXS3_TM, _MXS3_TN)

    # Output TV layout: domain ((thr), (val)) -> COORDINATE into the (32,128) tile. Composition
    # decodes it column-major (row-fastest), then the tile's row-major strides map to memory. To walk
    # contiguous COLUMNS the val stride is TM (=32) -> memory stride 1 (the STG.128 axis).
    #   val v      -> column +v        -> stride TM      (=32; contiguous 16-run -> STG.128)
    #   t_lo=t%8   -> column-run +VPT  -> stride TM*VPT  (=512)
    #   t_hi=t//8  -> row +1           -> stride 1
    input_tv_layout = cute.make_layout(
        ((_MXS3_TPR, _MXS3_TM), (_MXS3_VPT,)),
        stride=((_MXS3_TM * _MXS3_VPT, 1), (_MXS3_TM,)),
    )

    gOutput = cute.zipped_divide(mOutput, tiler_mn)

    if cutlass.const_expr(_DEBUG):
        print('mWords', mWords)
        print('mOutput', mOutput)
        print('input_tv_layout', input_tv_layout)

    mxfp8_swizzle_v3_kernel(
        mWords, gOutput, mScale, input_tv_layout, ncb
    ).launch(
        grid=(N // _MXS3_TN, M // _MXS3_TM, 1),
        block=(_MXS3_THREADS, 1, 1),
    )


def mxfp8_swizzle_v3(input: torch.Tensor, **kwargs):
    assert len(input.shape) == 2, "unsupported"
    assert input.is_contiguous(), "unsupported"
    assert input.dtype == torch.bfloat16, "v3 word load is bf16-only"
    M, N = input.shape
    ngc = N // 32
    # 32x128 tile + whole 128x4 swizzle atoms: M%32==0 and N%128==0 (=> ngc%4==0). Keep M%128==0 to
    # match v1/gold's dense scale layout. N%128==0 => the uint32 row stride (N//2) is a multiple of 4.
    assert M % 128 == 0 and N % _MXS3_TN == 0, "unsupported"
    nrb, ncb = M // 128, ngc // 4
    output = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=input.device)
    scale = torch.zeros(nrb * ncb * 32 * 16, dtype=torch.uint8, device=input.device)
    # uint32 view of the bf16 input: (M, N//2). Mark the row stride divisible by 4 words so the
    # compiler proves the 128-bit (4-word) copy atoms are aligned -- this is what makes the load
    # vectorize (the aligned-word load path).
    words = input.view(torch.uint32)  # (M, N//2)
    mWords = (from_dlpack(words, assumed_align=16)
              .mark_layout_dynamic(leading_dim=1)
              .mark_compact_shape_dynamic(mode=1, divisibility=4))
    output_cute = from_dlpack(output, assumed_align=16)
    scale_cute = from_dlpack(scale)
    mxfp8_swizzle_v3_jit(mWords, output_cute, scale_cute, M, N, ncb)
    return output, scale.view(nrb, ncb, 32, 16).view(torch.float8_e8m0fnu)


MXFP8_SWIZZLE_V3 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleGold, cute_fn=mxfp8_swizzle_v3
)


# ---------------------------------------------------------------------------
# mxfp8_swizzle_v4: same 2-D 32x128 tile, same 16 bf16/thread, same numerics as v3 -- but WITHOUT the
# uint32-word load. It turns out mWords was never necessary.
#
# What differs from v3 (the ONLY difference -- everything downstream is identical):
#   v3:  loads 16 bf16 as one 128-bit uint32-word copy path (mWords: input viewed as uint32 with a
#        divisibility=4 row stride, two CopyUniversalOp atoms into 4 u32 regs each, _bf16x2_to_f32x2
#        register unpack). This exists only because a SINGLE 16-wide bf16 fragment .load() fails to
#        vectorize (it serializes to 1 sector/element, 268M sectors) -- the compiler won't fuse it
#        into 2xLDG.128 without a proven-aligned pointer.
#   v4:  feeds the bf16 input DIRECTLY (no uint32 view, no host divisibility marking) and issues TWO
#        separate 8-elem fragment .load()s. Each 8-bf16 run is 16B = one LDG.128 (the v1 case, which
#        always vectorizes); the second load's +16B offset is a static constant added to a 16-aligned
#        base, and the compiler propagates that alignment on its own -> both loads emit LDG.128.
#
# The catch and its fix: the two loads must be CONCAT'd back into a single 16-wide value before the
# reduce/cast, or you pay for it. There is no SSA-level concat, so the concat goes through a register
# fragment: the two LDG.128s write adjacent halves of one 16-elem rmem tensor, then a single .load()
# reads it back as one 16-wide SSA value -- restoring v3's single 16-wide reduce, single fp8 cast, and
# single STG.128 store. (An earlier attempt that kept the halves split -- two 8-wide reduces + a
# cute.max, and two STG.64 stores -- cost ~1.7pt: the store did NOT coalesce as two STG.64, and the
# split compute added latency this near-roofline kernel couldn't hide.)
#
# Result (16384^2, B200, cutlass-dsl 4.6.0): v4 = 81.1% vs v3 = 80.7% -- a hair FASTER, and much
# simpler. ncu: load 8.39M sectors and store 2.36M sectors, both IDENTICAL to v3 (two LDG.128 + one
# STG.128), 29 vs 28 regs. So the whole word-load apparatus (mWords / divisibility=4 /
# _bf16x2_to_f32x2 / _load_bf16x16_as_f32) is dead weight; v4 deletes it. Bit-exact vs v3 and gold.
# `rounding_mode` adds a compile-time stochastic specialization to this same kernel: RTNE retains the
# original vector cast, while STOCHASTIC uses one Philox4x32 call and four Blackwell e4m3x4 cvt.rs
# instructions per thread. Both modes keep the same load, reduction, scale-write, and launch paths.
# Ragged M and K%128 tails launch over complete padded 128x128 swizzle atoms. Padded lanes load zero,
# skip qdata stores, and explicitly write zero scale bytes; the aligned specialization stays unchanged.
_MXS4_HALF = _MXS3_VPT // 2   # 8 elems per sub-load (2 halves of the 16-elem/thread run)


@cute.jit
def _mxfp8_v4_quantize(thrInput, rcp, mSeed: cute.Tensor, flat_start, stochastic: cutlass.Constexpr):
    scaled = thrInput * rcp
    if cutlass.const_expr(stochastic):
        # One Philox call supplies the four random words needed by this thread's four e4m3x4
        # conversions. Keying by the global 16-element run makes the result independent of tiling.
        frgKey = cute.make_rmem_tensor(cute.make_layout(2), mSeed.element_type)
        cute.copy(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mSeed.element_type),
            mSeed,
            frgKey,
        )
        key64 = cute.recast_tensor(frgKey, dtype=cutlass.Uint64)
        k0 = cutlass.Uint32(key64[0] & cutlass.Uint64(0xFFFFFFFF))
        k1 = cutlass.Uint32(key64[0] >> 32)
        ctr = key64[1] + cutlass.Uint64(flat_start // _MXS3_VPT)
        c0 = cutlass.Uint32(ctr & cutlass.Uint64(0xFFFFFFFF))
        c1 = cutlass.Uint32(ctr >> 32)
        zero = cutlass.Uint32(0)
        r0, r1, r2, r3 = _philox_4x32(c0, c1, zero, zero, k0, k1)

        qwords = cute.make_rmem_tensor(cute.make_layout(_MXS3_VPT // 4), cutlass.Uint32)
        qwords[0] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[0], scaled[1], scaled[2], scaled[3], r0
        )
        qwords[1] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[4], scaled[5], scaled[6], scaled[7], r1
        )
        qwords[2] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[8], scaled[9], scaled[10], scaled[11], r2
        )
        qwords[3] = _cvt_rs_satfinite_e4m3x4_f32(
            scaled[12], scaled[13], scaled[14], scaled[15], r3
        )
        return cute.recast_tensor(qwords, dtype=cutlass.Float8E4M3FN).load()
    else:
        # Preserve v4's original RTNE expression exactly in the default specialization.
        return scaled.to(cutlass.Float8E4M3FN)


@cute.kernel
def mxfp8_swizzle_v4_kernel(
    gInput: cute.Tensor,   # 2-D bf16 input, zipped_divide'd into (32,128) tiles
    gOutput: cute.Tensor,
    mScaleLogical: cute.Tensor,
    mSeed: cute.Tensor,
    input_tv_layout: cute.Layout,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ragged: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    row = bidy * _MXS3_TM + tidx // _MXS3_TPR
    col_start = bidx * _MXS3_TN + (tidx % _MXS3_TPR) * _MXS3_VPT

    # TWO separate 8-elem fragment loads (each a clean LDG.128), concat'd into one 16-elem rmem
    # fragment via adjacent register writes, then read back as a SINGLE 16-wide SSA value so the whole
    # downstream (reduce, mul, fp8 cast, store) is v3's single-16-wide path -- no per-half cute.max or
    # store-side concat. There is no SSA-level concat; the concat has to happen through a register
    # fragment.
    blkInput = gInput[((None, None), (bidy, bidx))]
    tidfrgInput = cute.composition(blkInput, input_tv_layout)
    thrFrg = tidfrgInput[(tidx, None)]              # this thread's 16 contiguous cols
    blkOutput = gOutput[((None, None), (bidy, bidx))]
    tidfrgOutput = cute.composition(blkOutput, input_tv_layout)

    # The compile-time term makes every CTA take the full path for aligned shapes. For ragged shapes,
    # the remaining test is CTA-uniform and confines zero-fill/predication to boundary CTAs.
    use_full_tile = cutlass.const_expr(not ragged) or (
        ((bidy + 1) * _MXS3_TM <= M) & ((bidx + 1) * _MXS3_TN <= N)
    )
    # Keep each complete path inside this branch: CuTe cannot carry the mutable rmem fragment across
    # dynamic control flow, and folding the bounds into per-thread load/store predicates slows every
    # interior CTA of a ragged launch.
    if use_full_tile:
        frg2 = cute.tiled_divide(thrFrg, (_MXS4_HALF,))
        rIn = cute.make_rmem_tensor(_MXS3_VPT, cutlass.BFloat16)
        ri2 = cute.tiled_divide(rIn, (_MXS4_HALF,))
        ri2[(None, 0)] = frg2[(None, 0)].load()
        ri2[(None, 1)] = frg2[(None, 1)].load()

        thrInput = rIn.load().to(cutlass.Float32)
        thrA_abs = cute.math.absf(thrInput)
        thrA_amax16 = thrA_abs.reduce(
            cute.ReductionOp.MAX, init_val=0.0, reduction_profile=1
        )
        thrA_amax32 = cute.arch.warp_reduction_max(
            thrA_amax16, threads_in_group=_MXS3_LANES
        )
        rcp, biased = _e8m0(thrA_amax32)
        tidfrgOutput[(tidx, None)] = _mxfp8_v4_quantize(
            thrInput, rcp, mSeed, row * N + col_start, stochastic
        )

        if tidx % _MXS3_LANES == 0:
            mScaleLogical[(row, col_start // 32)] = biased.to(mScaleLogical.element_type)
    else:
        # Every lane must participate in the reduction, but padded lanes must not dereference qdata.
        frg2 = cute.tiled_divide(thrFrg, (_MXS4_HALF,))
        rIn = cute.make_rmem_tensor(_MXS3_VPT, cutlass.BFloat16)
        rIn.fill(0)
        ri2 = cute.tiled_divide(rIn, (_MXS4_HALF,))
        if row < M:
            if col_start < N:
                ri2[(None, 0)] = frg2[(None, 0)].load()
                ri2[(None, 1)] = frg2[(None, 1)].load()

        thrInput = rIn.load().to(cutlass.Float32)
        thrA_abs = cute.math.absf(thrInput)
        thrA_amax16 = thrA_abs.reduce(
            cute.ReductionOp.MAX, init_val=0.0, reduction_profile=1
        )
        thrA_amax32 = cute.arch.warp_reduction_max(
            thrA_amax16, threads_in_group=_MXS3_LANES
        )
        rcp, biased = _e8m0(thrA_amax32)
        qdata = _mxfp8_v4_quantize(
            thrInput, rcp, mSeed, row * N + col_start, stochastic
        )

        if row < M:
            if col_start < N:
                tidfrgOutput[(tidx, None)] = qdata

        if tidx % _MXS3_LANES == 0:
            col = col_start // 32
            if row < M:
                if col_start < N:
                    mScaleLogical[(row, col)] = biased.to(mScaleLogical.element_type)
                else:
                    mScaleLogical[(row, col)] = cutlass.Uint8(0)
            else:
                mScaleLogical[(row, col)] = cutlass.Uint8(0)


@cute.jit
def mxfp8_swizzle_v4_jit(
    mInput: cute.Tensor,
    mOutput: cute.Tensor,
    mScale: cute.Tensor,
    mSeed: cute.Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ncb: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
):
    padded_M = _ceil_div(M, 128) * 128
    padded_N = _ceil_div(N, 128) * 128
    tiler_mn = (_MXS3_TM, _MXS3_TN)
    input_tv_layout = cute.make_layout(
        ((_MXS3_TPR, _MXS3_TM), (_MXS3_VPT,)),
        stride=((_MXS3_TM * _MXS3_VPT, 1), (_MXS3_TM,)),
    )
    gInput = cute.zipped_divide(mInput, tiler_mn)
    gOutput = cute.zipped_divide(mOutput, tiler_mn)

    # Logical (row, 1x32-col-group) view over the physical (nrb,ncb,32,16) scale buffer. The
    # hierarchical modes split row into (b, a, br) and col into (c4, bc), and their strides encode
    # the blocked/swizzled destination directly, so the kernel need not calculate a physical offset.
    # This also saves one register versus the explicit _swizzle_flat address calculation.
    nrb = cute.size(mScale) // (ncb * 32 * 16)
    scale_layout = cute.make_layout(
        ((32, 4, nrb), (4, ncb)),
        stride=((16, 4, ncb * 32 * 16), (1, 32 * 16)),
    )
    mScaleLogical = cute.make_tensor(mScale.iterator, scale_layout)

    mxfp8_swizzle_v4_kernel(
        gInput, gOutput, mScaleLogical, mSeed, input_tv_layout, M, N,
        M != padded_M or N != padded_N,
        stochastic,
    ).launch(
        # Walk every slot of the padded scale grid. Predicates prevent padded coordinates from
        # touching the unpadded qdata input/output.
        grid=(padded_N // _MXS3_TN, padded_M // _MXS3_TM, 1),
        block=(_MXS3_THREADS, 1, 1),
    )


def mxfp8_swizzle_v4(
    input: torch.Tensor,
    key: torch.Tensor | None = None,
    rounding_mode: str = "rtne",
    **kwargs,
):
    assert len(input.shape) == 2, "unsupported"
    assert input.is_contiguous(), "unsupported"
    assert input.dtype == torch.bfloat16, "v4 is bf16-only"
    rounding_mode = str(getattr(rounding_mode, "value", rounding_mode)).lower()
    assert rounding_mode in ("rtne", "stochastic"), f"unsupported rounding_mode: {rounding_mode}"
    stochastic = rounding_mode == "stochastic"
    if stochastic:
        assert key is not None, "stochastic rounding requires a Philox key"
        assert key.device == input.device, "input and Philox key must be on the same device"
        assert key.dtype == torch.uint64 and key.numel() == 2, "Philox key must be uint64[2]"
    else:
        assert key is None, "RTNE rounding does not use a Philox key"
    M, N = input.shape
    assert M > 0 and N > 0, "v4 requires non-empty dimensions"
    assert N % 32 == 0, "v4 requires K % 32 == 0"
    ngc = N // 32
    nrb, ncb = _ceil_div(M, 128), _ceil_div(ngc, 4)
    output = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=input.device)
    scale = torch.empty(nrb * ncb * 32 * 16, dtype=torch.uint8, device=input.device)
    input_cute = from_dlpack(input, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)
    scale_cute = from_dlpack(scale)
    # RTNE's compile-time branch never reads mSeed, so reuse an existing tensor as its dummy kernel
    # argument and avoid allocating any RNG state on the default path.
    seed_cute = from_dlpack(key.reshape(-1).view(torch.int64)) if stochastic else scale_cute
    mxfp8_swizzle_v4_jit(
        input_cute, output_cute, scale_cute, seed_cute, M, N, ncb, stochastic
    )
    return output, scale.view(nrb, ncb, 32, 16).view(torch.float8_e8m0fnu)


MXFP8_SWIZZLE_V4 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleGold, cute_fn=mxfp8_swizzle_v4
)


MXFP8_SWIZZLE_SR_V4 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleSRGold,
    cute_fn=partial(mxfp8_swizzle_v4, rounding_mode="stochastic"),
)


# ---------------------------------------------------------------------------
# mxfp8_swizzle_v5: the best of v1 and v4.
#   from v1: the FLAT 1-D grid -- input flattened to 1-D, tile = one contiguous run, so each CTA's
#            memory footprint is a single contiguous slab (v1's DRAM-locality advantage; v4's 2-D
#            32x128 tile scatters each CTA across 32 rows).
#   from v4: 16 bf16/thread loaded as TWO 8-elem fragment .load()s (each a clean LDG.128) concat'd in
#            a register fragment into one 16-wide value, and a single 16-fp8 STG.128 store. (No
#            uint32-word path: two plain loads vectorize where a single 16-wide .load() would not.)
#
# At 256 threads x 16 elems the flat tile is 4096 contiguous elements. val_layout = (16):(1) gives
# each thread a contiguous 16-run; a 1x32 block is 2 threads, so warp_reduction_max(threads_in_group=2)
# (ONE shuffle stage) and every 2nd thread stores the scale -- same block arithmetic as v4, but on the
# flat tile. Ragged shapes use a padded logical flat view: padded lanes load zero, skip qdata stores,
# and explicitly write zero scale bytes; the aligned specialization keeps the original contiguous view.
_MXS5_VPT = 16                        # elems/thread (16 bf16 = 2x LDG.128 / 1x STG.128)
_MXS5_THREADS = 256
_MXS5_HALF = _MXS5_VPT // 2           # 8 elems per sub-load
_MXS5_LANES = 32 // _MXS5_VPT         # 2 threads per 1x32 block


@cute.kernel
def mxfp8_swizzle_v5_kernel(
    gInput: cute.Tensor,
    gOutput: cute.Tensor,
    mScaleLogical: cute.Tensor,
    gId: cute.Tensor,
    input_tv_layout: cute.Layout,
    orig_shape: cute.Shape,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    padded_N: cutlass.Constexpr,
    ragged: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    if cutlass.const_expr(ragged):
        blk_coord = ((None,), bidx)
        tidfrgInput = cute.composition(gInput[blk_coord], input_tv_layout)
        tidfrgOutput = cute.composition(gOutput[blk_coord], input_tv_layout)
        thr_coord = (tidx, None)
        padded_tile = cute.size(input_tv_layout)
        padded_vpt = cute.size(input_tv_layout, mode=[1])
        flat_idx = bidx * padded_tile + tidx * padded_vpt
        row = flat_idx // padded_N
        col_start = flat_idx % padded_N

        thrFrg = tidfrgInput[thr_coord]
        tile_start = bidx * padded_tile
        tile_end = tile_start + padded_tile - 1
        if cutlass.const_expr(N == padded_N):
            # With no column padding, the real matrix is one contiguous prefix of the padded grid.
            use_full_tile = tile_end < M * N
        elif cutlass.const_expr(padded_N <= padded_tile):
            # A flat CTA spans at least one entire padded row, so every CTA includes its column tail.
            use_full_tile = False
        else:
            first_row = tile_start // padded_N
            last_row = tile_end // padded_N
            use_full_tile = (
                (last_row < M) & (first_row == last_row) & (tile_end % padded_N < N)
            )

        if use_full_tile:
            frg2 = cute.tiled_divide(thrFrg, (_MXS5_HALF,))
            rIn = cute.make_rmem_tensor(_MXS5_VPT, cutlass.BFloat16)
            ri2 = cute.tiled_divide(rIn, (_MXS5_HALF,))
            ri2[(None, 0)] = frg2[(None, 0)].load()
            ri2[(None, 1)] = frg2[(None, 1)].load()

            thrInput = rIn.load().to(cutlass.Float32)
            thrA_amax16 = cute.math.absf(thrInput).reduce(
                cute.ReductionOp.MAX, init_val=0.0, reduction_profile=1
            )
            thrA_amax32 = cute.arch.warp_reduction_max(
                thrA_amax16, threads_in_group=_MXS5_LANES
            )
            rcp, biased = _e8m0(thrA_amax32)
            tidfrgOutput[thr_coord] = (thrInput * rcp).to(cutlass.Float8E4M3FN)

            if tidx % _MXS5_LANES == 0:
                mScaleLogical[flat_idx // 32] = biased.to(mScaleLogical.element_type)
        else:
            # All threads execute the two-lane reduction. Padded lanes contribute zero but never
            # dereference qdata; K%32 makes both lanes of a quantization group equally valid.
            frg2 = cute.tiled_divide(thrFrg, (_MXS5_HALF,))
            rIn = cute.make_rmem_tensor(_MXS5_VPT, cutlass.BFloat16)
            rIn.fill(0)
            ri2 = cute.tiled_divide(rIn, (_MXS5_HALF,))
            if row < M:
                if col_start < N:
                    ri2[(None, 0)] = frg2[(None, 0)].load()
                    ri2[(None, 1)] = frg2[(None, 1)].load()

            thrInput = rIn.load().to(cutlass.Float32)
            thrA_amax16 = cute.math.absf(thrInput).reduce(
                cute.ReductionOp.MAX, init_val=0.0, reduction_profile=1
            )
            thrA_amax32 = cute.arch.warp_reduction_max(
                thrA_amax16, threads_in_group=_MXS5_LANES
            )
            rcp, biased = _e8m0(thrA_amax32)
            qdata = (thrInput * rcp).to(cutlass.Float8E4M3FN)

            if row < M:
                if col_start < N:
                    tidfrgOutput[thr_coord] = qdata

            # The grid covers the complete padded scale tensor, so every scale byte is initialized
            # even though the host allocation uses torch.empty.
            if tidx % _MXS5_LANES == 0:
                gblk = flat_idx // 32
                if row < M:
                    if col_start < N:
                        mScaleLogical[gblk] = biased.to(mScaleLogical.element_type)
                    else:
                        mScaleLogical[gblk] = cutlass.Uint8(0)
                else:
                    mScaleLogical[gblk] = cutlass.Uint8(0)
    else:
        # Keep the aligned specialization identical to the original v5 kernel.
        blk_coord = ((None,), bidx)
        blkInput = gInput[blk_coord]
        blkOutput = gOutput[blk_coord]
        blkId = gId[blk_coord]
        tidfrgInput = cute.composition(blkInput, input_tv_layout)
        tidfrgOutput = cute.composition(blkOutput, input_tv_layout)
        tidfrgId = cute.composition(blkId, input_tv_layout)

        thr_coord = (tidx, None)
        thrId = tidfrgId[thr_coord]
        if cute.elem_less(thrId[0], orig_shape):
            # v4's load: two 8-elem LDG.128 concat'd in a register fragment -> one 16-wide value.
            aThrFrg = tidfrgInput[thr_coord]
            aFrg2 = cute.tiled_divide(aThrFrg, (_MXS5_HALF,))
            aRIn = cute.make_rmem_tensor(_MXS5_VPT, cutlass.BFloat16)
            aRi2 = cute.tiled_divide(aRIn, (_MXS5_HALF,))
            aRi2[(None, 0)] = aFrg2[(None, 0)].load()
            aRi2[(None, 1)] = aFrg2[(None, 1)].load()

            aThrInput = aRIn.load().to(cutlass.Float32)
            aThrA_abs = cute.math.absf(aThrInput)
            aThrA_amax16 = aThrA_abs.reduce(
                cute.ReductionOp.MAX, init_val=0.0, reduction_profile=1
            )
            aThrA_amax32 = cute.arch.warp_reduction_max(
                aThrA_amax16, threads_in_group=_MXS5_LANES
            )
            aRcp, aBiased = _e8m0(aThrA_amax32)
            aQdata = (aThrInput * aRcp).to(cutlass.Float8E4M3FN)

            aThrOutput = tidfrgOutput[thr_coord]
            aThrOutput[None] = aQdata

            if tidx % _MXS5_LANES == 0:
                aTile = cute.size(input_tv_layout)
                aVpt = cute.size(input_tv_layout, mode=[1])
                aGblk = (bidx * aTile + tidx * aVpt) // 32
                mScaleLogical[aGblk] = aBiased.to(mScaleLogical.element_type)


@cute.jit
def mxfp8_swizzle_v5_jit(
    mInput: cute.Tensor,
    mOutput: cute.Tensor,
    mScale: cute.Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ncb: cutlass.Constexpr,
):
    nrb = cute.size(mScale) // (ncb * 32 * 16)
    padded_M = nrb * 128
    padded_N = ncb * 128
    ragged = M != padded_M or N != padded_N

    thr_layout = cute.make_ordered_layout((_MXS5_THREADS,), order=(0,))   # (256):(1)
    val_layout = cute.make_ordered_layout((_MXS5_VPT,), order=(0,))       # (16):(1)
    tiler_mn, input_tv_layout = cute.make_layout_tv(thr_layout, val_layout)  # (4096,) (256,16):(16,1)

    # Give the physically flat (nrb,ncb,32,16) scale buffer a rank-1 LOGICAL view in row-major
    # (row, 1x32-col-group) order. A scalar logical coordinate is decomposed colexicographically as
    #   (c4, bc, b, a, br),
    # where row = br*128 + a*32 + b and col = bc*4 + c4. The strides then map it directly to the
    # physical blocked layout [br, bc, b, a*4+c4], replacing the explicit _swizzle_flat arithmetic.
    scale_layout = cute.make_layout(
        ((4, ncb, 32, 4, nrb),),
        stride=((1, 32 * 16, 16, 4, ncb * 32 * 16),),
    )
    mScaleLogical = cute.make_tensor(mScale.iterator, scale_layout)

    if cutlass.const_expr(ragged):
        # Rank-1 logical views over the padded matrix. Real rows retain stride N; padded coordinates
        # may form pointers outside storage but are never dereferenced by the predicated kernel.
        padded_layout = cute.make_layout(((padded_N, padded_M),), stride=((1, N),))
        mInputGrid = cute.make_tensor(mInput.iterator, padded_layout)
        mOutputGrid = cute.make_tensor(mOutput.iterator, padded_layout)
    else:
        mInputGrid = mInput
        mOutputGrid = mOutput

    gInput = cute.zipped_divide(mInputGrid, tiler_mn)
    gOutput = cute.zipped_divide(mOutputGrid, tiler_mn)
    mId = cute.make_identity_tensor(mInputGrid.shape)
    gId = cute.zipped_divide(mId, tiler_mn)
    mxfp8_swizzle_v5_kernel(
        gInput, gOutput, mScaleLogical, gId, input_tv_layout, mInputGrid.shape,
        M, N, padded_N, ragged,
    ).launch(
        grid=(cute.size(gInput, mode=[1]), 1, 1),
        block=(cute.size(input_tv_layout, mode=[0]), 1, 1),
    )


def mxfp8_swizzle_v5(input: torch.Tensor, **kwargs):
    assert len(input.shape) == 2, "unsupported"
    assert input.is_contiguous(), "unsupported"
    assert input.dtype == torch.bfloat16, "v5 is bf16-only"
    M, N = input.shape
    assert M > 0 and N > 0, "v5 requires non-empty dimensions"
    assert N % 32 == 0, "v5 requires K % 32 == 0"
    ngc = N // 32
    nrb, ncb = _ceil_div(M, 128), _ceil_div(ngc, 4)
    input = input.view(-1)
    output = torch.empty(input.shape, dtype=torch.float8_e4m3fn, device=input.device)
    scale = torch.empty(nrb * ncb * 32 * 16, dtype=torch.uint8, device=input.device)
    input_cute = from_dlpack(input, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)
    scale_cute = from_dlpack(scale)
    mxfp8_swizzle_v5_jit(input_cute, output_cute, scale_cute, M, N, ncb)
    return output.view(M, N), scale.view(nrb, ncb, 32, 16).view(torch.float8_e8m0fnu)


MXFP8_SWIZZLE_V5 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleGold, cute_fn=mxfp8_swizzle_v5
)


# ---------------------------------------------------------------------------
# NVFP4 casts. Every mode implements the same 1x16 quantization and blocked scale layout:
#   inner = clamp((amax / 6) * outer, e4m3_tiny, 448) -> e4m3
#   qdata = fp4_rne(input * outer / inner), packed two values per byte
# The dim-K direct version uses v4-style vector loads for small problems and switches to a
# row-owned, load-ILP mapping for larger ones. The TMA and pipelined RHT families live in
# nvfp4_tma.py and nvfp4_pipelined.py, respectively.
_NVFP4_GROUP = 16

_NVFP4_DIRECT_VPT = 16
_NVFP4_DIRECT_QVPT = _NVFP4_DIRECT_VPT // 2
_NVFP4_DIRECT_HALF = 8
_NVFP4_DIRECT_WARPS = 4
_NVFP4_DIRECT_ROW_THREADS = _NVFP4_DIRECT_WARPS * 32
_NVFP4_DIRECT_MAX_XSPLIT = 8
_NVFP4_DIRECT_MAX_ILP = 4
_NVFP4_DIRECT_TILE_M = 32
_NVFP4_DIRECT_TILE_N = 128
_NVFP4_DIRECT_TILE_TPR = _NVFP4_DIRECT_TILE_N // _NVFP4_GROUP
_NVFP4_DIRECT_TILE_THREADS = _NVFP4_DIRECT_TILE_M * _NVFP4_DIRECT_TILE_TPR




@cute.kernel
def nvfp4_swizzle_direct_kernel(
    mInput: cute.Tensor,
    mOutput: cute.Tensor,
    mScale: cute.Tensor,
    mOuter: cute.Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ncb: cutlass.Constexpr,
    xsplit: cutlass.Constexpr,
    ilp: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    x_tile, row_tile, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    row = row_tile * _NVFP4_DIRECT_WARPS + warp

    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    store128_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=128
    )
    store64_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=64
    )
    frgOuter = cute.make_rmem_tensor(cute.make_layout(1), mOuter.element_type)
    cute.copy(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mOuter.element_type),
        mOuter,
        frgOuter,
    )

    row128 = row % 128
    row_base = (
        (row // 128) * ncb * 32 + row128 % 32
    ) * 16 + (row128 // 32) * 4
    mScale16 = cute.make_tensor(
        cute.recast_ptr(mScale.iterator, dtype=cutlass.Uint16),
        cute.make_layout(cute.size(mScale) // 2),
    )
    mScale32 = cute.make_tensor(
        cute.recast_ptr(mScale.iterator, dtype=cutlass.Uint32),
        cute.make_layout(cute.size(mScale) // 4),
    )

    # A warp owns one row. Its lanes and the x grid stripe 32-element units (two NVFP4 groups)
    # across that row; each ILP batch issues all vector loads before beginning the reductions.
    groups32 = N // 32
    groups_per_wave = xsplit * 32
    base = x_tile * 32 + lane
    if row < M:
        while base < groups32:
            rInput = cute.make_rmem_tensor(ilp * 32, cutlass.BFloat16)
            rInputGroups = cute.tiled_divide(rInput, (32,))
            for item in cutlass.range_constexpr(ilp):
                group32 = base + item * groups_per_wave
                if group32 < groups32:
                    offset = cute.assume(row * N + group32 * 32, divby=8)
                    rInputVecs = cute.tiled_divide(
                        rInputGroups[(None, item)], (_NVFP4_DIRECT_HALF,)
                    )
                    for vec in cutlass.range_constexpr(4):
                        cute.copy(
                            load_atom,
                            cute.make_tensor(
                                mInput.iterator
                                + offset
                                + vec * _NVFP4_DIRECT_HALF,
                                cute.make_layout(_NVFP4_DIRECT_HALF),
                            ),
                            rInputVecs[(None, vec)],
                        )

            for item in cutlass.range_constexpr(ilp):
                group32 = base + item * groups_per_wave
                if group32 < groups32:
                    rQ = cute.make_rmem_tensor(16, cutlass.Uint8)
                    rQGroups = cute.tiled_divide(rQ, (_NVFP4_DIRECT_QVPT,))
                    rScale = cute.make_rmem_tensor(2, cutlass.Uint8)
                    for group16 in cutlass.range_constexpr(2):
                        values = cute.make_tensor(
                            rInputGroups[(None, item)].iterator
                            + group16 * _NVFP4_GROUP,
                            cute.make_layout(_NVFP4_GROUP),
                        ).load().to(cutlass.Float32)
                        qdata, inner_e4m3 = _nvfp4_quantize_x16(
                            values, frgOuter[0]
                        )
                        rQGroups[(None, group16)].store(qdata)
                        rScale[group16] = inner_e4m3

                    output_offset = row * (N // 2) + group32 * 16
                    if cutlass.const_expr(N % 32 == 0):
                        output_offset = cute.assume(output_offset, divby=16)
                        cute.copy(
                            store128_atom,
                            rQ,
                            cute.make_tensor(
                                mOutput.iterator + output_offset,
                                cute.make_layout(16),
                            ),
                        )
                    else:
                        output_offset = cute.assume(output_offset, divby=8)
                        rQHalves = cute.tiled_divide(rQ, (8,))
                        for half in cutlass.range_constexpr(2):
                            cute.copy(
                                store64_atom,
                                rQHalves[(None, half)],
                                cute.make_tensor(
                                    mOutput.iterator + output_offset + half * 8,
                                    cute.make_layout(8),
                                ),
                            )

                    scale_col = group32 * 2
                    scale_flat = cute.assume(
                        row_base
                        + (scale_col // 4) * 32 * 16
                        + scale_col % 4,
                        divby=2,
                    )
                    mScale16[scale_flat // 2] = cute.recast_tensor(
                        rScale, dtype=cutlass.Uint16
                    )[0]
            base = base + groups_per_wave * ilp

        # K may contain one final unpaired 1x16 block. One lane handles its aligned input load and
        # naturally aligned eight-byte packed output store.
        if cutlass.const_expr(N % 32 == 16):
            if (x_tile == 0) & (lane == 0):
                rTail = cute.make_rmem_tensor(_NVFP4_GROUP, cutlass.BFloat16)
                rTailVecs = cute.tiled_divide(rTail, (_NVFP4_DIRECT_HALF,))
                tail_offset = cute.assume(
                    row * N + N - _NVFP4_GROUP, divby=8
                )
                for vec in cutlass.range_constexpr(2):
                    cute.copy(
                        load_atom,
                        cute.make_tensor(
                            mInput.iterator
                            + tail_offset
                            + vec * _NVFP4_DIRECT_HALF,
                            cute.make_layout(_NVFP4_DIRECT_HALF),
                        ),
                        rTailVecs[(None, vec)],
                    )
                qtail, stail = _nvfp4_quantize_x16(
                    rTail.load().to(cutlass.Float32), frgOuter[0]
                )
                rQTail = cute.make_rmem_tensor(8, cutlass.Uint8)
                rQTail.store(qtail)
                cute.copy(
                    store64_atom,
                    rQTail,
                    cute.make_tensor(
                        mOutput.iterator
                        + cute.assume(
                            row * (N // 2) + N // 2 - 8, divby=8
                        ),
                        cute.make_layout(8),
                    ),
                )
                tail_col = N // _NVFP4_GROUP - 1
                mScale[(
                    row_base
                    + (tail_col // 4) * 32 * 16
                    + tail_col % 4
                )] = stail

        # The physical scale row has up to three slots beyond the logical K extent.
        if (x_tile == 0) & (lane == 0):
            scale_col = N // _NVFP4_GROUP
            while scale_col < ncb * 4:
                mScale[(
                    row_base
                    + (scale_col // 4) * 32 * 16
                    + scale_col % 4
                )] = cutlass.Uint8(0)
                scale_col = scale_col + 1
    else:
        # Only one x-split initializes each padded row. Lanes distribute its four-byte blocks.
        if x_tile == 0:
            scale_block = lane
            while scale_block < ncb:
                scale_flat = cute.assume(
                    row_base + scale_block * 32 * 16, divby=4
                )
                mScale32[scale_flat // 4] = cutlass.Uint32(0)
                scale_block = scale_block + 32


@cute.jit
def nvfp4_swizzle_direct_jit(
    mInput: cute.Tensor,
    mOutput: cute.Tensor,
    mScale: cute.Tensor,
    mOuter: cute.Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ncb: cutlass.Constexpr,
    xsplit: cutlass.Constexpr,
    ilp: cutlass.Constexpr,
):
    padded_M = _ceil_div(M, 128) * 128
    nvfp4_swizzle_direct_kernel(
        mInput,
        mOutput,
        mScale,
        mOuter,
        M,
        N,
        ncb,
        xsplit,
        ilp,
    ).launch(
        grid=(xsplit, padded_M // _NVFP4_DIRECT_WARPS, 1),
        block=(_NVFP4_DIRECT_ROW_THREADS, 1, 1),
    )


@cute.kernel
def nvfp4_swizzle_direct_group_kernel(
    gInput: cute.Tensor,
    gOutput: cute.Tensor,
    mScaleLogical: cute.Tensor,
    mOuter: cute.Tensor,
    input_tv_layout: cute.Layout,
    output_tv_layout: cute.Layout,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ncb: cutlass.Constexpr,
    ragged: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    n_tile, m_tile, _ = cute.arch.block_idx()
    row = m_tile * _NVFP4_DIRECT_TILE_M + tidx // _NVFP4_DIRECT_TILE_TPR
    col_start = (
        n_tile * _NVFP4_DIRECT_TILE_N
        + (tidx % _NVFP4_DIRECT_TILE_TPR) * _NVFP4_GROUP
    )
    blkInput = gInput[((None, None), (m_tile, n_tile))]
    thrInput = cute.composition(blkInput, input_tv_layout)[(tidx, None)]
    blkOutput = gOutput[((None, None), (m_tile, n_tile))]
    thrOutput = cute.composition(blkOutput, output_tv_layout)[(tidx, None)]

    frgOuter = cute.make_rmem_tensor(cute.make_layout(1), mOuter.element_type)
    cute.copy(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mOuter.element_type),
        mOuter,
        frgOuter,
    )

    use_full_tile = cutlass.const_expr(not ragged) or (
        ((m_tile + 1) * _NVFP4_DIRECT_TILE_M <= M)
        & ((n_tile + 1) * _NVFP4_DIRECT_TILE_N <= N)
    )
    if use_full_tile:
        rInput = cute.make_rmem_tensor(_NVFP4_GROUP, cutlass.BFloat16)
        rInputVecs = cute.tiled_divide(rInput, (_NVFP4_DIRECT_HALF,))
        inputVecs = cute.tiled_divide(thrInput, (_NVFP4_DIRECT_HALF,))
        rInputVecs[(None, 0)] = inputVecs[(None, 0)].load()
        rInputVecs[(None, 1)] = inputVecs[(None, 1)].load()
        qdata, scale = _nvfp4_quantize_x16(
            rInput.load().to(cutlass.Float32), frgOuter[0]
        )
        thrOutput.store(qdata)
        mScaleLogical[(row, col_start // _NVFP4_GROUP)] = scale
    else:
        valid = (row < M) & (col_start < N)
        scale_col = col_start // _NVFP4_GROUP
        if valid:
            rInput = cute.make_rmem_tensor(_NVFP4_GROUP, cutlass.BFloat16)
            rInputVecs = cute.tiled_divide(rInput, (_NVFP4_DIRECT_HALF,))
            inputVecs = cute.tiled_divide(thrInput, (_NVFP4_DIRECT_HALF,))
            rInputVecs[(None, 0)] = inputVecs[(None, 0)].load()
            rInputVecs[(None, 1)] = inputVecs[(None, 1)].load()
            qdata, scale = _nvfp4_quantize_x16(
                rInput.load().to(cutlass.Float32), frgOuter[0]
            )
            thrOutput.store(qdata)
            mScaleLogical[(row, scale_col)] = scale
        else:
            if (row < _ceil_div(M, 128) * 128) & (scale_col < ncb * 4):
                mScaleLogical[(row, scale_col)] = cutlass.Uint8(0)


@cute.jit
def nvfp4_swizzle_direct_group_jit(
    mInput: cute.Tensor,
    mOutput: cute.Tensor,
    mScale: cute.Tensor,
    mOuter: cute.Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ncb: cutlass.Constexpr,
    ragged: cutlass.Constexpr,
):
    padded_M = _ceil_div(M, 128) * 128
    padded_N = _ceil_div(N, _NVFP4_DIRECT_TILE_N) * _NVFP4_DIRECT_TILE_N
    gInput = cute.zipped_divide(
        mInput, (_NVFP4_DIRECT_TILE_M, _NVFP4_DIRECT_TILE_N)
    )
    gOutput = cute.zipped_divide(
        mOutput, (_NVFP4_DIRECT_TILE_M, _NVFP4_DIRECT_TILE_N // 2)
    )
    input_tv_layout = cute.make_layout(
        (
            (_NVFP4_DIRECT_TILE_TPR, _NVFP4_DIRECT_TILE_M),
            (_NVFP4_GROUP,),
        ),
        stride=(
            (_NVFP4_DIRECT_TILE_M * _NVFP4_GROUP, 1),
            (_NVFP4_DIRECT_TILE_M,),
        ),
    )
    output_tv_layout = cute.make_layout(
        (
            (_NVFP4_DIRECT_TILE_TPR, _NVFP4_DIRECT_TILE_M),
            (_NVFP4_DIRECT_QVPT,),
        ),
        stride=(
            (_NVFP4_DIRECT_TILE_M * _NVFP4_DIRECT_QVPT, 1),
            (_NVFP4_DIRECT_TILE_M,),
        ),
    )
    nrb = _ceil_div(M, 128)
    scale_layout = cute.make_layout(
        ((32, 4, nrb), (4, ncb)),
        stride=((16, 4, ncb * 32 * 16), (1, 32 * 16)),
    )
    mScaleLogical = cute.make_tensor(mScale.iterator, scale_layout)
    nvfp4_swizzle_direct_group_kernel(
        gInput,
        gOutput,
        mScaleLogical,
        mOuter,
        input_tv_layout,
        output_tv_layout,
        M,
        N,
        ncb,
        ragged,
    ).launch(
        grid=(
            padded_N // _NVFP4_DIRECT_TILE_N,
            padded_M // _NVFP4_DIRECT_TILE_M,
            1,
        ),
        block=(_NVFP4_DIRECT_TILE_THREADS, 1, 1),
    )




def nvfp4_swizzle_direct(input: torch.Tensor, outer_scale: torch.Tensor, **kwargs):
    M, N = _validate_nvfp4_swizzle_inputs(
        input, outer_scale, "nvfp4_swizzle_direct", 16
    )
    output, scale, nrb, ncb = _allocate_nvfp4_swizzle_outputs(input, M, N)
    row_owned = M * N > 4096 * 4096
    mInput = from_dlpack(
        input.reshape(-1) if row_owned else input, assumed_align=16
    )
    mOutput = from_dlpack(
        output.reshape(-1) if row_owned else output, assumed_align=16
    )
    mScale = from_dlpack(scale, assumed_align=4)
    mOuter = from_dlpack(outer_scale.reshape(1))
    groups32 = N // 32
    xsplit = min(
        _NVFP4_DIRECT_MAX_XSPLIT, max(1, _ceil_div(groups32, 32))
    )
    ilp = min(
        _NVFP4_DIRECT_MAX_ILP,
        max(1, _ceil_div(groups32, xsplit * 32)),
    )
    jit_fn = (
        nvfp4_swizzle_direct_jit
        if row_owned
        else nvfp4_swizzle_direct_group_jit
    )
    compile_args = (
        (mInput, mOutput, mScale, mOuter, M, N, ncb, xsplit, ilp)
        if row_owned
        else (
            mInput,
            mOutput,
            mScale,
            mOuter,
            M,
            N,
            ncb,
            M % 128 != 0 or N % _NVFP4_DIRECT_TILE_N != 0,
        )
    )
    fn = _compiled(
        ("nvfp4_swizzle_direct", M, N, row_owned, xsplit, ilp),
        jit_fn,
        *compile_args,
    )
    fn(mInput, mOutput, mScale, mOuter)
    return (
        output.view(torch.float4_e2m1fn_x2),
        scale.view(nrb, ncb, 32, 16).view(torch.float8_e4m3fn),
    )


NVFP4_SWIZZLE_DIRECT = QuantCastCuteRecipe.from_gold(
    Nvfp4GsSwizzleGold, cute_fn=nvfp4_swizzle_direct
)




ALL_RECIPES = [
    ("deepseek_1x128", FP8_DEEPSEEK_1X128),
    ("deepseek_1x128_dim_m", FP8_DEEPSEEK_1X128_DIM_M),
    ("deepseek_1x128_dim_m_v2", FP8_DEEPSEEK_1X128_DIM_M_V2),
    ("mxfp8_swizzle", MXFP8_SWIZZLE),
    ("mxfp8_swizzle_v2", MXFP8_SWIZZLE_V2),
    ("mxfp8_swizzle_sr_v2", MXFP8_SWIZZLE_SR_V2),
    ("mxfp8_32x32_swizzle_v2", MXFP8_32X32_SWIZZLE_V2),
    ("mxfp8_swizzle_v3", MXFP8_SWIZZLE_V3),
    ("mxfp8_swizzle_v4", MXFP8_SWIZZLE_V4),
    ("mxfp8_swizzle_sr_v4", MXFP8_SWIZZLE_SR_V4),
    ("mxfp8_swizzle_v5", MXFP8_SWIZZLE_V5),
    ("mxfp8_dim_m_swizzle_v2", MXFP8_DIM_M_SWIZZLE_V2),
    ("mxfp8_dim_m_swizzle_sr_v2", MXFP8_DIM_M_SWIZZLE_SR_V2),
    ("mxfp8_dim_km_swizzle_v2", MXFP8_DIM_KM_SWIZZLE_V2),
    ("mxfp8_dim_km_swizzle_sr_v2", MXFP8_DIM_KM_SWIZZLE_SR_V2),
    ("nvfp4_swizzle_direct", NVFP4_SWIZZLE_DIRECT),
    ("nvfp4_swizzle_tma", NVFP4_SWIZZLE_TMA),
    ("nvfp4_dim_m_swizzle_tma", NVFP4_DIM_M_SWIZZLE_TMA),
    ("nvfp4_dim_km_swizzle_tma", NVFP4_DIM_KM_SWIZZLE_TMA),
    ("nvfp4_dim_m_rht_swizzle_tma", NVFP4_DIM_M_RHT_SWIZZLE_TMA),
    ("nvfp4_dim_m_swizzle_rht_sr_tma", NVFP4_DIM_M_SWIZZLE_RHT_SR_TMA),
    ("nvfp4_swizzle_dim_k_dim_m_rht_tma", NVFP4_SWIZZLE_DIM_K_DIM_M_RHT_TMA),
    (
        "nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_tma",
        NVFP4_SWIZZLE_DIM_K_SR_DIM_M_RHT_SR_TMA,
    ),
    (
        "nvfp4_swizzle_dim_k_dim_m_rht_pipelined",
        NVFP4_SWIZZLE_DIM_K_DIM_M_RHT_PIPELINED,
    ),
    (
        "nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined",
        NVFP4_SWIZZLE_DIM_K_SR_DIM_M_RHT_SR_PIPELINED,
    ),
]
