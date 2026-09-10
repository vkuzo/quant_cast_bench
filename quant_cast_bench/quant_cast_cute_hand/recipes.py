# handwritten cute recipes, tracking learning CuTeDSL
#
# Started from FP8_DEEPSEEK_1X128, copied verbatim from quant_cast_cute/recipes.py; this is the
# playground where we iterate on it.

import os
from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector  # typed NVVM e8m0 cvt ops
from cutlass.cute.nvgpu import cpasync, tcgen05  # TMA copies + Blackwell smem layouts
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

import torch

# Gate debug output (host trace-time `print` + device `cute.printf`) behind an env var, read once
# at import. Gate with `cutlass.const_expr(_DEBUG)` inside kernels so that when off the tracer takes
# neither branch -- the printf ops are never emitted (no dead ops, no values kept live). Run with
# `CUTE_DEBUG=1 python -m ...` to enable.
_DEBUG = os.environ.get("CUTE_DEBUG", "0") == "1"

from quant_cast_bench.quant_cast_cute.recipes import QuantCastCuteRecipe, _philox_4x32
from quant_cast_bench.quant_cast_gold.recipes import (
    Deepseek1x128Gold, Deepseek1x128DimMGold, Mxfp8DimKmSwizzleGold,
    Mxfp8DimKmSwizzleSRGold, Mxfp8DimMSwizzleGold, Mxfp8DimMSwizzleSRGold,
    Mxfp8SwizzleGold, Mxfp8SwizzleSRGold,
)

def _ceil_div(num, den):
    return (num + den - 1) // den

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
COMPILE_CACHE: dict = {}


def _compiled(key, jit_fn, *cute_args):
    fn = COMPILE_CACHE.get(key)
    if fn is None:
        fn = cute.compile(jit_fn, *cute_args)
        COMPILE_CACHE[key] = fn
    return fn


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


# e8m0 RCEIL helper (device): amax (f32 scalar) -> (rcp_f32, biased_uint8). Direct port of
# torchao/prototype/moe_training/kernels/mxfp8/cute_utils.py -- the canonical CuTeDSL mxfp8 kernel.
# Unlike our earlier inline-asm version this uses the typed NVVM cvt ops:
#   f32 -> e8m0 : nvvm.cvt_packfloat_f32(UE8M0x2, rnd=RP)   (RP == RCEIL, high input 0.0, sat NONE)
#   e8m0 -> f32 : nvvm.cvt_packfloat(UE8M0x2 -> BF16x2, rnd=RN) then bf16 -> f32  (supported path)
# and the reciprocal is a real e8m0->f32 cast (reciprocal_biased = 254 - biased in uint8), matching
# torchao `_reciprocal_scale` -- no manual <<23 exponent-shift and no byte-0/255 special cases (the
# hardware cvt handles them). `view_as`/`pack`/`unpack` are the scalar bitcast/pack helpers.
@dsl_user_op
def view_as(x, dtype, *, loc=None, ip=None):
    """Bitcast one scalar to another scalar of equal width."""
    assert type(x).width == dtype.width
    # bitcast wants a signed IR type even for unsigned CUTLASS types.
    dst_type = (
        T.i(dtype.width) if ir.IntegerType.isinstance(dtype.mlir_type) else dtype.mlir_type
    )
    return dtype(arith.bitcast(dst_type, x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def unpack(x, dtype, *, loc=None, ip=None):
    """Unpack an integer carrier into a tuple of scalar values."""
    x = cute.typing.as_numeric(x)
    carrier_dtype = type(x)
    assert ir.IntegerType.isinstance(carrier_dtype.mlir_type)
    assert carrier_dtype.width % dtype.width == 0
    num_lanes = carrier_dtype.width // dtype.width
    # integer vector lanes: vector<N x FP8> can crash the compiler (NVIDIA/cutlass#3342).
    lanes = llvm.bitcast(
        T.vector(num_lanes, T.i(dtype.width)), x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    return tuple(
        view_as(
            cute.typing.as_numeric(
                vector.extract(lanes, dynamic_position=[], static_position=[i], loc=loc, ip=ip)
            ),
            dtype, loc=loc, ip=ip,
        )
        for i in range(num_lanes)
    )


@dsl_user_op
def pack(*values, carrier=None, loc=None, ip=None):
    """Pack same-typed scalar values into an integer carrier."""
    assert len(values) > 0
    lane_dtype = type(values[0])
    assert all(type(value) is lane_dtype for value in values)
    lane_type = T.i(lane_dtype.width)
    lanes = vector.from_elements(
        T.vector(len(values), lane_type),
        tuple(arith.bitcast(lane_type, value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip) for value in values),
        loc=loc, ip=ip,
    )
    packed_width = len(values) * lane_dtype.width
    packed = llvm.bitcast(T.i(packed_width), lanes, loc=loc, ip=ip)
    if carrier is None:
        return cute.typing.as_numeric(packed)
    assert ir.IntegerType.isinstance(carrier.mlir_type)
    assert carrier.width == packed_width
    return carrier(packed)


@dsl_user_op
def _cvt_f32_to_ue8m0(x, *, rounding_mode, loc=None, ip=None):
    """Convert x to a single E8M0 value without saturation (high input 0.0)."""
    # this cutlass binding infers the result type (no leading result-type arg): (a, b, c, to, ...).
    packed = nvvm.cvt_packfloat_f32(
        cutlass.Float32(0.0).ir_value(loc=loc, ip=ip),
        cutlass.Float32(x).ir_value(loc=loc, ip=ip),
        cutlass.Int32(0).ir_value(loc=loc, ip=ip),
        nvvm.CVTPackFloatKind.UE8M0x2,
        rnd=rounding_mode,
        sat=nvvm.SaturationModeKind.NONE,
        loc=loc, ip=ip,
    )
    return unpack(cutlass.Int32(packed), cutlass.Float8E8M0FNU, loc=loc, ip=ip)[0]


@dsl_user_op
def _cvt_ue8m0_to_f32(x, *, loc=None, ip=None):
    """Convert a single E8M0 value to f32 through the supported BF16 path."""
    x_e8m0x2 = pack(x, cutlass.Float8E8M0FNU(0), loc=loc, ip=ip)
    x_u32 = llvm.zext(T.i32(), x_e8m0x2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    bf16x2_bits = nvvm.cvt_packfloat(
        x_u32,
        cutlass.Int32(0).ir_value(loc=loc, ip=ip),
        nvvm.CVTPackFloatKind.UE8M0x2,
        nvvm.CVTPackFloatKind.BF16x2,
        rnd=nvvm.FPRoundingMode.RN,
        sat=nvvm.SaturationModeKind.NONE,
        loc=loc, ip=ip,
    )
    low_bf16 = unpack(cutlass.Int32(bf16x2_bits), cutlass.BFloat16, loc=loc, ip=ip)[0]
    return low_bf16.to(cutlass.Float32)


@cute.jit
def _reciprocal_scale(scale_e8m0):
    scale_biased = view_as(scale_e8m0, cutlass.Uint8)
    reciprocal_biased = cutlass.Uint8(254) - scale_biased
    return _cvt_ue8m0_to_f32(view_as(reciprocal_biased, cutlass.Float8E8M0FNU))


@cute.jit
def _e8m0(amax):
    # torchao compute_scale_rceil: descale = amax * (1/448), RP-round to e8m0, then reciprocal.
    descale = amax * cutlass.Float32(1.0 / 448.0)
    scale_e8m0 = _cvt_f32_to_ue8m0(descale, rounding_mode=nvvm.FPRoundingMode.RP)
    rcp = _reciprocal_scale(scale_e8m0)
    return rcp, view_as(scale_e8m0, cutlass.Uint8)


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


@dsl_user_op
def _cvt_rs_satfinite_e4m3x4_f32(v0, v1, v2, v3, rbits, *, loc=None, ip=None):
    """Stochastically round four f32 values to four packed E4M3 bytes (Blackwell SM100).

    PTX writes its first source into the high byte of the b32 result. Reverse the source operands so
    the little-endian byte view is [e4m3(v0), e4m3(v1), e4m3(v2), e4m3(v3)].
    """
    args = [
        cutlass.Float32(v3).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v2).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v1).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v0).ir_value(loc=loc, ip=ip),
        cutlass.Uint32(rbits).ir_value(loc=loc, ip=ip),
    ]
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            args,
            "cvt.rs.satfinite.e4m3x4.f32 $0, {$1, $2, $3, $4}, $5;",
            "=r,f,f,f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


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
# mxfp8_swizzle_v2: TMA (bulk-tensor) load/store of the main data, mxfp8 dim-K numerics.
#
# Structured on fp8_deepseek_1x128_dim_m_v2 (the warp-specialized TMA kernel above) -- the same
# mbarrier arrive-and-expect-tx handshake and warp-0-gated G2S/S2G copies. The M tile stays 128 while
# N is selected from 32/64/128: small problems gain CTAs, while large problems retain the efficient
# 128x128 transfer. The calculation differs because mxfp8_swizzle reduces along K (a 1x32 block
# within a row), not down M like deepseek dim-M:
#   - a 128xN tile holds 128 x (N/32) scale groups. With 128 threads, each thread owns one row and
#     visits its N/32 contiguous 1x32 groups.
#   - scale is an e8m0 RCEIL byte (_e8m0), written through a logical layout that maps directly onto
#     the swizzled (nrb,ncb,32,16) grid.
#   - the output is NOT transposed (dim-K keeps (M,N)); sOutput has the same logical (TM,TN) shape
#     and the S2G box matches the selected tile, so no register->smem transpose is needed.
# TMA handles ragged edge tiles directly: G2S zero-fills out-of-bounds input lanes and S2G drops
# out-of-bounds qdata stores. The scale grid is rounded up to complete 128x4 swizzle atoms, and
# boundary CTAs explicitly write zero to its padded slots.
#
# PERF REGRESSION/WORKAROUND (nvidia-cutlass-dsl 4.6.x): 4.6 stopped vectorizing the scalar
# sInput/sOutput loops that 4.5.2 lowered to LDS.128/STS.128. That changed 16 LDS.128 + 8 STS.128
# into 128 LDS.U16 + 128 STS.U8, raised bank conflicts from 13.3M ld + 2.6M st to 126M ld + 60M st,
# and dropped 16384^2 throughput from ~72.6% peak (0.140ms) to ~14% (0.72ms). The explicit 128-bit
# CopyUniversal atoms below restore the original 16/8 vector instructions and ~72-73% peak on 4.6.
# A plain cute.autovec_copy did not work; the explicit copy width is what prevents scalarization.
# Independent bf16/fp8 K_SW128 views cut the remaining ld/st conflicts from 13.4M/2.7M to 5.0M/0.4M
# without changing the ~0.140ms runtime; their lifetimes are separated by the aliasing barrier.
# Scheduling adjacent N tiles together is the larger win: an N-major grid with N-oriented CTA
# clusters improves row-major DRAM locality. Row-owned work makes each thread's scale bytes
# contiguous in a packed store and cuts shared-load conflicts without increasing registers.
# Together with skipping the redundant scale memset: 0.1375 -> 0.1213 ms at 16384^2 on B200.
# All modes additionally have a compile-time stochastic specialization. They keep the same TMA
# load, reductions, scale stores, shared-memory output staging, and TMA stores; only each enabled
# qdata conversion changes to Philox plus Blackwell cvt.rs.e4m3x4.
_MXS_TM, _MXS_MAX_TN, _MXS_WARPS = 128, 128, 4
_MXS_THREADS = _MXS_WARPS * 32                       # 128, one thread per tile row
_MXS_MODE_DIM_K = 0
_MXS_MODE_DIM_M = 1
_MXS_MODE_DIM_KM = 2


def _mxfp8_swizzle_v2_tile_n(M, N):
    """Choose the measured B200 N tile from the padded 128x128 CTA count."""
    num_128_tiles = _ceil_div(M, _MXS_TM) * _ceil_div(N, _MXS_MAX_TN)
    if num_128_tiles <= 64:
        tile_n = 32
    elif num_128_tiles <= 512:
        tile_n = 64
    else:
        tile_n = 128

    # Do not compute padding merely to reach the selected width for very narrow matrices.
    if N <= 32:
        return 32
    if N <= 64:
        return min(tile_n, 64)
    return tile_n


@cute.jit
def _mxfp8_v2_quantize_stochastic_x32(
    values, rcp, flat_start, k0, k1, counter_base
):
    """Quantize one contiguous 32-value output run with two Philox counters."""
    scaled = values * rcp
    qwords = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Uint32)
    for half in cutlass.range_constexpr(2):
        ctr = counter_base + cutlass.Uint64(flat_start // 16 + half)
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
    scale_k_tv_layout: cute.Layout,
    scale_m_tv_layout: cute.Layout,
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
    ncb_k: cutlass.Constexpr,
    ncb_m: cutlass.Constexpr,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ragged: cutlass.Constexpr,
    mode: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    n_tile, m_tile, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    do_dim_k = mode != _MXS_MODE_DIM_M
    do_dim_m = mode != _MXS_MODE_DIM_K

    smem = utils.SmemAllocator()
    tma_bar_ptr = smem.allocate_array(cutlass.Int64, 1)
    input_storage = smem.allocate_array(
        cutlass.BFloat16, tile_m * tile_n, byte_alignment=1024
    )
    if cutlass.const_expr(do_dim_m):
        output_m_storage = smem.allocate_array(
            cutlass.Float8E4M3FN, tile_m * tile_n, byte_alignment=1024
        )

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
    gInput = cute.local_tile(input_tma_tensor, (tile_m, tile_n), (None, None))
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
            output_k_tma_tensor, (tile_m, tile_n), (None, None)
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
            output_m_tma_tensor, (tile_n, tile_m), (None, None)
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
        cute.copy(
            input_tma_atom,
            tInputgInput[(None, m_tile, n_tile)],
            tInputsInput,
            tma_bar_ptr=tma_bar_ptr,
        )
    cute.arch.mbarrier_wait(tma_bar_ptr, 0)

    smem_load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    smem_store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=128
    )
    if cutlass.const_expr(stochastic):
        # Both passes share the CTA-invariant key. Each thread derives counters from the flattened
        # position of the contiguous 32-value run in the output orientation it is producing.
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

    # Each dim-M thread owns one input column and processes every 32-row group. Its transposed
    # qdata uses separate shared memory, so the optional dim-K pass can still read sInput.
    if cutlass.const_expr(do_dim_m):
        row_blocks = tile_m // 32
        rScaleM = cute.make_rmem_tensor(row_blocks, cutlass.Uint8)
        for row_block in cutlass.range_constexpr(row_blocks):
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
                sr_output_row_m = n_tile * tile_n + tidx
                sr_output_col_m = m_tile * tile_m + row_block * 32
                flat_start_m = sr_output_row_m * M + sr_output_col_m
                rQM.store(
                    _mxfp8_v2_quantize_stochastic_x32(
                        vm, rcp_m, flat_start_m, k0, k1, counter_base
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

        scaleTileM = cute.local_tile(
            mScaleMLogical, (tile_n, row_blocks), (n_tile, m_tile)
        )
        tidfrgScaleM = cute.composition(scaleTileM, scale_m_tv_layout)
        thrScaleM = tidfrgScaleM[(tidx, None)]
        scale_m_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Uint8,
            num_bits_per_copy=row_blocks * 8,
        )
        use_full_tile_m = cutlass.const_expr(not ragged) or (
            ((m_tile + 1) * tile_m <= M) & ((n_tile + 1) * tile_n <= N)
        )
        if use_full_tile_m:
            cute.copy(scale_m_store_atom, rScaleM, thrScaleM)
        else:
            output_row_m = n_tile * tile_n + tidx
            rScaleMPadded = cute.make_rmem_tensor(row_blocks, cutlass.Uint8)
            rScaleMPadded.fill(0)
            if output_row_m < N:
                for row_block in cutlass.range_constexpr(row_blocks):
                    if m_tile * row_blocks + row_block < M // 32:
                        rScaleMPadded[row_block] = rScaleM[row_block]
            if output_row_m < _ceil_div(N, 128) * 128:
                cute.copy(scale_m_store_atom, rScaleMPadded, thrScaleM)

    # This is the original row-owned v2 dim-K pass. Modes dim_k and dim_km execute the same code;
    # the ownership layout changes for shorter fused tiles so all 128 threads remain useful.
    if cutlass.const_expr(do_dim_k):
        bpr = tile_n // 32
        row_owned_k = tile_m == _MXS_TM
        iters = bpr if cutlass.const_expr(row_owned_k) else tile_m // 32
        tidfrgInputK = cute.composition(sInput, data_k_tv_layout)
        tidfrgOutputK = cute.composition(sOutputK, data_k_tv_layout)
        thrInputGroupsK = tidfrgInputK[(tidx, None)]
        thrOutputGroupsK = tidfrgOutputK[(tidx, None)]
        scaleTileK = cute.local_tile(
            mScaleKLogical, (tile_m, bpr), (m_tile, n_tile)
        )
        tidfrgScaleK = cute.composition(scaleTileK, scale_k_tv_layout)
        thrScaleK = tidfrgScaleK[(tidx, None)]
        rScaleK = cute.make_rmem_tensor(iters, cutlass.Uint8)
        rQK = cute.make_rmem_tensor(
            cute.make_layout((32, iters), stride=(1, 32)),
            cutlass.Float8E4M3FN,
        )

        for it in cutlass.range_constexpr(iters):
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
            rcp_k, biased_k = _e8m0(amax_k)
            if cutlass.const_expr(stochastic):
                if cutlass.const_expr(row_owned_k):
                    sr_input_row_k = m_tile * tile_m + tidx
                    sr_group_col_k = n_tile * bpr + it
                else:
                    sr_input_row_k = m_tile * tile_m + tidx // bpr + it * 32
                    sr_group_col_k = n_tile * bpr + tidx % bpr
                flat_start_k = sr_input_row_k * N + sr_group_col_k * 32
                rQK[(None, it)].store(
                    _mxfp8_v2_quantize_stochastic_x32(
                        vk, rcp_k, flat_start_k, k0, k1, counter_base
                    )
                )
            else:
                # Keep the original dim-K RTNE conversion unchanged in its specialization.
                rQK[(None, it)].store(
                    (vk * rcp_k).to(cutlass.Float8E4M3FN)
                )
            rScaleK[it] = biased_k.to(rScaleK.element_type)

        use_full_tile_k = cutlass.const_expr(not ragged) or (
            ((m_tile + 1) * tile_m <= M) & ((n_tile + 1) * tile_n <= N)
        )
        if cutlass.const_expr(row_owned_k):
            scale_k_store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Uint8,
                num_bits_per_copy=iters * 8,
            )
            if use_full_tile_k:
                cute.copy(scale_k_store_atom, rScaleK, thrScaleK)
            else:
                input_row_k = m_tile * tile_m + tidx
                rScaleKPadded = cute.make_rmem_tensor(iters, cutlass.Uint8)
                rScaleKPadded.fill(0)
                if input_row_k < M:
                    for it in cutlass.range_constexpr(iters):
                        if n_tile * bpr + it < N // 32:
                            rScaleKPadded[it] = rScaleK[it]
                cute.copy(scale_k_store_atom, rScaleKPadded, thrScaleK)

            grid_n = _ceil_div(N, tile_n)
            covered_groups = grid_n * bpr
            if cutlass.const_expr(covered_groups < ncb_k * 4):
                if n_tile == grid_n - 1:
                    input_row_k = m_tile * tile_m + tidx
                    for offset in cutlass.range_constexpr(3):
                        col = covered_groups + offset
                        if cutlass.const_expr(col < ncb_k * 4):
                            mScaleKLogical[(input_row_k, col)] = cutlass.Uint8(0)
        else:
            # Short fused tiles distribute (row, 1x32 group) pairs across all threads. Their scale
            # slots are strided across rows, so store them individually instead of packing bytes.
            local_group_k = tidx % bpr
            local_row_k = tidx // bpr
            for it in cutlass.range_constexpr(iters):
                input_row_k = m_tile * tile_m + local_row_k + it * 32
                scale_col_k = n_tile * bpr + local_group_k
                if use_full_tile_k:
                    thrScaleK[it] = rScaleK[it]
                else:
                    scale_k = cutlass.Uint8(0)
                    if input_row_k < M:
                        if scale_col_k < N // 32:
                            scale_k = rScaleK[it]
                    thrScaleK[it] = scale_k

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
    if warp == 0:
        if cutlass.const_expr(do_dim_k):
            cute.copy(
                output_k_tma_atom,
                tOutputsK,
                tOutputgK[(None, m_tile, n_tile)],
            )
        if cutlass.const_expr(do_dim_m):
            cute.copy(
                output_m_tma_atom,
                tOutputsM,
                tOutputgM[(None, n_tile, m_tile)],
            )




@cute.jit
def mxfp8_swizzle_v2_jit(
    mInput,
    mOutput,
    mScale,
    mSeed,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ncb: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
    cluster_n: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
):
    padded_M = _ceil_div(M, _MXS_TM) * _MXS_TM
    padded_N = _ceil_div(N, tile_n) * tile_n
    bpr = tile_n // 32
    iters = bpr
    # Match the swizzle width to the selected tile. The logical 128xN shape is unchanged while the
    # layout-aware fragments below see the conflict-reduced physical mapping.
    if cutlass.const_expr(tile_n == 32):
        input_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW64
        output_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW32
    elif cutlass.const_expr(tile_n == 64):
        input_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW128
        output_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW64
    else:
        input_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW128
        output_smem_kind = tcgen05.SmemLayoutAtomKind.K_SW128
    input_smem_atom = tcgen05.make_smem_layout_atom(input_smem_kind, cutlass.BFloat16)
    input_smem_layout = cute.coalesce(
        cute.tile_to_shape(input_smem_atom, (_MXS_TM, tile_n), order=(0, 1)),
        target_profile=(1, 1),
    )
    # The output phase aliases the same allocation but uses its own dtype-appropriate swizzle. Its
    # lifetime starts only after the bf16-read barrier, so the two interpretations cannot overlap.
    output_smem_atom = tcgen05.make_smem_layout_atom(output_smem_kind, cutlass.Float8E4M3FN)
    output_smem_layout = cute.coalesce(
        cute.tile_to_shape(output_smem_atom, (_MXS_TM, tile_n), order=(0, 1)),
        target_profile=(1, 1),
    )
    # (thread, value) -> logical coordinate in the 128xN tile. Each thread owns one row.
    data_tv_layout = cute.make_layout(
        ((_MXS_THREADS,), (32, iters)),
        stride=((1,), (_MXS_TM, _MXS_TM * 32)),
    )
    # The corresponding logical scale tile gives each thread N/32 adjacent columns.
    scale_tv_layout = cute.make_layout(
        ((_MXS_THREADS,), (iters,)),
        stride=((1,), (_MXS_TM,)),
    )

    input_tma_atom, input_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mInput, input_smem_layout, (_MXS_TM, tile_n))
    output_tma_atom, output_tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mOutput, output_smem_layout, (_MXS_TM, tile_n))

    # Logical (row, 1x32-col-group) view over the physical (nrb,ncb,32,16) scale buffer. The
    # hierarchical modes encode the blocked/swizzled destination, removing address arithmetic from
    # the kernel's scale store.
    # Manual _swizzle_flat indexing uses 76 registers versus 80 for this clearer layout-based form.
    nrb = cute.size(mScale) // (ncb * 32 * 16)
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
        scale_tv_layout,
        scale_tv_layout,
        _MXS_TM,
        tile_n,
        ncb,
        ncb,
        M,
        N,
        M != padded_M or N != padded_N,
        _MXS_MODE_DIM_K,
        stochastic,
    ).launch(
        # Make the fast-changing grid dimension follow contiguous columns. Clustering those CTAs
        # further preserves that locality in hardware scheduling.
        grid=(padded_N // tile_n, padded_M // _MXS_TM, 1),
        block=(_MXS_THREADS, 1, 1),
        cluster=(cluster_n, 1, 1),
    )


def mxfp8_swizzle_v2(
    input: torch.Tensor,
    mode: str = "dim_k",
    key: torch.Tensor | None = None,
    rounding_mode: str = "rtne",
    **kwargs,
):
    assert mode in ("dim_k", "dim_m", "dim_km"), f"unsupported mode: {mode}"
    assert input.dim() == 2, "unsupported"
    assert input.is_contiguous(), "unsupported"
    assert input.dtype == torch.bfloat16, "v2 is bf16-only"
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
    assert M > 0 and N > 0, "v2 requires non-empty dimensions"

    if mode != "dim_k":
        assert M % 32 == 0, "v2 dim-M requires M % 32 == 0"
        assert N % 16 == 0, "v2 dim-M requires N % 16 == 0"
        if mode == "dim_km":
            assert N % 32 == 0, "v2 dim-K requires N % 32 == 0"

        tile_m, tile_n = (
            _MXDMT_SMALL_TILE
            if M * N <= 2048 * 2048
            else (_MXDKMT_LARGE_TILE if mode == "dim_km" else _MXDMT_LARGE_TILE)
        )
        nrb_m, ncb_m = _ceil_div(N, 128), _ceil_div(M // 32, 4)
        padded_M = ncb_m * 128
        padded_N = _ceil_div(N, tile_n) * tile_n
        grid_n = padded_N // tile_n
        grid_m = padded_M // tile_m
        cluster_n = 1 if mode == "dim_km" else (
            next(c for c in (16, 8, 4, 2, 1) if c <= grid_n and grid_n % c == 0)
            if grid_m * grid_n <= 512
            else 1
        )

        output_m = torch.empty(
            N, M, dtype=torch.float8_e4m3fn, device=input.device
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
        mScaleM = from_dlpack(scale_m, assumed_align=4)
        if mode == "dim_km":
            nrb_k, ncb_k = _ceil_div(M, 128), _ceil_div(N // 32, 4)
            output_k = torch.empty(
                M, N, dtype=torch.float8_e4m3fn, device=input.device
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
            mScaleK = from_dlpack(scale_k, assumed_align=4)
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
        fn = _compiled(
            (
                "mxfp8_swizzle_v2", mode, M, N, tile_m, tile_n, cluster_n,
                rounding_mode,
            ),
            _mxfp8_swizzle_v2_m_jit,
            mInput,
            mOutputM,
            mScaleM,
            mOutputK,
            mScaleK,
            mSeed,
            M,
            N,
            ncb_m,
            ncb_k,
            tile_m,
            tile_n,
            cluster_n,
            mode_id,
            stochastic,
        )
        fn(mInput, mOutputM, mScaleM, mOutputK, mScaleK, mSeed)
        scale_m = scale_m.view(nrb_m, ncb_m, 32, 16).view(
            torch.float8_e8m0fnu
        )
        if mode == "dim_km":
            scale_k = scale_k.view(nrb_k, ncb_k, 32, 16).view(
                torch.float8_e8m0fnu
            )
            return output_k, scale_k, output_m, scale_m
        return output_m, scale_m

    assert N % 32 == 0, "v2 requires K % 32 == 0"
    ngc = N // 32
    nrb, ncb = _ceil_div(M, 128), _ceil_div(ngc, 4)
    # Smaller N tiles expose more CTAs when the 128x128 grid cannot fill the B200. Once there are
    # enough 128x128 tiles, their lower TMA/setup overhead wins.
    tile_n = _mxfp8_swizzle_v2_tile_n(M, N)
    grid_n = _ceil_div(N, tile_n)
    # Preserve the tuned cluster sizes where possible while requiring exact grid divisibility.
    cluster_n = next(c for c in (16, 8, 4, 2, 1) if c <= ncb and grid_n % c == 0)
    output = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=input.device)
    # Every slot is written by the kernel, so zero-initialization would launch a redundant memset.
    scale = torch.empty(nrb * ncb * 32 * 16, dtype=torch.uint8, device=input.device)
    # TMA needs full layout/divisibility marking (leading dim contiguous, 16-elem aligned).
    mInput = (from_dlpack(input, assumed_align=16).mark_layout_dynamic(leading_dim=1)
              .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mOutput = (from_dlpack(output, assumed_align=16).mark_layout_dynamic(leading_dim=1)
               .mark_compact_shape_dynamic(mode=1, divisibility=16))
    mScale = from_dlpack(scale, assumed_align=4)
    # The RTNE specialization never reads mSeed, so reuse mScale as a dummy argument on that path.
    mSeed = from_dlpack(key.reshape(-1).view(torch.int64)) if stochastic else mScale
    fn = _compiled(
        ("mxfp8_swizzle_v2", M, N, tile_n, rounding_mode),
        mxfp8_swizzle_v2_jit,
        mInput,
        mOutput,
        mScale,
        mSeed,
        M,
        N,
        ncb,
        tile_n,
        cluster_n,
        stochastic,
    )
    fn(mInput, mOutput, mScale, mSeed)
    return output, scale.view(nrb, ncb, 32, 16).view(torch.float8_e8m0fnu)


MXFP8_SWIZZLE_V2 = QuantCastCuteRecipe.from_gold(
    Mxfp8SwizzleGold, cute_fn=mxfp8_swizzle_v2
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
def _mxfp8_swizzle_v2_m_jit(
    mInput: cute.Tensor,
    mOutputM: cute.Tensor,
    mScaleM: cute.Tensor,
    mOutputK: cute.Tensor,
    mScaleK: cute.Tensor,
    mSeed: cute.Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    ncb_m: cutlass.Constexpr,
    ncb_k: cutlass.Constexpr,
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
    cluster_n: cutlass.Constexpr,
    mode: cutlass.Constexpr,
    stochastic: cutlass.Constexpr,
):
    padded_M = _ceil_div(M, 128) * 128
    padded_N = _ceil_div(N, tile_n) * tile_n
    if cutlass.const_expr(mode == _MXS_MODE_DIM_KM):
        # Keep each 128-bit row vector intact while XORing row bits into the shared-memory bank
        # selection. This targets the dim-K phase's 16-way row-read conflicts.
        input_smem_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        input_smem_layout = cute.coalesce(
            cute.tile_to_shape(input_smem_atom, (tile_m, tile_n), order=(0, 1)),
            target_profile=(1, 1),
        )
    else:
        # Keep the kernel argument type uniform while retaining the original unswizzled dim-M
        # address mapping.
        input_smem_layout = cute.make_composed_layout(
            cute.make_swizzle(0, 0, 0),
            0,
            cute.make_layout((tile_m, tile_n), stride=(tile_n, 1)),
        )
    output_m_smem_layout = cute.make_composed_layout(
        cute.make_swizzle(0, 0, 0),
        0,
        cute.make_layout((tile_n, tile_m), stride=(tile_m, 1)),
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
        (tile_n, tile_m),
    )

    if cutlass.const_expr(mode == _MXS_MODE_DIM_KM):
        output_k_smem_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Float8E4M3FN
        )
        output_k_smem_layout = cute.coalesce(
            cute.tile_to_shape(
                output_k_smem_atom, (tile_m, tile_n), order=(0, 1)
            ),
            target_profile=(1, 1),
        )
        output_k_tma_atom, output_k_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mOutputK,
            output_k_smem_layout,
            (tile_m, tile_n),
        )
    else:
        # These arguments disappear with the constexpr dim-K branch.
        output_k_smem_layout = output_m_smem_layout
        output_k_tma_atom = output_m_tma_atom
        output_k_tma_tensor = output_m_tma_tensor

    nrb_m = cute.size(mScaleM) // (ncb_m * 32 * 16)
    scale_m_layout = cute.make_layout(
        ((32, 4, nrb_m), (4, ncb_m)),
        stride=((16, 4, ncb_m * 32 * 16), (1, 32 * 16)),
    )
    mScaleMLogical = cute.make_tensor(mScaleM.iterator, scale_m_layout)
    nrb_k = cute.size(mScaleK) // (ncb_k * 32 * 16)
    scale_k_layout = cute.make_layout(
        ((32, 4, nrb_k), (4, ncb_k)),
        stride=((16, 4, ncb_k * 32 * 16), (1, 32 * 16)),
    )
    mScaleKLogical = cute.make_tensor(mScaleK.iterator, scale_k_layout)
    bpr = tile_n // 32
    if cutlass.const_expr(tile_m == _MXS_TM):
        data_k_tv_layout = cute.make_layout(
            ((tile_m,), (32, bpr)),
            stride=((1,), (tile_m, tile_m * 32)),
        )
        scale_k_tv_layout = cute.make_layout(
            ((tile_m,), (bpr,)),
            stride=((1,), (tile_m,)),
        )
    else:
        # Flatten (1x32 group within a row, row within a 32-row stage) over the 128 threads,
        # then advance the value iteration through successive 32-row stages.
        row_blocks = tile_m // 32
        data_k_tv_layout = cute.make_layout(
            ((bpr, 32), (32, row_blocks)),
            stride=((tile_m * 32, 1), (tile_m, 32)),
        )
        scale_k_tv_layout = cute.make_layout(
            ((bpr, 32), (row_blocks,)),
            stride=((tile_m, 1), (32,)),
        )
    scale_m_tv_layout = cute.make_layout(
        ((tile_n,), (tile_m // 32,)),
        stride=((1,), (tile_n,)),
    )

    mxfp8_swizzle_v2_kernel(
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
        scale_k_tv_layout,
        scale_m_tv_layout,
        tile_m,
        tile_n,
        ncb_k,
        ncb_m,
        M,
        N,
        M != padded_M or N != padded_N,
        mode,
        stochastic,
    ).launch(
        # N-major scheduling keeps adjacent row-major input columns together.
        grid=(padded_N // tile_n, padded_M // tile_m, 1),
        block=(max(_MXS_THREADS, tile_n), 1, 1),
        cluster=(cluster_n, 1, 1),
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


ALL_RECIPES = [
    ("deepseek_1x128", FP8_DEEPSEEK_1X128),
    ("deepseek_1x128_dim_m", FP8_DEEPSEEK_1X128_DIM_M),
    ("deepseek_1x128_dim_m_v2", FP8_DEEPSEEK_1X128_DIM_M_V2),
    ("mxfp8_swizzle", MXFP8_SWIZZLE),
    ("mxfp8_swizzle_v2", MXFP8_SWIZZLE_V2),
    ("mxfp8_swizzle_sr_v2", MXFP8_SWIZZLE_SR_V2),
    ("mxfp8_swizzle_v3", MXFP8_SWIZZLE_V3),
    ("mxfp8_swizzle_v4", MXFP8_SWIZZLE_V4),
    ("mxfp8_swizzle_sr_v4", MXFP8_SWIZZLE_SR_V4),
    ("mxfp8_swizzle_v5", MXFP8_SWIZZLE_V5),
    ("mxfp8_dim_m_swizzle_v2", MXFP8_DIM_M_SWIZZLE_V2),
    ("mxfp8_dim_m_swizzle_sr_v2", MXFP8_DIM_M_SWIZZLE_SR_V2),
    ("mxfp8_dim_km_swizzle_v2", MXFP8_DIM_KM_SWIZZLE_V2),
    ("mxfp8_dim_km_swizzle_sr_v2", MXFP8_DIM_KM_SWIZZLE_SR_V2),
]
