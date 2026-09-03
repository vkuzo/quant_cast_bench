# handwritten cute recipes, tracking learning CuTeDSL
#
# Started from FP8_DEEPSEEK_1X128, copied verbatim from quant_cast_cute/recipes.py; this is the
# playground where we iterate on it.

import os

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch

# Gate debug output (host trace-time `print` + device `cute.printf`) behind an env var, read once
# at import. Gate with `cutlass.const_expr(_DEBUG)` inside kernels so that when off the tracer takes
# neither branch -- the printf ops are never emitted (no dead ops, no values kept live). Run with
# `CUTE_DEBUG=1 python -m ...` to enable.
_DEBUG = os.environ.get("CUTE_DEBUG", "0") == "1"

from quant_cast_bench.quant_cast_cute.recipes import QuantCastCuteRecipe
from quant_cast_bench.quant_cast_gold.recipes import Deepseek1x128Gold

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


ALL_RECIPES = [
    ("deepseek_1x128", FP8_DEEPSEEK_1X128),
]
