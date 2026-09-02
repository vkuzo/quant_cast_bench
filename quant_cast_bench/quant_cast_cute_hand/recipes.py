# handwritten cute recipes, tracking learning CuTeDSL
#
# Started from FP8_DEEPSEEK_1X128, copied verbatim from quant_cast_cute/recipes.py; this is the
# playground where we iterate on it.

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch

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
def fp8_deepseek_1x128_kernel(input: cute.Tensor, output: cute.Tensor):
    # TODO(later): fill me out
    pass



@cute.jit
def fp8_deepseek_1x128_jit(input: cute.Tensor, output: cute.Tensor):
    # TODO(later): fill me out
    pass

def fp8_deepseek_1x128(input: torch.Tensor):
    # TODO(later): fill me out
    pass


FP8_DEEPSEEK_1X128 = QuantCastCuteRecipe.from_gold(
    Deepseek1x128Gold, cute_fn=fp8_deepseek_1x128
)


ALL_RECIPES = [
    ("deepseek_1x128", FP8_DEEPSEEK_1X128),
]
