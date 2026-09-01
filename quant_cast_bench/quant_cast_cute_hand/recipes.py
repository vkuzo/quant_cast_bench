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
    output = torch.empty_like(input) 
    input_cute = from_dlpack(input)
    output_cute = from_dlpack(output)
    add_v0_jit(input_cute, num, output_cute)
    return output


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
