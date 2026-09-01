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


@cute.kernel
def fp8_deepseek_1x128_kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.dynamic_expr(tidx == 0):
        cute.printf("hello from thread 0")
    elif cutlass.dynamic_expr(tidx == 1):
        cute.printf("hello from thread 1")


@cute.jit
def fp8_deepseek_1x128_jit(input: cute.Tensor, output: cute.Tensor):
    cute.printf("hello from host")
    fp8_deepseek_1x128_kernel().launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )

def fp8_deepseek_1x128(input: torch.Tensor):
    output = torch.empty_like(input) 
    fp8_deepseek_1x128_jit(from_dlpack(input), from_dlpack(output))


FP8_DEEPSEEK_1X128 = QuantCastCuteRecipe.from_gold(
    Deepseek1x128Gold, cute_fn=fp8_deepseek_1x128
)


ALL_RECIPES = [
    ("deepseek_1x128", FP8_DEEPSEEK_1X128),
]
