import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from quant_cast_bench.quant_cast_cute_hand.utils import _ceil_div


@cute.kernel
def _ceil_div_i32_kernel(
    output: cute.Tensor,
    numerator: cutlass.Int32,
    denominator: cutlass.Int32,
) -> None:
    output[0] = _ceil_div(numerator, denominator)


@cute.jit
def _ceil_div_i32_jit(
    output: cute.Tensor,
    numerator: cutlass.Int32,
    denominator: cutlass.Int32,
) -> None:
    _ceil_div_i32_kernel(output, numerator, denominator).launch(
        grid=(1, 1, 1), block=(1, 1, 1)
    )


def run_i32_ceil_div(numerator: int, denominator: int) -> int:
    output = torch.empty(1, dtype=torch.int32, device="cuda")
    _ceil_div_i32_jit(
        from_dlpack(output, assumed_align=4),
        cutlass.Int32(numerator),
        cutlass.Int32(denominator),
    )
    return output.item()
