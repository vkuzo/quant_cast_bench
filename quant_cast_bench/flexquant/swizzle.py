# Ported from torchao `to_blocked`
# (`/home/dev/ao/torchao/prototype/mx_formats/utils.py`), but returning the 2D
# blocked form instead of the final flattened 1D buffer.
#
# Implements NVIDIA's cuBLAS "D-block scaling factors" layout (the 128x4 tile
# with an internal 32x4x4 swizzle), see
# https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
import torch


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked_2d(scale: torch.Tensor) -> torch.Tensor:
    """Swizzle a row-major block-scale tensor into the 2D blocked layout.

    Input is a ``(H, W)`` row-major scale (one element per quantization
    block). Output is ``(32 * ceil(H / 128), 16 * ceil(W / 4))``, the same
    bytes torchao's ``to_blocked`` produces, just kept 2D rather than
    flattened: ``to_blocked_2d(x).flatten()`` equals ``to_blocked(x)``.
    """
    rows, cols = scale.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = scale
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=scale.device, dtype=scale.dtype
        )
        padded[:rows, :cols] = scale

    # (n_row_blocks, 128, n_col_blocks, 4) -> tiles in (row_block, col_block)
    # order, each tile a 128x4 patch.
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    # Each 128x4 tile -> 32x16 swizzle: split 128 rows into 4 groups of 32,
    # move the 32 axis ahead of the 4 row-groups, fold (4 row-groups x 4 cols)
    # into the 16 inner axis. Leading dim iterates tiles in (row_block,
    # col_block) order -> this is exactly torchao's pre-flatten buffer.
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    # Fold the flat GPU buffer into 2D. Row-major order is preserved, so
    # `to_blocked_2d(x).flatten()` equals torchao's `to_blocked(x)`.
    return rearranged.reshape(n_row_blocks * 32, n_col_blocks * 16)
