"""Plain-PyTorch helpers shared by the mxfp8 MoE grouped/batched casts in `api.py`.

These are the low-level pieces `quantize_to_mxfp8_grouped` and `quantize_to_mxfp8_batched` compose:
the 1x32 e8m0 cast wrapper, token-group padding, and the NVIDIA blocked/swizzled tcgen05 scale
layouts (2D M-groups / 2D K-groups for token operands, per-expert 3D for weights). All are
pure-PyTorch ports of torchao's `kernels/mxfp8/quant.py`, substituting this repo's `_to_blocked_4d`
for torchao's `to_blocked` (bit-identical flat buffer) and `mxfp8_f` for the e8m0 cast primitive.
"""

import torch

from quant_cast_bench.quant_cast_gold.recipes import _to_blocked_4d, mxfp8_f

BLOCK_SIZE = 32


def _ceil_div(a, b):
    return (a + b - 1) // b


def quantize_2d_act(act: torch.Tensor):
    """Quantize a 2d activation `(total_M, K)` to mxfp8 with 1x32 blocks along K.

    Returns:
        act_fp8:   `(total_M, K)` float8_e4m3fn.
        act_scale: `(total_M, K // 32)` e8m0 (float8_e8m0fnu), naive (unswizzled) layout.
    """
    assert act.ndim == 2, "act must be 2D"
    act_fp8, act_scale = mxfp8_f(act)  # blocks the last dim (K)
    return act_fp8, act_scale


def _pad_token_groups(inputs: torch.Tensor, group_offsets: torch.Tensor, alignment: int = BLOCK_SIZE):
    """Pad each token group's rows up to a multiple of `alignment` with zeros so every group is
    block-aligned. Port of torchao's `torch_pad_token_groups`. Over-allocates the output to the
    upper bound `num_tokens + num_groups * alignment` (matching torchao's kernel).

    Returns:
        padded_tokens:        `(upper_bound, dim)` zero-padded tokens.
        padded_start_offsets: `(num_groups,)` int start row of each group after padding.
        padded_offsets:       `(num_groups,)` int32 end offsets after padding.
    """
    inputs = inputs.contiguous()
    num_tokens, dim = inputs.shape
    num_groups = group_offsets.shape[0]
    zero = torch.zeros(1, dtype=group_offsets.dtype, device=group_offsets.device)
    group_sizes = torch.diff(group_offsets, prepend=zero)
    padded_sizes = _ceil_div(group_sizes, alignment) * alignment
    padded_offsets = torch.cumsum(padded_sizes, 0, dtype=torch.int32)
    padded_start_offsets = padded_offsets - padded_sizes

    output_rows = _ceil_div(num_tokens + num_groups * alignment, alignment) * alignment
    padded_tokens = inputs.new_zeros(output_rows, dim)
    chunks = inputs.split(group_sizes.tolist(), dim=0)
    for chunk, padded_start in zip(chunks, padded_start_offsets.tolist()):
        padded_tokens[padded_start : padded_start + chunk.shape[0]] = chunk
    return padded_tokens, padded_start_offsets, padded_offsets


def _to_blocked_2d_m_groups(x_scales: torch.Tensor, group_offs: torch.Tensor) -> torch.Tensor:
    """Blocked scale layout for 2d scales grouped along rows (the token dim M). Port of torchao's
    `torch_to_blocked_2d_M_groups`: each group's scales are swizzled with `_to_blocked_4d` and written
    at a running row offset, each group padded to a multiple of 128 rows."""
    assert x_scales.ndim == 2, "x_scales must be 2D"
    total_M, scale_cols = x_scales.shape
    num_groups = group_offs.shape[0]
    blocked_scales = x_scales.new_zeros(total_M + num_groups * 128, scale_cols)
    group_start_idx = 0
    prev_start_row = 0
    for group_end_idx in group_offs.tolist():
        group_size = group_end_idx - group_start_idx
        if group_size == 0:
            continue
        group_blocked = _to_blocked_4d(x_scales[group_start_idx:group_end_idx]).reshape(-1, scale_cols)
        group_rows_padded = _ceil_div(group_size, 128) * 128
        blocked_scales[prev_start_row : prev_start_row + group_rows_padded] = group_blocked
        prev_start_row += group_blocked.shape[0]
        group_start_idx = group_end_idx
    return blocked_scales


def _to_blocked_per_group_3d(scales: torch.Tensor) -> torch.Tensor:
    """Blocked scale layout for a 3d weight `(E, rows, cols)`: swizzle each expert's 2d scale with
    `_to_blocked_4d` and flatten. Port of torchao's `torch_to_blocked_per_group_3d`. Returns
    `(E, flat_len)`."""
    assert scales.ndim == 3, "scales must be 3D (E, rows, cols)"
    per_expert = [_to_blocked_4d(scales[i]).reshape(-1) for i in range(scales.shape[0])]
    return torch.stack(per_expert, dim=0).contiguous()


def _to_blocked_2d_k_groups(x_scales: torch.Tensor, group_offs: torch.Tensor) -> torch.Tensor:
    """Blocked scale layout for 2d scales grouped along cols (the contraction dim, in scale space).
    Port of torchao's `torch_to_blocked_2d_K_groups`: rows padded to 128, each group's cols padded to
    4; per (128,4) scale tile the swizzled (512,) block is scattered into the flat output at a running
    column offset."""
    assert x_scales.ndim == 2, "x_scales must be 2D"
    M, total_K = x_scales.shape
    padded_M = _ceil_div(M, 128) * 128
    num_groups = group_offs.shape[0]
    blocked_scales = x_scales.new_zeros(padded_M, total_K + num_groups * 4)
    blocked_flat = blocked_scales.view(-1)
    stride_per_block = 128 * 4  # 512, a swizzled (128,4) scale tile
    num_row_blocks = _ceil_div(M, 128)
    group_start_idx = 0
    prev_start_col = 0
    for group_end_idx in group_offs.tolist():
        group_size = group_end_idx - group_start_idx
        if group_size == 0:
            continue
        cols_after_padding = _ceil_div(group_size, 4) * 4
        num_col_blocks = cols_after_padding // 4
        group_blocked = _to_blocked_4d(x_scales[:, group_start_idx:group_end_idx]).reshape(
            num_row_blocks, num_col_blocks, -1
        )
        base = prev_start_col * padded_M
        for row_block in range(num_row_blocks):
            for col_block in range(num_col_blocks):
                offset = (
                    base
                    + row_block * num_col_blocks * stride_per_block
                    + col_block * stride_per_block
                )
                blocked_flat[offset : offset + stride_per_block] = group_blocked[row_block, col_block]
        prev_start_col += cols_after_padding
        group_start_idx = group_end_idx
    return blocked_scales
