"""Padding helper for flex_tile_map. fp4 e2m1 conversion / 4-bit packing helpers have
migrated to quant_cast_gold/utils.py (see quant_cast_gold/recipes.py's nvfp4 recipes).
"""

from typing import Tuple
import torch
import torch.nn.functional as F


def _pad_to_multiple(input: torch.Tensor, pad_to: Tuple[int, int]) -> torch.Tensor:
    """Round each dim of a 2D `input` UP to a multiple of `pad_to`, zero-padding the high edge.

    Zero padding lands on the bottom/right, so a real reduction block extended with zeros keeps
    its amax and padded quant matches unpadded (for the real region). A no-op if already aligned.
    """
    M, K = input.shape

    def ceil_to(v, m):
        return ((v + m - 1) // m) * m

    M2, K2 = ceil_to(M, pad_to[0]), ceil_to(K, pad_to[1])
    if (M2, K2) == (M, K):
        return input
    # F.pad arg order is (left, right, top, bottom); pad only the high edge.
    return F.pad(input, (0, K2 - K, 0, M2 - M))
