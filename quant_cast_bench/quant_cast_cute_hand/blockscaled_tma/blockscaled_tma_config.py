"""Semantic configuration shared by block-scaled TMA planning and kernels."""

from enum import IntEnum


class ScaleAlgo(IntEnum):
    """Compile-time algorithm used to derive and encode each block scale."""

    RCEIL_E8M0 = 0
    NVFP4_FP8_E4M3 = 1
