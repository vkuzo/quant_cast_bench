"""Semantic configuration shared by block-scaled TMA planning and kernels."""

from enum import IntEnum


class ScaleAlgo(IntEnum):
    """Compile-time algorithm used to derive and encode each block scale."""

    RCEIL_E8M0 = 0
    NVFP4_FP8_E4M3 = 1


class RoundingVariant(IntEnum):
    """Compile-time rounding algorithm and Philox state-passing convention."""

    RTNE = 0
    STATELESS_SR = 1
    STATEFUL_SR_EAGER = 2
    STATEFUL_SR_CAPTURE = 3

    @property
    def is_stochastic(self) -> bool:
        return self != RoundingVariant.RTNE

    @property
    def is_stateful(self) -> bool:
        return self in (
            RoundingVariant.STATEFUL_SR_EAGER,
            RoundingVariant.STATEFUL_SR_CAPTURE,
        )
