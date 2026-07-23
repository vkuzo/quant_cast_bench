"""PyTorch reference for "Four Over Six" (4/6) NVFP4 quantization.

Reference (simulation) implementation of the algorithm from "Four Over Six: More Accurate NVFP4
Quantization with Adaptive Block Scaling" (Cook et al., arXiv:2512.02010, MIT-Han-Lab, CC BY 4.0).

NVFP4 stores each value as FP4 E2M1 (grid +/-{0,0.5,1,1.5,2,3,4,6}, max 6) in blocks of 16, with a
per-block FP8 E4M3 scale and a per-tensor FP32 global scale. Standard NVFP4 scales every block so
its max maps to 6 -- but FP4's steps are coarse near the top (nothing between 66.6% and 100% of the
block max), so large near-maximal values quantize poorly. 4/6 quantizes each block *twice* -- once
scaling the max to 6, once to 4 (where the value 3 then represents 75% of the block max) -- and
keeps whichever has lower reconstruction error. The 4-vs-6 choice needs no side channel: it is
absorbed into the stored E4M3 block scale (a scale-4 block just stores a 50%-larger scale), which
is why it only works with E4M3 (not MXFP4's power-of-2 E8M0) scales. The global scale uses a
reduced M_fp8 = 256 (not 448) so the block holding the tensor max can still pick scale-4 without
overflowing E4M3 (256 * 6/4 = 384 <= 448).

This module reuses the repo's canonical E2M1/packing bit-math from quant_cast_gold.utils.

Usage: python experiments/four_over_six/test.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quant_cast_bench.quant_cast_gold.recipes import (  # noqa: E402
    E4M3_EPS,
    F4_E2M1_MAX,
    F8E4M3_MAX,
)
from quant_cast_bench.quant_cast_gold.utils import (  # noqa: E402
    f32_to_f4_unpacked,
    f4_unpacked_to_f32,
    pack_uint4,
    unpack_uint4,
)

_TINY = 1e-30  # divisor floor so all-zero blocks don't produce NaNs


def _round_to_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round fp32 values to the nearest FP4 E2M1 grid point (RNE, saturating to +/-6)."""
    return f4_unpacked_to_f32(f32_to_f4_unpacked(v.float().contiguous()))


def _block_error(dq: torch.Tensor, x: torch.Tensor, rule: str) -> torch.Tensor:
    """Per-block reconstruction error, shape (n_blocks, 1). rule in {mse, mae, minmax}."""
    d = dq - x
    if rule == "mse":
        return (d * d).mean(dim=1, keepdim=True)
    if rule == "mae":
        return d.abs().mean(dim=1, keepdim=True)
    if rule == "minmax":
        return d.abs().amax(dim=1, keepdim=True)
    raise ValueError(f"unknown selection rule: {rule!r}")


def _quantize(x: torch.Tensor, block_size: int, scale_targets, m_fp8: float, rule: str):
    """Block-scaled NVFP4 encode. Each block is quantized once per target in `scale_targets`
    (e.g. (6.0,) for baseline, (4.0, 6.0) for 4/6); the lowest-error candidate wins per block.

    Returns (codes, block_scales, global_scale, shape):
      codes         packed uint8, two FP4 E2M1 codes per byte
      block_scales  (n_blocks, 1) FP8 E4M3 per-block scale as fp32 (holds the 4-vs-6 choice)
      global_scale  scalar fp32 per-tensor scale
      shape         original tensor shape
    """
    assert x.numel() % block_size == 0, f"numel {x.numel()} not divisible by block_size {block_size}"
    xb = x.float().reshape(-1, block_size)                        # (n_blocks, block_size)
    outer = xb.abs().amax() / (F4_E2M1_MAX * m_fp8)               # scalar fp32 global scale
    outer = outer.clamp(min=_TINY)
    local_amax = xb.abs().amax(dim=1, keepdim=True)               # (n_blocks, 1)

    best_codes = best_inner = best_err = None
    for target in scale_targets:
        inner = ((local_amax / target) / outer).clamp(E4M3_EPS, F8E4M3_MAX)
        inner = inner.to(torch.float8_e4m3fn).to(torch.float32)   # cast through E4M3
        total = inner * outer                                     # effective per-element scale
        codes = f32_to_f4_unpacked(xb / total.clamp(min=_TINY))   # (n_blocks, block_size) uint8
        dq = f4_unpacked_to_f32(codes) * total                    # reconstruction of xb
        err = _block_error(dq, xb, rule)                          # (n_blocks, 1)
        if best_err is None:
            best_codes, best_inner, best_err = codes, inner, err
        else:
            pick = err < best_err                                 # per-block: this target is better
            best_codes = torch.where(pick, codes, best_codes)
            best_inner = torch.where(pick, inner, best_inner)
            best_err = torch.where(pick, err, best_err)

    return pack_uint4(best_codes), best_inner, outer, x.shape


def _dequantize(codes: torch.Tensor, block_scales: torch.Tensor, global_scale: torch.Tensor,
                shape: torch.Size) -> torch.Tensor:
    """Decode back to fp32: unpack FP4 codes, apply block + global scales, restore shape."""
    unpacked = unpack_uint4(codes)                                # (n_blocks, block_size) uint8
    dq = f4_unpacked_to_f32(unpacked) * block_scales * global_scale
    return dq.reshape(shape)


def nvfp4_quantize(x: torch.Tensor, block_size: int = 16):
    """Baseline NVFP4: every block scaled so its max maps to 6 (M_fp8 = 448).
    Returns (codes, block_scales, global_scale, shape)."""
    return _quantize(x, block_size, scale_targets=(6.0,), m_fp8=448.0, rule="mse")


def nvfp4_dequantize(codes: torch.Tensor, block_scales: torch.Tensor, global_scale: torch.Tensor,
                     shape: torch.Size) -> torch.Tensor:
    return _dequantize(codes, block_scales, global_scale, shape)


def four_over_six_quantize(x: torch.Tensor, block_size: int = 16, rule: str = "mse"):
    """4/6 NVFP4: per-block adaptive scale-to-4-or-6, lowest-error wins (M_fp8 = 256).
    Returns (codes, block_scales, global_scale, shape)."""
    return _quantize(x, block_size, scale_targets=(4.0, 6.0), m_fp8=256.0, rule=rule)


def four_over_six_dequantize(codes: torch.Tensor, block_scales: torch.Tensor,
                             global_scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return _dequantize(codes, block_scales, global_scale, shape)


def nvfp4_roundtrip(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Baseline NVFP4 encode->decode, returning the dequantized approximation of x."""
    return nvfp4_dequantize(*nvfp4_quantize(x, block_size))


def four_over_six_roundtrip(x: torch.Tensor, block_size: int = 16, rule: str = "mse") -> torch.Tensor:
    """4/6 NVFP4 encode->decode, returning the dequantized approximation of x."""
    return four_over_six_dequantize(*four_over_six_quantize(x, block_size, rule))
