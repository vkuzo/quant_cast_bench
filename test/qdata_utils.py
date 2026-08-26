"""Shared helpers for the quant-cast test suites: dtype-aware bit-exact comparison of quantized
outputs (both qdata and scale).
"""

import torch


def _as_bytes_or_fp32(t):
    # packed fp4 (float4_e2m1fn_x2) and e8m0 scales (float8_e8m0fnu) have no float cast we want to
    # rely on (fp4 has none at all; an e8m0 NaN would spuriously compare unequal after to(fp32)),
    # so compare their raw bytes via the uint8 view; everything else (fp8_e4m3, fp32) casts to fp32
    # losslessly.
    if t.dtype in (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu):
        return t.view(torch.uint8)
    return t.to(torch.float32)


def qdata_and_scale_equal(a, b):
    """Bit-exact, dtype-aware equality for a quantized qdata/scale tensor pair."""
    return torch.equal(_as_bytes_or_fp32(a), _as_bytes_or_fp32(b))


def mismatch_fraction(a, b):
    """Fraction of elements that differ under the same dtype-aware comparison as ``qdata_and_scale_equal``."""
    av, bv = _as_bytes_or_fp32(a), _as_bytes_or_fp32(b)
    return (av != bv).float().mean().item()
