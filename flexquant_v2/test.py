"""Battle-test tile_map against a plain-PyTorch reference, recipe by recipe.

Comparison discipline mirrors flexquant v1 test.py: bit-exact `torch.equal` on
both qdata (compared as fp32) and scale.
"""

import pytest
import torch

from main import BLOCK_N, EPS, FP8_MAX, deepseek_1x128_f, tile_map

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (Helion GPU kernel)"
)


def _deepseek_1x128_reference(x):
    """Plain-PyTorch reference, from v1 recipes.py:60-70 (_deepseek_fp8_1_128_reference)."""
    m, n = x.shape
    n_blocks = n // BLOCK_N
    x_b = x.reshape(m, n_blocks, BLOCK_N)
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=EPS).to(torch.float32)
    scale = (amax / FP8_MAX).to(torch.float32)
    y = x_b.to(torch.float32) * (1.0 / scale)
    qdata = y.to(torch.float8_e4m3fn).reshape(m, n)
    return qdata, scale.squeeze(-1)


def test_deepseek_1x128_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata, scale = tile_map(deepseek_1x128_f, x, block_shape=(1, BLOCK_N))
    qdata_ref, scale_ref = _deepseek_1x128_reference(x)

    # shapes / dtypes
    assert qdata.shape == (256, 256)
    assert qdata.dtype == torch.float8_e4m3fn
    assert scale.shape == (256, 256 // BLOCK_N)
    assert scale.dtype == torch.float32

    # bit-exact vs reference (matches v1 discipline)
    assert torch.equal(qdata.to(torch.float32), qdata_ref.to(torch.float32))
    assert torch.equal(scale, scale_ref)
