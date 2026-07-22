"""Triton kernel for a 2-D transpose: y = x.t().

Simple, correctness-first version. A 2-D grid of (BLOCK_M x BLOCK_N) tiles: each program loads a
tile of x, transposes it in-registers with `tl.trans`, and stores the transposed tile to y. The
load is coalesced along x's contiguous axis (N); the store is coalesced along y's contiguous axis
(M). Optimization comes later.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(
    x_ptr, y_ptr, M, N,
    sxm, sxn, sym, syn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # load a (BLOCK_M, BLOCK_N) tile of x
    x_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=x_mask)

    # y is (N, M): y[n, m] = x[m, n]. Store the transposed (BLOCK_N, BLOCK_M) tile.
    yt = tl.trans(x)
    y_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(y_ptr + offs_n[:, None] * sym + offs_m[None, :] * syn, yt, mask=y_mask)


def transpose_triton(x: torch.Tensor, BLOCK_M: int = 32, BLOCK_N: int = 32) -> torch.Tensor:
    assert x.dim() == 2
    M, N = x.shape
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _transpose_kernel[grid](
        x, y, M, N,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return y


# --- shared contract (test.py / benchmark.py rely on these names) ---
kernel_fn = transpose_triton


def reference_fn(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous()


def get_inputs():
    return [torch.randn(16384, 16384, dtype=torch.bfloat16, device="cuda")]
