"""Memory-bandwidth benchmark for the handwritten CuTeDSL playground kernels.

Each kernel here is a memory-bound elementwise cast/op, so the signal we care about is achieved
memory bandwidth vs. the GPU's HBM ceiling (B200: 8 TB/s, H100 SXM5: 3.35 TB/s -- selected from the
device name). We build a bf16 (M, K) input, run the selected kernel, time it with
`do_bench_using_profiling`, and report GPU time + GB/s + % of peak.

    python -m quant_cast_bench.quant_cast_cute_hand.benchmark --kernel add_v0
    python -m quant_cast_bench.quant_cast_cute_hand.benchmark --kernel add_v0 --M 8192 --K 8192
"""

import os
import sys

import fire
import tabulate
import torch
from torch._inductor.utils import do_bench_using_profiling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quant_cast_bench.quant_cast_cute_hand.recipes import (
    add_v0, add_v1, add_v2, fp8_deepseek_1x128, fp8_deepseek_1x128_dim_m,
    fp8_deepseek_1x128_dim_m_v2, transpose_v0, transpose_v1,
)

# Peak HBM bandwidth per GPU family (GB/s), used for the "% of peak" column. Matched by substring
# against torch.cuda.get_device_name(0); H100 is the SXM5 HBM3 part (PCIe H100 is ~2 TB/s).
_PEAK_BW_GBPS = {
    "B200": 8000.0,  # 8 TB/s
    "H100": 3350.0,  # 3.35 TB/s
}


def _peak_bw_gbps(device_name):
    for family, bw in _PEAK_BW_GBPS.items():
        if family in device_name:
            return bw
    raise AssertionError(
        f"unsupported GPU {device_name!r}; known families: {sorted(_PEAK_BW_GBPS)}"
    )


def _bench_add_v0(M, K):
    # read input (M*K bf16) + write output (M*K bf16); a trivially memory-bound + 1 elementwise op.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float32, device="cuda")

    def run():
        return add_v0(x, 1.0)

    out = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor (e.g. a grid that
    # covers only one block): without this the timing is just launch overhead and the reported
    # bandwidth is fictional. Require the result to actually equal x + 1 across all elements.
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), x.float() + 1.0)
    bytes_per_iter = x.numel() * x.element_size() + out.numel() * out.element_size()
    return run, bytes_per_iter

def _bench_add_v1(M, K):
    # read input (M*K bf16) + write output (M*K bf16); a trivially memory-bound + 1 elementwise op.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float32, device="cuda")

    def run():
        return add_v1(x, 1.0)

    out = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor (e.g. a grid that
    # covers only one block): without this the timing is just launch overhead and the reported
    # bandwidth is fictional. Require the result to actually equal x + 1 across all elements.
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), x.float() + 1.0)
    bytes_per_iter = x.numel() * x.element_size() + out.numel() * out.element_size()
    return run, bytes_per_iter

def _bench_add_v2(M, K):
    # read input (M*K bf16) + write output (M*K bf16); a trivially memory-bound + 1 elementwise op.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float32, device="cuda")

    def run():
        return add_v2(x, 1.0)

    out = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor (e.g. a grid that
    # covers only one block): without this the timing is just launch overhead and the reported
    # bandwidth is fictional. Require the result to actually equal x + 1 across all elements.
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), x.float() + 1.0)
    bytes_per_iter = x.numel() * x.element_size() + out.numel() * out.element_size()
    return run, bytes_per_iter


def _deepseek_1x128_ref(x):
    # torch reference for the deepseek 1x128 quant-cast. Matches the kernel bit-for-bit, including
    # the scale step: `amax / 448.0` with a Python-float divisor on a CUDA tensor is lowered by
    # torch to multiply-by-reciprocal, and the kernel deliberately does the same (`amax * f32(1/448)`)
    # rather than an honest div.rn. Used as the "did the kernel actually run" guard below.
    M, K = x.shape
    x_b = x.reshape(M, K // 128, 128)
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    scale = (amax / 448.0).to(torch.float32)
    qdata = (x_b.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return qdata.reshape(M, K), scale.squeeze(-1)


def _bench_fp8_deepseek_1x128(M, K):
    # read input (M*K bf16) + write qdata (M*K fp8) + write scale (M*(K/128) fp32). Memory-bound
    # 1x128 blockwise quant-cast.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return fp8_deepseek_1x128(x)

    q, s = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor. The kernel is bit-exact
    # vs the torch reference (scale via reciprocal-multiply, matching torch's CUDA lowering), so
    # require exact equality on both outputs.
    torch.cuda.synchronize()
    q_ref, s_ref = _deepseek_1x128_ref(x)
    assert torch.equal(s, s_ref), "scale mismatch vs reference"
    assert torch.equal(q.float(), q_ref.float()), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()   # bf16 input read
        + q.numel() * q.element_size() # fp8 qdata write
        + s.numel() * s.element_size() # fp32 scale write
    )
    return run, bytes_per_iter


def _deepseek_1x128_dim_m_ref(x):
    # torch reference for the dim-M 1x128 quant-cast. Reduces 128-row blocks along dim-M and returns
    # transposed outputs (N,M) / (N,M//128), matching Deepseek1x128DimMGold. Scale via `amax / 448.0`
    # (torch lowers the python-float divide to multiply-by-reciprocal, which the kernel mirrors).
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    M, N = x.shape
    x_b = x.reshape(M // 128, 128, N)
    amax = x_b.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    scale = (amax / fp8_max).to(torch.float32)
    qdata = (x_b.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn).reshape(M, N)
    return qdata.t().contiguous(), scale.squeeze(1).t().contiguous()


def _bench_fp8_deepseek_1x128_dim_m(M, K):
    # read input (M*K bf16) + write qdata (K*M fp8) + write scale (K*(M/128) fp32). Memory-bound
    # 128x1 blockwise quant-cast that stages the tile through shared memory to transpose it, so the
    # dim-M reduction becomes the contiguous dim-K pattern and both outputs are written coalesced.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return fp8_deepseek_1x128_dim_m(x)

    q, s = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor. Bit-exact vs the torch
    # reference (reciprocal-multiply scale), so require exact equality on both outputs.
    torch.cuda.synchronize()
    q_ref, s_ref = _deepseek_1x128_dim_m_ref(x)
    assert torch.equal(s, s_ref), "scale mismatch vs reference"
    assert torch.equal(q.float(), q_ref.float()), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()   # bf16 input read
        + q.numel() * q.element_size() # fp8 qdata write
        + s.numel() * s.element_size() # fp32 scale write
    )
    return run, bytes_per_iter


def _bench_fp8_deepseek_1x128_dim_m_v2(M, K):
    # read input (M*K bf16) + write qdata (K*M fp8) + write scale (K*(M/128) fp32). Same dim-M 1x128
    # quant-cast as _bench_fp8_deepseek_1x128_dim_m, but the v2 kernel is the TMA warp-specialized
    # variant (128-row block staged through smem, transpose in the register->smem write). Same
    # transposed (N,M) outputs, so it shares the torch reference and bit-exact guard.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return fp8_deepseek_1x128_dim_m_v2(x)

    q, s = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor. Bit-exact vs the torch
    # reference (reciprocal-multiply scale), so require exact equality on both outputs.
    torch.cuda.synchronize()
    q_ref, s_ref = _deepseek_1x128_dim_m_ref(x)
    assert torch.equal(s, s_ref), "scale mismatch vs reference"
    assert torch.equal(q.float(), q_ref.float()), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()   # bf16 input read
        + q.numel() * q.element_size() # fp8 qdata write
        + s.numel() * s.element_size() # fp32 scale write
    )
    return run, bytes_per_iter


def _bench_transpose_v0(M, K):
    # read input (M*K bf16) + write transposed output (K*M bf16); a memory-bound 2D transpose.
    # v0 is the naive path: coalesced vectorized read, scattered (strided) column write.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return transpose_v0(x)

    out = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor. A transpose is a pure
    # data movement, so require exact equality with the reference transpose across all elements.
    torch.cuda.synchronize()
    assert torch.equal(out, x.t().contiguous()), "transpose mismatch vs reference"
    bytes_per_iter = x.numel() * x.element_size() + out.numel() * out.element_size()
    return run, bytes_per_iter


def _bench_transpose_v1(M, K):
    # read input (M*K bf16) + write transposed output (K*M bf16); a memory-bound 2D transpose.
    # v1 stages the tile through shared memory so BOTH the gmem read and gmem write are coalesced
    # (the transpose becomes a strided smem read instead of a scattered gmem write).
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return transpose_v1(x)

    out = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor. A transpose is a pure
    # data movement, so require exact equality with the reference transpose across all elements.
    torch.cuda.synchronize()
    assert torch.equal(out, x.t().contiguous()), "transpose mismatch vs reference"
    bytes_per_iter = x.numel() * x.element_size() + out.numel() * out.element_size()
    return run, bytes_per_iter


# name -> builder returning (run_fn, bytes_per_iter). Add new playground kernels here.
_KERNELS = {
    "add_v0": _bench_add_v0,
    "add_v1": _bench_add_v1,
    "add_v2": _bench_add_v2,
    "fp8_deepseek_1x128": _bench_fp8_deepseek_1x128,
    "fp8_deepseek_1x128_dim_m": _bench_fp8_deepseek_1x128_dim_m,
    "fp8_deepseek_1x128_dim_m_v2": _bench_fp8_deepseek_1x128_dim_m_v2,
    "transpose_v0": _bench_transpose_v0,
    "transpose_v1": _bench_transpose_v1,
}


def main(
    kernel: str = "add_v0",
    M: int = 16384,
    K: int = 16384,
):
    """Benchmark one handwritten CuTeDSL kernel and print GPU time + achieved memory bandwidth."""
    device_name = torch.cuda.get_device_name(0)
    peak_bw = _peak_bw_gbps(device_name)

    if kernel not in _KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; have {sorted(_KERNELS)}")

    run, bytes_per_iter = _KERNELS[kernel](M, K)

    # warm up so first-call costs (compile, autotune, allocator) don't leak into the timing.
    for _ in range(2):
        run()
    torch.cuda.synchronize()

    gpu_time_ms = do_bench_using_profiling(run)
    gbps = bytes_per_iter / (gpu_time_ms * 1e-3) / 1e9
    pct_peak = gbps / peak_bw * 100

    print(f"kernel: {kernel}  shape: ({M}, {K})  dtype: bfloat16")
    print(f"device: {device_name} (peak {peak_bw / 1000:.2f} TB/s)")
    print(
        tabulate.tabulate(
            [[kernel, f"{gpu_time_ms:.4f}", f"{gbps:.1f}", f"{pct_peak:.1f}%"]],
            headers=["kernel", "gpu_time_ms", "gbps", "pct_peak"],
            colalign=("left", "right", "right", "right"),
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
