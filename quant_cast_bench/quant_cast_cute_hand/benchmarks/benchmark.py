"""Memory-bandwidth benchmark for the handwritten CuTeDSL playground kernels.

Each kernel here is a memory-bound elementwise cast/op, so the signal we care about is achieved
memory bandwidth vs. the GPU's HBM ceiling (B200: 8 TB/s, H100 SXM5: 3.35 TB/s -- selected from the
device name). We build an (M, K) input of the selected dtype, run the selected kernel, time it with
`do_bench_using_profiling`, and report GPU time + GB/s + % of peak.

    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark --kernel add_v0
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark --kernel add_v0 --M 8192 --K 8192
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark --kernel add_v0 \
        --M 2048,4096,8192 --K 2048,4096,8192
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark --kernel add_v0 \
        --M 2048,4096 --K 2048,4096 --output_metrics gpu_time_ms,tb_s,pct_peak
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark \
        --kernel mxfp8_swizzle_v2,mxfp8_swizzle_v4 --M 2048,4096 --K 2048,4096
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark --kernel add_v0 \
        --M 2048,4096 --K 8192,16384 --mk_mode pair
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark \
        --kernel mxfp8_swizzle_v2 --shapes_for_model gpt-oss-120b
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark \
        --kernel mxfp8_swizzle_v2 --dtype float16
    python -m quant_cast_bench.quant_cast_cute_hand.benchmarks.benchmark --kernel add_v0 --csv_output results.csv
"""

import csv
import os
from pathlib import Path
import sys

# Suppress Kineto's profiler_start/profiler_stop USDT messages before PyTorch initializes it.
os.environ["KINETO_LOG_LEVEL"] = "6"

import fire
import tabulate
import torch
import torch.func._random as prng
from torch._inductor.utils import do_bench_using_profiling

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from quant_cast_bench.quant_cast_cute_hand.blockscaled_tma.blockscaled_tma_impl import (
    mxfp4,
    mxfp4_dim_km_swizzle_v2,
    mxfp4_dim_m_swizzle_v2,
    mxfp4_swizzle_v2,
    mxfp8,
    mxfp8_32x32_swizzle_v2,
    mxfp8_swizzle_v2,
    nvfp4,
    nvfp4_swizzle_16x16_tma,
)
from quant_cast_bench.quant_cast_cute_hand.recipes import (
    add_v0, add_v1, add_v2, fp8_deepseek_1x128, fp8_deepseek_1x128_dim_m,
    fp8_deepseek_1x128_dim_m_v2, mxfp8_swizzle,
    mxfp8_swizzle_v3, mxfp8_swizzle_v4, mxfp8_swizzle_v5,
    nvfp4_dim_km_swizzle_tma, nvfp4_dim_m_rht_swizzle_pipelined,
    nvfp4_dim_m_swizzle_rht_sr_pipelined,
    nvfp4_dim_m_swizzle_tma, nvfp4_swizzle_direct, nvfp4_swizzle_tma,
    nvfp4_swizzle_dim_k_dim_m_rht_pipelined,
    nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined,
    transpose_v0, transpose_v1,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    mxfp4_dim_km_swizzle_f, mxfp4_dim_m_swizzle_f, mxfp4_f, mxfp4_swizzle_f,
    mxfp8_32x32_swizzle_f, mxfp8_dim_km_swizzle_f, mxfp8_dim_km_swizzle_sr_f,
    mxfp8_dim_m_swizzle_f, mxfp8_dim_m_swizzle_sr_f, mxfp8_f, mxfp8_swizzle_f,
    mxfp8_swizzle_sr_f, hadamard_rht_fp32_f,
    nvfp4_gs_16x16_swizzle_f, nvfp4_gs_f, nvfp4_gs_scale,
    nvfp4_gs_swizzle_dim_km_f, nvfp4_gs_swizzle_dim_m_f,
    nvfp4_gs_swizzle_f,
    Nvfp4GsDimMSwizzleRHTSRGold, Nvfp4GsSwizzleDimMRHTGold,
    Nvfp4GsSwizzle_DimK_DimMRHT_Gold,
    Nvfp4GsSwizzle_DimKSR_DimMRHTSR_Gold,
)
from quant_cast_bench.quant_cast_cute_hand.benchmarks.shape_utils import (
    gpt_oss_120b_m8192_tp8_ep8,
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


def _bench_mxfp8_swizzle(M, K):
    # read input (M*K bf16) + write qdata (M*K fp8) + write scale ((M/128)*(K/32/4)*32*16 e8m0
    # bytes). Memory-bound 1x32 blockwise mxfp8 quant-cast whose e8m0 scale is stored in the NVIDIA
    # 128x4 -> 32x16 swizzled layout. Requires cuda capability 10.0 (Blackwell-only scale cvt).
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return mxfp8_swizzle(x)

    q, s = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor. The kernel is bit-exact
    # vs the gold reference (hardware `cvt.rp.ue8m0x2.f32` matches the gold's software e8m0 RCEIL),
    # so require exact equality on qdata and on the raw scale bytes.
    torch.cuda.synchronize()
    q_ref, s_ref = mxfp8_swizzle_f(x)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale mismatch vs reference"
    assert torch.equal(q.float(), q_ref.float()), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()   # bf16 input read
        + q.numel() * q.element_size() # fp8 qdata write
        + s.numel() * s.element_size() # e8m0 (1-byte) scale write
    )
    return run, bytes_per_iter


def _bench_mxfp8_swizzle_v2(M, K, dtype):
    # Same 1x32 blockwise mxfp8 quant-cast + swizzled e8m0 scale as _bench_mxfp8_swizzle, but the v2
    # kernel uses TMA (bulk-tensor) for the main-data load/store on a 128x128 tile (dim-K, no
    # transpose). Same outputs, so it shares the gold reference and bit-exact guard. Requires cuda
    # capability 10.0 (Blackwell-only scale cvt). Needs M%128==0 and K%128==0 for the TMA tile.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")

    def run():
        return mxfp8_swizzle_v2(x)

    q, s = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor. Bit-exact vs the gold
    # reference (hardware e8m0 RCEIL cvt), so require exact equality on qdata and raw scale bytes.
    torch.cuda.synchronize()
    q_ref, s_ref = mxfp8_swizzle_f(x)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale mismatch vs reference"
    assert torch.equal(q.float(), q_ref.float()), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()   # 16-bit input read
        + q.numel() * q.element_size() # fp8 qdata write
        + s.numel() * s.element_size() # e8m0 (1-byte) scale write
    )
    return run, bytes_per_iter


def _bench_mxfp8(M, K, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")

    def run():
        return mxfp8(x)

    qdata, scale = run()
    torch.cuda.synchronize()
    qdata_ref, scale_ref = mxfp8_f(x)
    assert torch.equal(
        qdata.view(torch.uint8), qdata_ref.view(torch.uint8)
    ), "qdata mismatch vs reference"
    assert torch.equal(
        scale.view(torch.uint8), scale_ref.view(torch.uint8)
    ), "scale mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()
        + qdata.numel() * qdata.element_size()
        + scale.numel() * scale.element_size()
    )
    return run, bytes_per_iter


def _bench_mxfp8_32x32_swizzle_v2(M, K, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")

    def run():
        return mxfp8_32x32_swizzle_v2(x)

    q, s = run()
    torch.cuda.synchronize()
    q_ref, s_ref = mxfp8_32x32_swizzle_f(x)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale mismatch vs reference"
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()
        + q.numel() * q.element_size()
        + s.numel() * s.element_size()
    )
    return run, bytes_per_iter


def _bench_mxfp8_swizzle_v3(M, K):
    # Same 1x32 blockwise mxfp8 quant-cast + swizzled e8m0 scale as _bench_mxfp8_swizzle, but the v3
    # kernel uses a 2-D 32x128 tile + a 16-elem/thread aligned uint32-word load (2x LDG.128,
    # register bf16->f32 unpack) and a single 2-lane shuffle. Same outputs, so it shares the gold
    # reference and bit-exact guard. Requires cuda capability 10.0 (Blackwell-only scale cvt). Needs
    # M%128==0 and K%128==0. v3 is bf16-only (the word load reinterprets the input as uint32).
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return mxfp8_swizzle_v3(x)

    q, s = run()
    # Guard against a kernel that "runs" but doesn't touch the whole tensor. Bit-exact vs the gold
    # reference (hardware e8m0 RCEIL cvt), so require exact equality on qdata and raw scale bytes.
    torch.cuda.synchronize()
    q_ref, s_ref = mxfp8_swizzle_f(x)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale mismatch vs reference"
    assert torch.equal(q.float(), q_ref.float()), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()   # bf16 input read
        + q.numel() * q.element_size() # fp8 qdata write
        + s.numel() * s.element_size() # e8m0 (1-byte) scale write
    )
    return run, bytes_per_iter


def _bench_mxfp8_swizzle_v4(M, K):
    # Same 2-D 32x128 tile + 16 bf16/thread as v3, but the load is two plain 8-elem fragment .load()s
    # (each an LDG.128) concat'd in registers -- no uint32-word path. Same outputs (bit-exact vs v3 and
    # the gold), so it shares the gold reference and bit-exact guard. Requires cuda capability 10.0.
    # K must be divisible by 32; ragged M and partial 128-column tiles are supported. bf16-only.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return mxfp8_swizzle_v4(x)

    q, s = run()
    torch.cuda.synchronize()
    q_ref, s_ref = mxfp8_swizzle_f(x)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale mismatch vs reference"
    assert torch.equal(q.float(), q_ref.float()), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()   # bf16 input read
        + q.numel() * q.element_size() # fp8 qdata write
        + s.numel() * s.element_size() # e8m0 (1-byte) scale write
    )
    return run, bytes_per_iter


def _bench_mxfp8_swizzle_sr_v2(M, K, dtype):
    # v2's existing TMA dim-K kernel with only its qdata conversion specialized to Philox-backed
    # Blackwell cvt.rs.e4m3x4 stochastic rounding.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    key = prng.key(0, device=x.device)

    def run():
        return mxfp8_swizzle_v2(x, key=key, rounding_mode="stochastic")

    q, s = run()
    torch.cuda.synchronize()
    q_ref, s_ref = mxfp8_swizzle_sr_f(x, key)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale mismatch vs reference"
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()
        + q.numel() * q.element_size()
        + s.numel() * s.element_size()
    )
    return run, bytes_per_iter


def _bench_mxfp8_swizzle_sr_v4(M, K):
    # Same v4 kernel and launch geometry, with its compile-time stochastic specialization. Each
    # thread generates one Philox counter and uses four cvt.rs.e4m3x4 instructions for its 16 values.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    key = prng.key(0, device=x.device)

    def run():
        return mxfp8_swizzle_v4(x, key, rounding_mode="stochastic")

    q, s = run()
    torch.cuda.synchronize()
    q_ref, s_ref = mxfp8_swizzle_sr_f(x, key)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale mismatch vs reference"
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()
        + q.numel() * q.element_size()
        + s.numel() * s.element_size()
    )
    return run, bytes_per_iter


def _bench_mxfp8_swizzle_v5(M, K):
    # Best of v1 and v4: v1's flat 1-D grid + v4's 16-elem/thread load (two LDG.128 concat'd in
    # registers) and single STG.128 store. Same outputs (bit-exact vs v1 and the gold), so it shares
    # the gold reference and bit-exact guard. Requires cuda capability 10.0. K must be divisible by
    # 32; ragged M and partial 128-column tiles are supported. bf16-only.
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        return mxfp8_swizzle_v5(x)

    q, s = run()
    torch.cuda.synchronize()
    q_ref, s_ref = mxfp8_swizzle_f(x)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale mismatch vs reference"
    assert torch.equal(q.float(), q_ref.float()), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()   # bf16 input read
        + q.numel() * q.element_size() # fp8 qdata write
        + s.numel() * s.element_size() # e8m0 (1-byte) scale write
    )
    return run, bytes_per_iter


def _bench_mxfp8_dim_m_swizzle_impl(M, K, dtype, kernel_fn):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")

    def run():
        return kernel_fn(x)

    q, s = run()
    torch.cuda.synchronize()
    q_ref, s_ref = mxfp8_dim_m_swizzle_f(x)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8)), "scale mismatch vs reference"
    assert torch.equal(q.float(), q_ref.float()), "qdata mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()
        + q.numel() * q.element_size()
        + s.numel() * s.element_size()
    )
    return run, bytes_per_iter


def _bench_mxfp8_dim_m_swizzle_v2(M, K, dtype):
    return _bench_mxfp8_dim_m_swizzle_impl(
        M, K, dtype, lambda x: mxfp8_swizzle_v2(x, quant_orientation="dim_m")
    )


def _bench_mxfp8_dim_m_swizzle_sr_v2(M, K, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    key = prng.key(0, device=x.device)

    def run():
        return mxfp8_swizzle_v2(
            x, quant_orientation="dim_m", key=key, rounding_mode="stochastic"
        )

    outputs = run()
    torch.cuda.synchronize()
    ref_outputs = mxfp8_dim_m_swizzle_sr_f(x, key)
    for output, ref_output in zip(outputs, ref_outputs):
        assert torch.equal(
            output.view(torch.uint8), ref_output.view(torch.uint8)
        ), "output mismatch vs reference"
    bytes_per_iter = x.numel() * x.element_size() + sum(
        output.numel() * output.element_size() for output in outputs
    )
    return run, bytes_per_iter


def _bench_mxfp8_dim_km_swizzle_impl(M, K, dtype, kernel_fn):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")

    def run():
        return kernel_fn(x)

    outputs = run()
    torch.cuda.synchronize()
    ref_outputs = mxfp8_dim_km_swizzle_f(x)
    for output, ref_output in zip(outputs, ref_outputs):
        assert torch.equal(
            output.view(torch.uint8), ref_output.view(torch.uint8)
        ), "output mismatch vs reference"
    bytes_per_iter = x.numel() * x.element_size() + sum(
        output.numel() * output.element_size() for output in outputs
    )
    return run, bytes_per_iter


def _bench_mxfp8_dim_km_swizzle_v2(M, K, dtype):
    return _bench_mxfp8_dim_km_swizzle_impl(
        M, K, dtype, lambda x: mxfp8_swizzle_v2(x, quant_orientation="dim_km")
    )


def _bench_mxfp4_swizzle_impl(M, K, dtype, kernel_fn, reference_fn):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")

    def run():
        return kernel_fn(x)

    outputs = run()
    torch.cuda.synchronize()
    ref_outputs = reference_fn(x)
    for output, ref_output in zip(outputs, ref_outputs):
        assert torch.equal(
            output.view(torch.uint8), ref_output.view(torch.uint8)
        ), "output mismatch vs reference"
    bytes_per_iter = x.numel() * x.element_size() + sum(
        output.numel() * output.element_size() for output in outputs
    )
    return run, bytes_per_iter


def _bench_mxfp4_swizzle_v2(M, K, dtype):
    return _bench_mxfp4_swizzle_impl(
        M, K, dtype, mxfp4_swizzle_v2, mxfp4_swizzle_f
    )


def _bench_mxfp4(M, K, dtype):
    return _bench_mxfp4_swizzle_impl(M, K, dtype, mxfp4, mxfp4_f)


def _bench_mxfp4_dim_m_swizzle_v2(M, K, dtype):
    return _bench_mxfp4_swizzle_impl(
        M, K, dtype, mxfp4_dim_m_swizzle_v2, mxfp4_dim_m_swizzle_f
    )


def _bench_mxfp4_dim_km_swizzle_v2(M, K, dtype):
    return _bench_mxfp4_swizzle_impl(
        M, K, dtype, mxfp4_dim_km_swizzle_v2, mxfp4_dim_km_swizzle_f
    )


def _bench_mxfp8_dim_km_swizzle_sr_v2(M, K, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    key = prng.key(0, device=x.device)

    def run():
        return mxfp8_swizzle_v2(
            x, quant_orientation="dim_km", key=key, rounding_mode="stochastic"
        )

    outputs = run()
    torch.cuda.synchronize()
    ref_outputs = mxfp8_dim_km_swizzle_sr_f(x, key)
    for output, ref_output in zip(outputs, ref_outputs):
        assert torch.equal(
            output.view(torch.uint8), ref_output.view(torch.uint8)
        ), "output mismatch vs reference"
    bytes_per_iter = x.numel() * x.element_size() + sum(
        output.numel() * output.element_size() for output in outputs
    )
    return run, bytes_per_iter


def _bench_nvfp4_swizzle_impl(
    M,
    K,
    kernel_fn,
    dtype=torch.bfloat16,
    reference_fn=nvfp4_gs_swizzle_f,
):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    outer_scale = nvfp4_gs_scale(x).reciprocal()

    def run():
        return kernel_fn(x, outer_scale)

    qdata, scale = run()
    torch.cuda.synchronize()
    qdata_ref, scale_ref = reference_fn(x, outer_scale)
    assert torch.equal(
        qdata.view(torch.uint8), qdata_ref.view(torch.uint8)
    ), "qdata mismatch vs reference"
    assert torch.equal(
        scale.view(torch.uint8), scale_ref.view(torch.uint8)
    ), "scale mismatch vs reference"
    bytes_per_iter = (
        x.numel() * x.element_size()
        + qdata.numel() * qdata.element_size()
        + scale.numel() * scale.element_size()
    )
    return run, bytes_per_iter


def _bench_nvfp4_swizzle_direct(M, K):
    return _bench_nvfp4_swizzle_impl(M, K, nvfp4_swizzle_direct)


def _bench_nvfp4_swizzle_tma(M, K, dtype=torch.bfloat16):
    return _bench_nvfp4_swizzle_impl(M, K, nvfp4_swizzle_tma, dtype)


def _bench_nvfp4_swizzle_16x16_tma(M, K, dtype=torch.bfloat16):
    return _bench_nvfp4_swizzle_impl(
        M,
        K,
        nvfp4_swizzle_16x16_tma,
        dtype,
        reference_fn=nvfp4_gs_16x16_swizzle_f,
    )


def _bench_nvfp4(M, K, dtype=torch.bfloat16):
    return _bench_nvfp4_swizzle_impl(
        M, K, nvfp4, dtype, reference_fn=nvfp4_gs_f
    )


def _bench_nvfp4_dim_m_swizzle_tma(M, K, dtype=torch.bfloat16):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    outer_scale = nvfp4_gs_scale(x).reciprocal()

    def run():
        return nvfp4_dim_m_swizzle_tma(x, outer_scale)

    outputs = run()
    torch.cuda.synchronize()
    references = nvfp4_gs_swizzle_dim_m_f(x, outer_scale)
    for output, reference in zip(outputs, references):
        assert torch.equal(output.view(torch.uint8), reference.view(torch.uint8))
    bytes_per_iter = x.numel() * x.element_size() + sum(
        output.numel() * output.element_size() for output in outputs
    )
    return run, bytes_per_iter


def _bench_nvfp4_dim_km_swizzle_tma(M, K, dtype=torch.bfloat16):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    outer_scale = nvfp4_gs_scale(x).reciprocal()

    def run():
        return nvfp4_dim_km_swizzle_tma(x, outer_scale, outer_scale)

    outputs = run()
    torch.cuda.synchronize()
    references = nvfp4_gs_swizzle_dim_km_f(x, outer_scale, outer_scale)
    for output, reference in zip(outputs, references):
        assert torch.equal(output.view(torch.uint8), reference.view(torch.uint8))
    bytes_per_iter = x.numel() * x.element_size() + sum(
        output.numel() * output.element_size() for output in outputs
    )
    return run, bytes_per_iter


def _bench_nvfp4_dim_m_rht(M, K, *, stochastic):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rht_sign = torch.tensor([1, -1] * 8, device=x.device, dtype=x.dtype)
    (x_t_rht,) = hadamard_rht_fp32_f(x.t().contiguous(), rht_sign)
    outer_scale = nvfp4_gs_scale(x_t_rht).reciprocal()

    if stochastic:
        key = prng.key(0, device=x.device)

        def run():
            return nvfp4_dim_m_swizzle_rht_sr_pipelined(
                x, outer_scale, rht_sign, key
            )

        gold = Nvfp4GsDimMSwizzleRHTSRGold
        gold_inputs = (x, outer_scale, rht_sign, key)
    else:

        def run():
            return nvfp4_dim_m_rht_swizzle_pipelined(
                x, outer_scale, rht_sign
            )

        gold = Nvfp4GsSwizzleDimMRHTGold
        gold_inputs = (x, outer_scale, rht_sign)

    outputs = run()
    torch.cuda.synchronize()
    gold.correctness_fn(gold_inputs, outputs)
    bytes_per_iter = x.numel() * x.element_size() + sum(
        output.numel() * output.element_size() for output in outputs
    )
    return run, bytes_per_iter


def _bench_nvfp4_dim_m_rht_swizzle_pipelined(M, K):
    return _bench_nvfp4_dim_m_rht(M, K, stochastic=False)


def _bench_nvfp4_dim_m_swizzle_rht_sr_pipelined(M, K):
    return _bench_nvfp4_dim_m_rht(M, K, stochastic=True)


def _bench_nvfp4_dim_km_rht_pipelined(M, K, *, stochastic):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    rht_sign = torch.tensor([1, -1] * 8, device=x.device, dtype=x.dtype)
    (x_t_rht,) = hadamard_rht_fp32_f(x.t().contiguous(), rht_sign)
    outer_scale_k = nvfp4_gs_scale(x).reciprocal()
    outer_scale_m = nvfp4_gs_scale(x_t_rht).reciprocal()
    if stochastic:
        key = prng.key(0, device=x.device)

        def run():
            return nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined(
                x, outer_scale_k, outer_scale_m, rht_sign, key
            )

        gold = Nvfp4GsSwizzle_DimKSR_DimMRHTSR_Gold
        gold_inputs = (x, outer_scale_k, outer_scale_m, rht_sign, key)
    else:

        def run():
            return nvfp4_swizzle_dim_k_dim_m_rht_pipelined(
                x, outer_scale_k, outer_scale_m, rht_sign
            )

        gold = Nvfp4GsSwizzle_DimK_DimMRHT_Gold
        gold_inputs = (x, outer_scale_k, outer_scale_m, rht_sign)

    outputs = run()
    torch.cuda.synchronize()
    # The TE-style path intentionally uses BF16 UMMA plus approximate reciprocal math.
    gold.correctness_fn(gold_inputs, outputs)
    bytes_per_iter = x.numel() * x.element_size() + sum(
        output.numel() * output.element_size() for output in outputs
    )
    return run, bytes_per_iter


def _bench_nvfp4_swizzle_dim_k_dim_m_rht_pipelined(M, K):
    return _bench_nvfp4_dim_km_rht_pipelined(M, K, stochastic=False)


def _bench_nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined(M, K):
    return _bench_nvfp4_dim_km_rht_pipelined(M, K, stochastic=True)


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
    "mxfp8_swizzle": _bench_mxfp8_swizzle,
    "mxfp8": _bench_mxfp8,
    "mxfp8_swizzle_v2": _bench_mxfp8_swizzle_v2,
    "mxfp8_swizzle_sr_v2": _bench_mxfp8_swizzle_sr_v2,
    "mxfp8_32x32_swizzle_v2": _bench_mxfp8_32x32_swizzle_v2,
    "mxfp8_swizzle_v3": _bench_mxfp8_swizzle_v3,
    "mxfp8_swizzle_v4": _bench_mxfp8_swizzle_v4,
    "mxfp8_swizzle_sr_v4": _bench_mxfp8_swizzle_sr_v4,
    "mxfp8_swizzle_v5": _bench_mxfp8_swizzle_v5,
    "mxfp8_dim_m_swizzle_v2": _bench_mxfp8_dim_m_swizzle_v2,
    "mxfp8_dim_m_swizzle_sr_v2": _bench_mxfp8_dim_m_swizzle_sr_v2,
    "mxfp8_dim_km_swizzle_v2": _bench_mxfp8_dim_km_swizzle_v2,
    "mxfp8_dim_km_swizzle_sr_v2": _bench_mxfp8_dim_km_swizzle_sr_v2,
    "mxfp4_swizzle_v2": _bench_mxfp4_swizzle_v2,
    "mxfp4": _bench_mxfp4,
    "mxfp4_dim_m_swizzle_v2": _bench_mxfp4_dim_m_swizzle_v2,
    "mxfp4_dim_km_swizzle_v2": _bench_mxfp4_dim_km_swizzle_v2,
    "nvfp4_swizzle_direct": _bench_nvfp4_swizzle_direct,
    "nvfp4": _bench_nvfp4,
    "nvfp4_swizzle_tma": _bench_nvfp4_swizzle_tma,
    "nvfp4_swizzle_16x16_tma": _bench_nvfp4_swizzle_16x16_tma,
    "nvfp4_dim_m_swizzle_tma": _bench_nvfp4_dim_m_swizzle_tma,
    "nvfp4_dim_km_swizzle_tma": _bench_nvfp4_dim_km_swizzle_tma,
    "nvfp4_dim_m_rht_swizzle_pipelined": _bench_nvfp4_dim_m_rht_swizzle_pipelined,
    "nvfp4_dim_m_swizzle_rht_sr_pipelined": _bench_nvfp4_dim_m_swizzle_rht_sr_pipelined,
    "nvfp4_swizzle_dim_k_dim_m_rht_pipelined": _bench_nvfp4_swizzle_dim_k_dim_m_rht_pipelined,
    "nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined": _bench_nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined,
    "transpose_v0": _bench_transpose_v0,
    "transpose_v1": _bench_transpose_v1,
}

_KERNELS_SUPPORTING_FLOAT16_AND_FLOAT32 = frozenset({
    "mxfp8",
    "mxfp8_swizzle_v2",
    "mxfp8_swizzle_sr_v2",
    "mxfp8_32x32_swizzle_v2",
    "mxfp8_dim_m_swizzle_v2",
    "mxfp8_dim_m_swizzle_sr_v2",
    "mxfp8_dim_km_swizzle_v2",
    "mxfp8_dim_km_swizzle_sr_v2",
    "mxfp4_swizzle_v2",
    "mxfp4_dim_m_swizzle_v2",
    "mxfp4_dim_km_swizzle_v2",
    "mxfp4",
    "nvfp4",
    "nvfp4_swizzle_tma",
    "nvfp4_swizzle_16x16_tma",
    "nvfp4_dim_m_swizzle_tma",
    "nvfp4_dim_km_swizzle_tma",
})

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _parse_sizes(value: str, name: str) -> list[int]:
    """Parse one integer or a comma-separated list of integers."""
    items = str(value).split(",")
    if any(not item.strip() for item in items):
        raise ValueError(
            f"{name} must be an integer or comma-separated integers, got {value!r}"
        )
    try:
        sizes = [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer or comma-separated integers, got {value!r}"
        ) from exc
    if any(size <= 0 for size in sizes):
        raise ValueError(f"{name} values must be positive, got {value!r}")
    return sizes


def _benchmark_one(
    kernel: str, M: int, K: int, peak_bw: float, dtype: torch.dtype
):
    builder = _KERNELS[kernel]
    if kernel in _KERNELS_SUPPORTING_FLOAT16_AND_FLOAT32:
        run, bytes_per_iter = builder(M, K, dtype)
    else:
        run, bytes_per_iter = builder(M, K)
    # warm up so first-call costs (compile, autotune, allocator) don't leak into the timing.
    for _ in range(2):
        run()
    torch.cuda.synchronize()

    gpu_time_ms = do_bench_using_profiling(run)
    gbps = bytes_per_iter / (gpu_time_ms * 1e-3) / 1e9
    pct_peak = gbps / peak_bw * 100
    return gpu_time_ms, gbps, pct_peak


_OUTPUT_METRICS = ("gpu_time_ms", "tb_s", "pct_peak")

_CSV_FIELDS = ("kernel", "dtype", "M", "K", "gpu_time_ms", "tb_s", "pct_peak")

_MODEL_SHAPES = {
    "gpt-oss-120b": gpt_oss_120b_m8192_tp8_ep8,
}


def _parse_output_metrics(value: str) -> list[str]:
    metrics = [item.strip() for item in str(value).split(",")]
    if any(not metric for metric in metrics):
        raise ValueError(
            "output_metrics must be a comma-separated combination of "
            f"{_OUTPUT_METRICS}, got {value!r}"
        )
    invalid = [metric for metric in metrics if metric not in _OUTPUT_METRICS]
    if invalid:
        raise ValueError(
            f"unsupported output metrics {invalid}; choose from {_OUTPUT_METRICS}"
        )
    return list(dict.fromkeys(metrics))


def _format_metric(result, metric: str) -> str:
    gpu_time_ms, gbps, pct_peak = result
    if metric == "gpu_time_ms":
        return f"{gpu_time_ms:.4f}"
    if metric == "tb_s":
        return f"{gbps / 1000:.3f}"
    return f"{pct_peak:.1f}%"


def _parse_kernels(value: str) -> list[str]:
    kernels = [item.strip() for item in str(value).split(",")]
    if any(not kernel for kernel in kernels):
        raise ValueError(f"kernel must be a name or comma-separated names, got {value!r}")
    invalid = [kernel for kernel in kernels if kernel not in _KERNELS]
    if invalid:
        raise ValueError(f"unknown kernels {invalid}; have {sorted(_KERNELS)}")
    return list(dict.fromkeys(kernels))


def _parse_dtype(value: str) -> tuple[str, torch.dtype]:
    name = str(value).strip().lower()
    if name not in _DTYPES:
        raise ValueError(f"unsupported dtype {value!r}; choose from {tuple(_DTYPES)}")
    return name, _DTYPES[name]


@fire.decorators.SetParseFns(
    kernel=str,
    M=str,
    K=str,
    mk_mode=str,
    output_metrics=str,
    shapes_for_model=str,
    csv_output=str,
    dtype=str,
)
def main(
    kernel: str = "add_v0",
    M: str | None = None,
    K: str | None = None,
    mk_mode: str | None = None,
    output_metrics: str = "tb_s",
    shapes_for_model: str = "",
    csv_output: str = "",
    dtype: str = "bfloat16",
):
    """Benchmark handwritten CuTeDSL kernels over one shape or an M-by-K shape grid."""
    kernels = _parse_kernels(kernel)
    metrics = _parse_output_metrics(output_metrics)
    dtype_name, input_dtype = _parse_dtype(dtype)
    if input_dtype != torch.bfloat16:
        unsupported = [
            name
            for name in kernels
            if name not in _KERNELS_SUPPORTING_FLOAT16_AND_FLOAT32
        ]
        if unsupported:
            raise ValueError(
                f"{dtype_name} input is unsupported by kernels: "
                + ", ".join(unsupported)
            )

    shapes_for_model = shapes_for_model.strip().lower()
    if shapes_for_model:
        specified_shape_args = [
            name
            for name, value in (("M", M), ("K", K), ("mk_mode", mk_mode))
            if value is not None
        ]
        if specified_shape_args:
            raise ValueError(
                "shapes_for_model cannot be combined with "
                + ", ".join(specified_shape_args)
            )
        if shapes_for_model not in _MODEL_SHAPES:
            raise ValueError(
                f"unsupported shapes_for_model {shapes_for_model!r}; "
                f"choose from {tuple(_MODEL_SHAPES)}"
            )
        shape_pairs = _MODEL_SHAPES[shapes_for_model]()
        m_values = [m for m, _ in shape_pairs]
        k_values = [k for _, k in shape_pairs]
        mk_mode = "pair"
    else:
        m_values = _parse_sizes("16384" if M is None else M, "M")
        k_values = _parse_sizes("16384" if K is None else K, "K")
        mk_mode = "pair" if mk_mode is None else mk_mode.strip().lower()
        if mk_mode not in ("cartesian", "pair"):
            raise ValueError(
                f"unsupported mk_mode {mk_mode!r}; choose from ('cartesian', 'pair')"
            )
        if mk_mode == "pair":
            if len(m_values) != len(k_values):
                raise ValueError(
                    "pair mk_mode requires the same number of M and K values, got "
                    f"{len(m_values)} and {len(k_values)}"
                )
            shape_pairs = list(zip(m_values, k_values))
        else:
            shape_pairs = [(m, k) for m in m_values for k in k_values]

    device_name = torch.cuda.get_device_name(0)
    peak_bw = _peak_bw_gbps(device_name)

    # Prime Kineto and bring the GPU out of its initial idle state before recording the first real
    # data point. Without this, a short first kernel can report a substantially inflated duration.
    profiler_scratch = torch.empty(1, dtype=torch.int32, device="cuda")
    do_bench_using_profiling(
        profiler_scratch.zero_, warmup=2, rep=2, is_vetted_benchmarking=True
    )

    csv_path = None
    if csv_output:
        csv_path = Path(csv_output).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as csv_file:
            csv.DictWriter(
                csv_file, fieldnames=_CSV_FIELDS, lineterminator="\n"
            ).writeheader()

    for kernel_index, kernel_name in enumerate(kernels):
        # Keep kernel outermost so one kernel's complete shape grid finishes before the next starts.
        results = {
            (m, k): _benchmark_one(kernel_name, m, k, peak_bw, input_dtype)
            for m, k in shape_pairs
        }

        if csv_path is not None:
            with csv_path.open("a", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=_CSV_FIELDS, lineterminator="\n"
                )
                for m, k in shape_pairs:
                    gpu_time_ms, gbps, pct_peak = results[(m, k)]
                    writer.writerow(
                        {
                            "kernel": kernel_name,
                            "dtype": dtype_name,
                            "M": m,
                            "K": k,
                            "gpu_time_ms": f"{gpu_time_ms:.6f}",
                            "tb_s": f"{gbps / 1000:.6f}",
                            "pct_peak": f"{pct_peak:.6f}",
                        }
                    )

        if kernel_index:
            print()
        if mk_mode == "pair":
            print(f"kernel: {kernel_name}  dtype: {dtype_name}")
            print(f"device: {device_name} (peak {peak_bw / 1000:.2f} TB/s)")
            for metric_index, metric in enumerate(metrics):
                if metric_index:
                    print()
                print(f"metric: {metric}")
                rows = [
                    [f"({m}, {k})", _format_metric(results[(m, k)], metric)]
                    for m, k in shape_pairs
                ]
                print(
                    tabulate.tabulate(
                        rows,
                        headers=["(M, K)", metric],
                        colalign=("right", "right"),
                    )
                )
        elif len(m_values) > 1 or len(k_values) > 1:
            print(f"kernel: {kernel_name}  dtype: {dtype_name}")
            print(f"device: {device_name} (peak {peak_bw / 1000:.2f} TB/s)")
            for metric_index, metric in enumerate(metrics):
                if metric_index:
                    print()
                print(f"metric: {metric}")
                rows = []
                for m in m_values:
                    rows.append(
                        [m]
                        + [
                            _format_metric(results[(m, k)], metric)
                            for k in k_values
                        ]
                    )
                print(
                    tabulate.tabulate(
                        rows,
                        headers=["M \\ K", *k_values],
                        colalign=("right",) * (len(k_values) + 1),
                    )
                )
        else:
            m, k = m_values[0], k_values[0]
            result = results[(m, k)]
            print(f"kernel: {kernel_name}  shape: ({m}, {k})  dtype: {dtype_name}")
            print(f"device: {device_name} (peak {peak_bw / 1000:.2f} TB/s)")
            print(
                tabulate.tabulate(
                    [
                        [
                            kernel_name,
                            *[_format_metric(result, metric) for metric in metrics],
                        ]
                    ],
                    headers=["kernel", *metrics],
                    colalign=("left",) + ("right",) * len(metrics),
                )
            )

    if csv_path is not None:
        print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
