# Handwritten CuTe DSL quantization kernels

This directory contains optimized handwritten CuTe DSL quantization kernels. The
block-scaled TMA implementation lives in the `blockscaled_tma/` package, split
across `blockscaled_tma_config.py`, `blockscaled_tma_plan.py`,
`blockscaled_tma_impl.py`, and `blockscaled_tma_kernels.py`. It is an optimized
prototype rather than an upstream-ready PyTorch operator. The notes below capture
the remaining work and review findings for upstreaming it into PyTorch core.
The persistent NVFP4 RHT implementation follows the same plan/kernel/host split
in the `nvfp4_pipelined/` package.

## Support matrix

| Recipe | `kernel_family` | Format | Orientation | Swizzle | `rounding_mode` | RHT | `is_square_scaling` | Input dtypes |
|---|---|---|---|---|---|---|---|---|
| `mxfp8` | `blockscaled_tma` | MXFP8 | dim-K | No | `rtne` | No | No | FP32, BF16, FP16 |
| `mxfp8_swizzle_v2` | `blockscaled_tma` | MXFP8 | dim-K | Yes | `rtne`, `stochastic` | No | No | FP32, BF16, FP16 |
| `mxfp8_dim_m_swizzle_v2` | `blockscaled_tma` | MXFP8 | dim-M | Yes | `rtne`, `stochastic` | No | No | FP32, BF16, FP16 |
| `mxfp8_dim_km_swizzle_v2` | `blockscaled_tma` | MXFP8 | dim-KM | Yes | `rtne`, `stochastic` | No | No | FP32, BF16, FP16 |
| `mxfp8_32x32_swizzle_v2` | `blockscaled_tma` | MXFP8 | dim-K | Yes | `rtne` | No | Yes (32x32) | FP32, BF16, FP16 |
| `mxfp4` | `blockscaled_tma` | MXFP4 | dim-K | No | `rtne` | No | No | FP32, BF16, FP16 |
| `mxfp4_swizzle_v2` | `blockscaled_tma` | MXFP4 | dim-K | Yes | `rtne` | No | No | FP32, BF16, FP16 |
| `mxfp4_dim_m_swizzle_v2` | `blockscaled_tma` | MXFP4 | dim-M | Yes | `rtne` | No | No | FP32, BF16, FP16 |
| `mxfp4_dim_km_swizzle_v2` | `blockscaled_tma` | MXFP4 | dim-KM | Yes | `rtne` | No | No | FP32, BF16, FP16 |
| `nvfp4` | `blockscaled_tma` | NVFP4 | dim-K | No | `rtne` | No | No | FP32, BF16, FP16 |
| `nvfp4_swizzle_tma` | `blockscaled_tma` | NVFP4 | dim-K | Yes | `rtne` | No | No | FP32, BF16, FP16 |
| `nvfp4_dim_m_swizzle_tma` | `blockscaled_tma` | NVFP4 | dim-M | Yes | `rtne` | No | No | FP32, BF16, FP16 |
| `nvfp4_dim_km_swizzle_tma` | `blockscaled_tma` | NVFP4 | dim-KM | Yes | `rtne` | No | No | FP32, BF16, FP16 |
| `nvfp4_swizzle_16x16_tma` | `blockscaled_tma` | NVFP4 | dim-K | Yes | `rtne` | No | Yes (16x16) | FP32, BF16, FP16 |
| `nvfp4_dim_m_rht_swizzle_pipelined` | `nvfp4_pipelined` | NVFP4 | dim-M | Yes | `rtne` | Yes | No | BF16 |
| `nvfp4_dim_m_swizzle_rht_sr_pipelined` | `nvfp4_pipelined` | NVFP4 | dim-M | Yes | `stochastic` | Yes | No | BF16 |
| `nvfp4_swizzle_dim_k_dim_m_rht_pipelined` | `nvfp4_pipelined` | NVFP4 | dim-KM | Yes | `rtne` | Yes | No | BF16 |
| `nvfp4_swizzle_dim_k_sr_dim_m_rht_sr_pipelined` | `nvfp4_pipelined` | NVFP4 | dim-KM | Yes | `stochastic` | Yes | No | BF16 |

For the dim-KM pipelined recipes, RHT is applied only to the dim-M pass. The
`nvfp4_pipelined` family is currently BF16-only because its UMMA path is built
around BF16 operands; the `blockscaled_tma` family supports all three listed
input dtypes.

## For later

#### Define the stochastic-rounding RNG contract

The current API accepts an explicit two-element Philox key instead of using
PyTorch's generator and Philox-offset machinery. An upstream operator must
define:

- generator advancement and counter consumption per call;
- deterministic and reproducibility behavior;
- CUDA graph capture behavior;
- distributed-execution behavior;
- overflow/wraparound behavior for the counter.
