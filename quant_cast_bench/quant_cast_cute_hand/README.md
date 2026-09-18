# Handwritten CuTe DSL quantization kernels

This directory contains optimized handwritten CuTe DSL quantization kernels. The
block-scaled TMA implementation lives in the `blockscaled_tma/` package, split
across `blockscale_tma_plan.py`, `blockscaled_tma_impl.py`, and
`blockscaled_tma_kernels.py`. It is an optimized
prototype rather than an upstream-ready PyTorch operator. The notes below capture
the remaining work and review findings for upstreaming it into PyTorch core.
The persistent NVFP4 RHT implementation follows the same plan/kernel/host split
in the `nvfp4_pipelined/` package.

## Block-scaled TMA upstream-readiness review

The kernel supports dim-K, dim-M, and dim-KM quantization, RTNE and stochastic
rounding, BF16/FP16/FP32 inputs, padded scale outputs.
Its synchronization sequence appears coherent for currently supported shapes:

1. Initialize the input TMA barrier.
2. Issue and await the input TMA transfer.
3. Complete the dim-M input reads and output writes, when enabled.
4. Synchronize before dim-K overwrites the aliased input shared-memory buffer.
5. Fence and synchronize the qdata shared-memory writes.
6. Issue the enabled qdata TMA stores.
7. Overlap the dim-K global scale stores with the qdata TMA store.

The following remaining issues should be addressed before upstreaming.

### For later

#### Define the stochastic-rounding RNG contract

The current API accepts an explicit two-element Philox key instead of using
PyTorch's generator and Philox-offset machinery. An upstream operator must
define:

- generator advancement and counter consumption per call;
- deterministic and reproducibility behavior;
- CUDA graph capture behavior;
- distributed-execution behavior;
- overflow/wraparound behavior for the counter.
