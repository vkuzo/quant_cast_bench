# Handwritten CuTe quantization kernels

## Performance versus TransformerEngine

The charts report logical throughput for BF16 square inputs on B200. Each panel compares
matching kernels exposed by `benchmark_transformer_engine.py`. CuTe-hand results are blue,
TransformerEngine results are red, RTNE is solid, and stochastic rounding is dashed. RHT
and non-RHT kernels remain in separate panels. A missing red SR line means that
TransformerEngine has no comparable stochastic MXFP8 implementation. The MXFP8 comparisons
include both the conventional 1x32 scale blocks and the 32x32 scale blocks used by
`mxfp8_32x32_swizzle_v2`. Panels are grouped into MXFP8, NVFP4, and deprecated NVFP4
sections.

![CuTe-hand versus TransformerEngine throughput](transformer_engine_comparison.png)

Regenerate the benchmark CSV and chart from the repository root with:

```bash
CUDA_VISIBLE_DEVICES=1 \
  quant_cast_bench/quant_cast_cute_hand/benchmarks/update_transformer_engine_comparison.sh
```

The script runs a paired square-shape sweep over powers of two and their midpoints from
2048 through 24576, writes
[`transformer_engine_comparison.csv`](transformer_engine_comparison.csv), and renders the
figure with Matplotlib.

## Performance versus MSLK

This chart compares the CuTe-hand TMA NVFP4 Dim-K kernel with MSLK's dense Triton
NVFP4 kernel on BF16 square inputs. Both kernels receive the same precomputed global
scale, produce bitwise-identical packed FP4 data and swizzled scale bytes, and are
reported using the same logical byte count. CuTe hand is blue and MSLK is red.

![CuTe-hand versus MSLK throughput](mslk_comparison.png)

Regenerate the benchmark CSV and chart from the repository root with:

```bash
CUDA_VISIBLE_DEVICES=1 \
  quant_cast_bench/quant_cast_cute_hand/benchmarks/update_mslk_comparison.sh
```

The script runs a paired square-shape sweep over powers of two and their midpoints from
2048 through 24576, writes
[`mslk_comparison.csv`](mslk_comparison.csv), and renders the figure with Matplotlib.
