# quant_cast_bench

## quantization cast golden set

`quant_cast_gold` contains definitions of various quantization casts.  LLM friendly. Go to the directory's README.md for more context.

## backends

`quant_cast_helion`, `quant_cast_cute`, `flex_tile_map`, `quant_cast_triton` are backends implementing kernels for the casts in `quant_cast_gold`. See `benchmarks/README.md` for performance.  Run `pytest test/* -s` for correctness.
