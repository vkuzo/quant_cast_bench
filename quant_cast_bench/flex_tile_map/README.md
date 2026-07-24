# flex_tile_map

## context

This is a study of how to express quantization of a tensor in a tile 
invariant way, to inform:

1. what could a general tensor quantization API in PyTorch look like 
   (`flex_tile_map` below), and whether this makes sense to build
2. what are requirements that other flex* projects 
   (flex_gemm, flex_ep, flex_moe) should consider to cover quantization kernel
   authoring

## HOPs: two separate paths

There are two independent HigherOrderOperators, on purpose:

- **`hop/` — the hand-rolled Triton-template HOP** (`TRITON_TEMPLATE` backend). Traces `f` on the
  full input shape and lowers it onto a hand-written Triton template. Kept hand-rolled (not
  migrated to `BaseHOP`) to avoid re-homing the FxTritonEmitter lowering + full-shape reduction
  tracing. **Not fused** into flex_gemm.
- **`reference_hop.py` — a `BaseHOP`** (`REFERENCE` backend). This is the fusible path: under
  `torch.compile` the post-grad pass in `flex_gemm_to_tile_map_fusion.py` rewrites a preceding `mm` into a single
  `flex_gemm` call (in both the forward and backward graphs). It is a `BaseHOP` because that
  supplies correct forward+backward autograd *and* Dynamo captured-freevar lifting for free — the
  latter is what makes the backward VJP epilogue (which captures a saved activation) fuse. Its
  subgraph-first arg order also matches flex_gemm/flex_attention.

Future work: the hand-rolled Triton-template HOP could migrate to `BaseHOP` too, for the same
autograd/freevar-lifting benefits, once its template lowering is re-homed under a `BaseHOP`.

### Notes / caveats

- A HOP `@register_lowering` that raises does **not** gracefully fall back to the eager body — it
  hard-errors (`InductorError: LoweringException`). So a surviving reference HOP (no preceding mm)
  is handled explicitly by a splice-inline stage in `flex_gemm_to_tile_map_fusion.py`, not left to a fallback.
- The fusion pass auto-installs into `torch._inductor.config.post_grad_custom_post_pass` on import.
  That is a single global slot, so this stomps any pass a user already set (acceptable for now).
