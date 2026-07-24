# flex_tile_map

API:

```python
# TODO align aux_inputs with flex_gemm
outputs = flex_tile_map(fn, input, aux_inputs)
```

Reason #1 to exist: express CODA in fwd+bwd without resorting to large `torch.autograd.Function`

**Before** (without flex_tile_map): user writes large `torch.autograd.Function` and
fuses gemm to epilogue by hand

```python
class MmSinMm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, w):
        d, c = flex_gemm(torch.mm, (a, b), lambda c: (c.sin(), c))
        e = torch.mm(d, w)
        ctx.save_for_backward(a, b, w, c, d)
        return e

    @staticmethod
    def backward(ctx, grad_e):
        a, b, w, c, d = ctx.saved_tensors
        grad_d = flex_gemm(torch.mm, (grad_e, w.t()), lambda gd: gd * c.cos())
        grad_a = torch.mm(grad_d, b.t())
        grad_b = torch.mm(a.t(), grad_d)
        grad_w = torch.mm(d.t(), grad_e)
        return grad_a, grad_b, grad_w

e = MmSinMm.apply(a, b, w)
```

**After** (with flex_tile_map): user writes `torch.mm` + `flex_tile_map(..., fn)`,
torch.compile fuses the fwd+bwd parts to get equivalent code to the manual
`torch.autograd.Function` above.

```python
@torch.compile()
def f(a, b, w):
    c = torch.mm(a, b)
    d = flex_tile_map(c, lambda c: c.sin())  # fuses with the mm above -> one flex_gemm kernel
    return torch.mm(d, w)

e = f(a, b, w)
```


Reason #2 to exist: general frontend for easy to medium cases for quant casting. Punt on hard cases

TODO insert quant cast graphs here

TODO talk somewhere about quant cast taxonomy, tile invariant-ness, and aligning everything

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

  The fused `flex_gemm` uses the **QUACK** backend (`kernel_options={"backend": "QUACK"}`) — the
  only flex_gemm backend that lowers to a single fused GEMM+epilogue CuteDSL kernel. flex_gemm's
  default `TRITON` backend merely decomposes back into `mm` + a separate pointwise, i.e. no actual
  fusion, so it is not used here. QUACK requires `nvidia-cutlass-dsl >= 4.5.2` (older releases have
  incompatible `cutlass.cute` APIs); the fusion tests gate on that version.

  **Dual-output epilogue (no redundant matmul).** When the mm result is also consumed outside the
  epilogue — the usual forward case, where the raw product `c = a @ b` is saved for the backward's
  VJP (`grad_c = grad_out * cos(c)`) — the pass builds the fused body to return a two-tuple
  `(epilogue(mm), mm)`. This drives flex_gemm's aux-output path so the raw accumulator is emitted as
  a *second output of the same fused kernel* (both derived from the one in-register accumulator),
  instead of leaving a separate `extern_kernels.mm` behind to recompute `a @ b` just for the save.
  The forward then computes `a @ b` exactly once (one fused kernel + the un-fused second gemm).
  flex_gemm's QUACK backend supports at most one such aux output.

Future work: the hand-rolled Triton-template HOP could migrate to `BaseHOP` too, for the same
autograd/freevar-lifting benefits, once its template lowering is re-homed under a `BaseHOP`.

### Notes / caveats

- A HOP `@register_lowering` that raises does **not** gracefully fall back to the eager body — it
  hard-errors (`InductorError: LoweringException`). So a surviving reference HOP (no preceding mm)
  is handled explicitly by a splice-inline stage in `flex_gemm_to_tile_map_fusion.py`, not left to a fallback.
- The fusion pass auto-installs into `torch._inductor.config.post_grad_custom_post_pass` on import.
  That is a single global slot, so this stomps any pass a user already set (acceptable for now).
