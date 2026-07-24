# flex_gemm user API

Summary of the `flex_gemm` user-facing API **as it exists in the currently installed PyTorch**:

```
torch 2.14.0.dev20260720+cu130
```

Everything below is taken from that installed version
(`torch/_higher_order_ops/flex_gemm.py` and `torch/_inductor/kernel/flex_gemm/`). Nothing here is
unlanded / speculative — it reflects what actually ships in this build.

## The public function

```python
from torch._higher_order_ops.flex_gemm import flex_gemm

def flex_gemm(
    gemm_op,                       # which GEMM to run (see below)
    gemm_args,                     # tuple of the GEMM's operands
    epilogue_fn,                   # callable applied to the GEMM accumulator
    *,
    gemm_kwargs=None,              # scalar kwargs forwarded to gemm_op (alpha/beta)
    kernel_options=None,           # backend selection + tuning
):
    ...
```

`flex_gemm` runs `gemm_op(*gemm_args, **gemm_kwargs)` and applies `epilogue_fn` to the result,
fusing the two into a single kernel (when a fusing backend is chosen). It is a thin wrapper that
builds a body function `lambda *args: epilogue_fn(gemm_op(*args, **gemm_kwargs))` and calls the
underlying HOP `flex_gemm_hop` (see the last section).

### `gemm_op`

One of the supported GEMM ops. Both the `torch.*` alias and the `aten.*.default` overload are
accepted (aliases are normalized to the overload):

| alias        | aten overload             | operand layout (`gemm_args`)        | bias |
| ------------ | ------------------------- | ----------------------------------- | ---- |
| `torch.mm`   | `aten.mm.default`         | `(mat1, mat2)`                      | —    |
| `torch.addmm`| `aten.addmm.default`      | `(bias, mat1, mat2)`                | arg 0|
| `torch.bmm`  | `aten.bmm.default`        | `(mat1, mat2)` (3-D, batched)       | —    |
| `torch.baddbmm`| `aten.baddbmm.default`  | `(bias, mat1, mat2)` (3-D, batched) | arg 0|

Any other op raises `RuntimeError: unsupported GEMM op for FlexGEMM`.

### `gemm_args`

A tuple/list of the GEMM operands, positionally matching `gemm_op` (the table above). Elements must
be tensors or scalars (`Tensor`, `SymInt`, `SymFloat`, `SymBool`, `int`, `float`, `bool`) — anything
else raises. Any extra tensors the `epilogue_fn` needs are **not** passed here; they are captured by
closure and lifted automatically when tracing (see "Captured (aux) epilogue inputs" below).

### `epilogue_fn`

`epilogue_fn(acc) -> Tensor | tuple[Tensor, ...]`. It receives the GEMM accumulator `acc` and
returns:

- **a single tensor** — the fused kernel's one output; or
- **a tuple `(output, *aux_outputs)`** — a multi-output epilogue. The first element is the main
  output; the rest are auxiliary outputs emitted from the *same* in-register accumulator (e.g.
  returning `(acc.sin(), acc)` writes both `sin(a@b)` and the raw `a@b` from one kernel, avoiding a
  recompute). On the QUACK backend, aux tuple outputs are supported **only for `aten.mm`**, and each
  aux output's shape must match the main output's shape.

### Captured (aux) epilogue inputs

Any tensor the `epilogue_fn` reads besides the GEMM accumulator is a **captured input** (an "aux
input"). These are not listed in `gemm_args` — the epilogue just closes over them, and tracing lifts
them out and appends them to the operand tuple as trailing tensors. In the traced body they become
extra placeholders after the GEMM operands; the generated kernel exposes them as `aux0, aux1, …`
parameters and loads them per output tile.

```python
bias = torch.randn(1, N, device="cuda")       # captured, not in gemm_args
scale = torch.randn(M, 1, device="cuda")       # captured
d = flex_gemm(torch.mm, (a, b), lambda c: (c + bias) * scale,
              kernel_options={"backend": "QUACK"})
```

**Shape restrictions on captured inputs (QUACK backend):**

- Captured tensor reads are supported **only for `aten.mm`** (2-D output `[M, N]`). With
  `addmm`/`bmm`/`baddbmm`, capturing a tensor in the epilogue raises `NotImplementedError`.
- Each captured tensor's static shape must be exactly one of:
  - `[M, N]` — full output shape ("tile", read elementwise);
  - `[1, N]` — broadcast across rows ("row");
  - `[M, 1]` — broadcast across columns ("col").
- Any other shape raises `NotImplementedError` ("captured tensor epilogue args currently must match
  the GEMM output shape or broadcast as `[1, N]` / `[M, 1]`"). Shapes must be statically known.

### `gemm_kwargs`

Scalar keyword args forwarded to `gemm_op`. Must be a dict with no tensor values (pass tensors
through `gemm_args`). For `addmm` / `baddbmm` these are the `alpha` / `beta` scalars; the QUACK
backend accepts only `alpha` and `beta`, and they must be static scalars.

### `kernel_options`

A dict selecting the backend and tuning. Recognized keys:

- `"backend"`: `"TRITON"` (default) or `"QUACK"`.
  - **`TRITON`** — lowers via the ordinary subgraph path: it decomposes back into the GEMM plus a
    separate pointwise epilogue, i.e. **no real fusion**.
  - **`QUACK`** — lowers to a single fused GEMM+epilogue CuteDSL kernel (the actually-fused path).
    Requires `nvidia-cutlass-dsl >= 4.5.2`.
- `"tuned"`: `bool` (default `False`) — autotune over configs (QUACK only).

Any other key raises `NotImplementedError: unsupported FlexGEMM kernel options`. An unknown backend
raises `RuntimeError: unsupported FlexGEMM backend`.

## What the epilogue function may contain (QUACK backend)

The QUACK backend generates a CuteDSL epilogue by walking the traced body once in topological order
and mapping each node to a CuteDSL op. This constrains what the epilogue may do. (The default TRITON
backend does not fuse and is not subject to these rules — it lowers through ordinary Inductor.)

**Body structure**

- The body must contain **exactly one** GEMM node (the `gemm_op`); zero or more than one raises
  `NotImplementedError: FlexGEMM expects one GEMM body`.
- Every non-GEMM node must be a `call_function`. Anything else (a submodule call, a `get_attr`,
  Python-level control flow that survives into the graph, etc.) raises `unsupported FlexGEMM
  epilogue node`. In practice the epilogue must be a straight-line tensor expression — no
  data-dependent branching on tensor values.
- There must be exactly one output node, and it is either a single tensor node or a tuple/list
  `(output, *aux_outputs)` whose **every element is a tensor node** (returning a Python scalar or a
  non-tensor in the tuple raises `FlexGEMM expects tensor outputs`).

**Legal ops (elementwise / pointwise)**

Each op is resolved by its ATen overload-packet name against the CuteDSL op handler; if a handler of
that name exists, the op is legal. That covers the usual elementwise math, e.g.:

- arithmetic: `add`, `sub`, `mul`, `div`/`truediv`, `neg`, `reciprocal`, `pow`, `square`, `fma`;
- transcendental/activation: `sin`, `cos`, `tan`, `exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt`,
  `erf`, `tanh`, `sigmoid`, `relu`, `abs`, `sign`;
- compare/select: `eq`/`ne`/`lt`/`le`/`gt`/`ge`, `maximum`, `minimum`, `where`;
- clamping and casts: `clamp`, `clamp_min`, `clamp_max`, `_to_copy` / `convert_element_type` (dtype
  conversion). `_to_copy` accepts only a `dtype` (plus no-op `None` / `False` / `preserve_format`
  kwargs); other kwargs (e.g. `layout`, `device`, `memory_format=<real>`) raise.

An op with no matching handler raises `unsupported FlexGEMM epilogue op: <target>`.

**Shape rules for ops**

- Pointwise ops must preserve the output shape or broadcast to it; a captured/other input may
  broadcast in but the node's output shape must equal the GEMM output shape.

**Reductions (narrow "local reduction" contract only)**

Most reductions are **not** supported. Only a grouped local reduction over the GEMM output's M or N
dimension is, and only for these ops: `sum` (`aten.sum.dim_IntList`), `mean` (`aten.mean.dim`),
`prod` (`aten.prod.dim_int`), `amax` (`aten.amax.default`), `amin` (`aten.amin.default`) — and only
when the reduced dimension matches the recognized grouped layout (exact M/N reshape). `std`, `var`,
`any`, `all`, `argmax`, `argmin` are explicitly rejected, as is any reduction that doesn't fit the
grouped-dimension contract.

**Multi-output rules** (see `epilogue_fn` above): tuple `(output, *aux_outputs)`, aux outputs only
for `aten.mm`, each aux output's shape must equal the main output's shape, and at most one aux may be
a compressed local-reduction output.

## Examples

```python
# 1) simple fused mm + pointwise epilogue (QUACK = real fusion)
d = flex_gemm(torch.mm, (a, b), lambda c: c.sin(),
              kernel_options={"backend": "QUACK"})

# 2) dual output: emit sin(a@b) AND the raw accumulator a@b from one kernel
d, c = flex_gemm(torch.mm, (a, b), lambda c: (c.sin(), c),
                 kernel_options={"backend": "QUACK"})

# 3) addmm with alpha/beta
out = flex_gemm(torch.addmm, (bias, a, b), lambda acc: torch.relu(acc),
                gemm_kwargs={"alpha": 1.0, "beta": 1.0},
                kernel_options={"backend": "QUACK"})
```

## The underlying HOP (what shows up in an FX graph)

`flex_gemm` calls the `HigherOrderOperator`:

```python
from torch._higher_order_ops.flex_gemm import flex_gemm_hop

flex_gemm_hop(
    gemm_op,        # e.g. aten.mm.default
    body_fn,        # traced (a, b, *captured_aux) -> epilogue(gemm_op(a, b))
    gemm_args,      # tuple of operands (with any captured epilogue tensors appended)
    gemm_kwargs,    # dict
    kernel_options, # dict
)
```

Five **positional** args. This is the form the `flex_tile_map` fusion pass emits directly (it builds
`body_fn` from the traced epilogue). Note the argument order differs from the public `flex_gemm`:
the HOP is `(gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options)`, whereas `flex_gemm` is
`(gemm_op, gemm_args, epilogue_fn, *, gemm_kwargs, kernel_options)`.
