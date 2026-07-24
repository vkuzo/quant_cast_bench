# Future ideas: generalizing the TRITON_TEMPLATE backend (`hop/`)

Brainstorm on making `hop/` (the hand-rolled Triton-template backend) more general — more
inputs, more outputs, more reductions in a single template — without building a general compiler.
The load-bearing assumption throughout is that `f` is **tile-invariant**.

## What's hardcoded today

Three files conspire to make this deepseek-dim-M-only:

- **`fx_triton_emitter.py`** — hard-asserts **exactly one placeholder** (`emit()` raises on ≠1
  input); `output_names` is a fixed 2-list `["qdata_var", "scale_var"]`; `_lower_view` explicitly
  *rejects* the dim-K split. So: 1 input, 2 outputs, dim-M reduction only. It also carries a
  **single-group assumption as scalar state**: `self.group` is one `int`, and
  `_lower_view` / `_lower_squeeze` / the `_lower_reduction` keepdim path emit hardcoded shape
  *strings* (`[BM // G, G, BN]`, `[NG, 1, BN]`, `[BM//G, BN]`), all derived from that one group and
  all dim-M-shaped.
- **`template_deepseek_dim_m.py.jinja`** — bakes in the entire I/O geometry: one input `X` loaded
  row-major, the **transpose** of both outputs, the `//128` group, the scale layout `(N, M//128)`,
  and which output is "primary" (`store_output`) vs "mutated" (`S`).
- **`inductor_lowering.py`** — hardcodes `output_names=["qdata_var","scale_var"]`, unpacks exactly
  `(qdata_node, scale_node)`, selects the one template by name, and computes the two output layouts
  by hand.

The emitter's *math* walk (pointwise/cast/clamp/view/reduce) is already fairly general. The
narrowness lives in (a) the I/O prologue/epilogue that the template owns and hardcodes, and (b) the
single-`self.group` shape bookkeeping.

## The one unifying idea: an IO-descriptor layer

Most hacks above are "we know statically what inputs/outputs look like, but wrote it as prose in a
template instead of deriving it." Introduce a descriptor layer between the trace and a *generic*
template:

- **Input descriptors** — one per placeholder: dtype + shape-class relative to the tile
  (`full [M,N]`, `broadcast-row [1,N]`, `broadcast-col [M,1]`, scalar). This is what flex_gemm
  already does for captured aux, and what the API's existing `aux_kinds` encodes.
- **Output descriptors** — one per graph output: dtype, transpose flag, reduction-group divisor.
  All derivable from the traced graph (the emitter already tracks `tl.trans` and the group reshape;
  it just throws the info away). `output_kinds` in the API is the same info at the frontend.

With these lists, "dim-M vs dim-K" stops being two templates — it's `transpose=True,
store_shape=(N,M)` vs `transpose=False, store_shape=(M//G, N)` in an output descriptor. And
"2 outputs" becomes `len(output_descriptors)`.

## Generalizing each axis

- **More inputs.** Drop the single-placeholder assertion; bind placeholders to `in0_var, in1_var,
  …`. The load prologue becomes a jinja loop over input descriptors (each emits its own
  `offs/mask/tl.load`, respecting broadcast-class). New emitter capability required:
  **broadcasting between tiles of different shape-classes** (`[M,N] * [1,N]`) — track each CSE
  value's tile-shape and emit the right `[:,None]`/`[None,:]`. This is the only genuinely new
  emitter logic multi-input needs.
- **More outputs.** Generalize `output_names` to arbitrary length; generate a store per output
  descriptor. Inductor harness constraint: one primary output (`store_output`) + the rest as
  `mutated_inputs` (flexquant v1 already lives with this). So M outputs = 1 primary + (M−1)
  pre-allocated mutated buffers, layouts computed from descriptors.
- **More reductions in one template.** Each reduction is already an in-fragment `tl.<fn>(axis=k)`,
  so multiple reductions are just multiple such lines over different static axes — *no template
  change*, once the store side stops assuming a single scale output. Add `mean` (sum/count) and
  `prod` to `_FUNCTION_REDUCTIONS`. The dim-K variant becomes reachable purely by removing the
  `_lower_view` rejection and letting the output descriptor carry `transpose=False`.

## The generic template

Collapse the per-recipe `.jinja` into **one skeleton** that owns only the invariant plumbing
(`def_kernel`, `program_id`, block symbols, the `__EMITTER_BODY__` hole). The load prologue and
store epilogue are generated from the descriptors — lean toward **jinja-loops-over-descriptors** to
preserve the clean "template owns I/O, emitter owns math" split that mirrors flex_gemm's epilogue
emitter.

## Tile-invariance: what it does and doesn't buy

**Necessary precondition for the whole backend.** The template picks a tiling and autotunes over it,
so if `f` weren't tile-invariant, different tilings would give different answers.

**The real enabler for reductions.** Because `f`'s output-per-tile is independent of tiling, we can
always choose a tiling where each reduction group is contained in a single tile:

> reductions become a pure per-tile emit (`tl.<fn>` over a static axis), never a *scheduling*
> problem (cross-tile combine / atomics / two-pass — the hard part of a real compiler).

This is why the emitter stays ~300 lines and needs no reduction scheduler. Encode it as an
**autotune constraint**: each reduction group must divide, and be ≤, a tunable tile dim; with
multiple reductions the BLOCK dims must be a common multiple of all groups along that axis.

**Two sharpenings (tile-invariance is not the whole story):**

1. Tile-invariance alone is **not sufficient even for reductions** — the reduction must also be
   **tile-local**, i.e. the group is *bounded* so some feasible tile contains it. A per-tensor amax
   is tile-invariant trivially but its group is the whole dimension → no tile contains it → needs a
   real combine → **out of scope, must fall back**. Tile-invariance lets us freely pick a tiling to
   satisfy the fit, but only for bounded groups.
2. **Multi-input / multi-output / more ops are mostly orthogonal to tile-invariance** — they're
   plain emitter engineering (descriptor layer + per-value symbolic shape tracking + broadcast).
   Tile-invariance neither blocks nor grants them.

**One-line gate for "is this doable in this backend?":** *every output of `f` for a tile depends
only on that tile's inputs, and every reduction is over a bounded axis that fits inside a tile.*

## Multiple reductions specifically

Yes — multiple generic reductions land, *as long as they stay tile-local*. At the Triton level any
number of in-fragment reductions compose natively (no combine). The **actual blocker is not the
reduction op** — it's the emitter's single-`self.group` scalar + hardcoded reshape strings. The fix
is the same "track each CSE value's symbolic tile-shape (a list of dim expressions)" work needed for
multi-input: derive every `tl.reshape`/reduce-axis/broadcast from per-value shape, not from one
memorized group. Once values carry their own shapes, reductions with different group widths or axes
fall out with no special-casing.

A reduction qualifies as tile-local only if:
1. its reduced axis is a **grouped/reshaped inner axis** (e.g. the 128-split), never the tiled outer
   axis that spans program-ids;
2. every group **divides and fits** a tile dim (the constraint composes across reductions);
3. reductions may feed each other or feed pointwise, as long as each stays over an inside-tile axis.

Out of scope (must fall back, not mis-emit): group = whole dimension, or reduction along the tiled
axis (reduce-across-tiles).

## Cross-cutting: graceful fallback is not optional

Today an unsupported op/shape raises `NotImplementedError` in the lowering, which **hard-errors**
under `torch.compile` (verified — a HOP `@register_lowering` that raises does not fall back to the
eager body). A general framework needs a clean `can_emit(gm) -> bool` predicate so unsupported `f`
routes to the inductor-inline path instead of crashing the compile. This is what turns "narrow but
safe" into a real contract: accept exactly the tile-local set, route everything else out.

## Suggested staging

1. **Descriptor layer first** — derive input/output descriptors from the trace (+ reuse the API's
   `aux_kinds`/`output_kinds`); rewrite the current deepseek path through it with *no behavior
   change*. Pure refactor.
2. **Generic template** — replace hardcoded load/store with descriptor-driven jinja; deepseek
   becomes one instance.
3. **Per-value symbolic shape tracking** in the emitter (replaces `self.group`) — unblocks both
   multi-reduction and multi-input broadcast.
4. **Multi-output**, then **dim-K** (remove the rejection), then **multi-input**.
5. **`can_emit` predicate + fallback** to the inductor-inline path — alongside, not after.

Net: the generalization isn't a bigger emitter — it's **hoisting I/O geometry out of a hand-written
template into descriptors derived from the trace, plus per-value shape tracking**, with
tile-invariance letting every (bounded) reduction stay in-fragment so we never build a scheduler.
