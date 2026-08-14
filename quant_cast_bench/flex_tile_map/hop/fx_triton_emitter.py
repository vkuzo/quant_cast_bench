"""A from-scratch FX -> Triton code emitter that fills a hand-written template hole.

This is the flex_tile_map analog of torch's `flex_gemm` epilogue emitter
(`torch/_inductor/kernel/flex_gemm/epilogue.py`): we walk a traced FX `GraphModule` (the body
of the user callback `f`) ourselves and emit a Triton code *string* to splice into a
hand-written template hole. We do NOT push the subgraph through Inductor's
`PointwiseSubgraphLowering` (the `{{ modification }}` hook), because that path is pointwise-only
and raises on a reduction. Owning the walk lets us emit a group reduction as a static
`tl.reshape` + `tl.max`.

What we borrow from Inductor (exactly the three things flex_gemm borrows):
  1. the FX graph as the source IR;
  2. `TritonOverrides` as an op -> Triton-string library (like flex_gemm drives
     `CuteDSLOpOverrides`), plus `CSEVariable` as the value type;
  3. (elsewhere) the template/autotune harness.

Scope (see the plan): a single 2D input tile `(BLOCK_M, BLOCK_N)` with `BLOCK_N` a multiple of
the reduction group (128); in-fragment reductions only -- a whole 128-group lives in one tile, so
the reduction is a `tl.max` over a static axis with no cross-tile combine.
"""

import torch
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.triton import triton_type, TritonOverrides
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges

aten = torch.ops.aten
prims = torch.ops.prims

# reduction op -> Triton reduce fn. In-fragment only (the group axis fits one tile), so each maps
# to a plain `tl.<fn>(x, axis=k)` over the static trailing axis produced by our group reshape.
_FUNCTION_REDUCTIONS = {
    aten.amax.default: "tl.max",
    aten.amin.default: "tl.min",
    aten.sum.dim_IntList: "tl.sum",
}


class _Body:
    """Line buffer for the emitted code (mirrors flex_gemm's FlexGemmCuteDSLBody)."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)

    def getvalue(self) -> str:
        return "\n".join(self.lines)


class _CSE:
    """`tmpN` allocator: emits `tmpN = <expr>` and returns a CSEVariable naming it.

    Mirrors flex_gemm's FlexGemmCuteDSLCSE. We reuse Inductor's CSEVariable as the value type so
    it flows through TritonOverrides unchanged (they format `f"...{var}..."`).
    """

    def __init__(self, body: _Body, prefix: str = "tmp") -> None:
        self.body = body
        self.prefix = prefix
        self.count = 0

    def generate(self, expr, dtype=None, shape=None) -> CSEVariable:
        name = f"{self.prefix}{self.count}"
        self.count += 1
        self.body.writeline(f"{name} = {expr}")
        return CSEVariable(name, ValueRanges.unknown(), dtype=dtype, shape=shape)


class _StubKernel:
    """Minimal object bound as `V.kernel` while emitting.

    `TritonOverrides` staticmethods return bare strings and mostly touch nothing on the kernel;
    `to_dtype` reads `.min_elem_per_thread` ONLY when `src_dtype` is passed (we never pass it).
    We still bind a kernel + ops handler so any incidental `ops.*`/`V.kernel.*` access resolves,
    exactly like flex_gemm wraps its walk in `V.set_kernel_handler` + `V.set_ops_handler`.
    """

    def __init__(self, cse: _CSE) -> None:
        self.cse = cse
        self.min_elem_per_thread = 0


class FxTritonEmitter:
    """Walk a traced `f` GraphModule and emit Triton code filling the template hole.

    Usage:
        emitter = FxTritonEmitter(graph_module, output_names=["qdata_var", "scale_var"])
        body_str, group, reduce_kind = emitter.emit()

    `body_str` assumes the template already loaded the input tile into `input_var` (default
    "x_var") with shape `(BLOCK_M, BLOCK_N)`, and defines the block-size symbols `BLOCK_M`,
    `BLOCK_N`. Any aux inputs (extra subgraph placeholders after the first) are assumed to be
    loaded into the corresponding `aux_names` (e.g. the nvfp4 per-tensor outer scale into
    "outer_var"). It ends by aliasing each graph output to the corresponding `output_names` entry.
    `group` is the detected reduction group width (e.g. 128), or None if `f` had no group reshape;
    `reduce_kind` is "dim_m" (split dim0, transposed outputs), "dim_k" (split the last dim, no
    transpose -- the nvfp4/mxfp8 1xG-along-columns shape), "block_2d" (split BOTH dims into square
    blocks and reduce over the whole block -- mxfp8 32x32), or None.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        output_names: list[str],
        input_var: str = "x_var",
        aux_names: list[str] | None = None,
        block_m: str = "BLOCK_M",
        block_n: str = "BLOCK_N",
    ) -> None:
        self.gm = graph_module
        self.output_names = output_names
        self.input_var = input_var
        self.aux_names = aux_names or []
        self.bm = block_m
        self.bn = block_n
        self.body = _Body()
        self.cse = _CSE(self.body)
        self.env: dict[torch.fx.Node, object] = {}
        self.group: int | None = None  # group width, set when the split reshape is seen
        self.reduce_kind: str | None = None  # "dim_m" | "dim_k", set at the split reshape
        # cache of (source node -> deinterleave-reshaped var) for the even/odd fp4-pack split.
        self._split_cache: dict[torch.fx.Node, object] = {}

    # --- helpers -----------------------------------------------------------------

    def _val(self, arg):
        """Resolve an FX arg to its emitted value (CSEVariable) or a python constant."""
        if isinstance(arg, torch.fx.Node):
            if arg not in self.env:
                raise NotImplementedError(f"unresolved node {arg} ({arg.target})")
            return self.env[arg]
        return arg  # int/float/dtype constant passes through

    @staticmethod
    def _meta_val(node):
        return node.meta.get("val") if isinstance(node, torch.fx.Node) else None

    def _meta_dtype(self, node):
        v = self._meta_val(node)
        return getattr(v, "dtype", None)

    def _meta_rank(self, node):
        v = self._meta_val(node)
        return len(v.shape) if hasattr(v, "shape") else None

    @staticmethod
    def _op_name(target) -> str:
        """FX target -> TritonOverrides method name (mirrors flex_gemm's `_cute_op_name`)."""
        if isinstance(target, torch._ops.OpOverload):
            name = target.overloadpacket.__name__
        else:
            name = getattr(target, "__name__", str(target))
        # aten's `div` maps to python `truediv`; the bit-shift dunders map to the bitwise_* names
        # in the op-overrides tables.
        return {
            "div": "truediv",
            "__lshift__": "bitwise_left_shift",
            "__rshift__": "bitwise_right_shift",
        }.get(name, name)

    # --- top-level walk ----------------------------------------------------------

    def emit(self) -> tuple[str, int | None, str | None]:
        placeholders = [n for n in self.gm.graph.nodes if n.op == "placeholder"]
        expected = 1 + len(self.aux_names)
        if len(placeholders) != expected:
            raise NotImplementedError(
                f"expected {expected} placeholders (1 input + {len(self.aux_names)} aux), "
                f"got {len(placeholders)}"
            )
        # placeholder[0] is the input tile; placeholder[1:] are the aux inputs, each already
        # loaded into its `aux_names` var by the template (REPLICATE: loaded once, whole).
        self.env[placeholders[0]] = CSEVariable(
            self.input_var, ValueRanges.unknown(), dtype=self._meta_dtype(placeholders[0])
        )
        for ph, name in zip(placeholders[1:], self.aux_names):
            self.env[ph] = CSEVariable(name, ValueRanges.unknown(), dtype=self._meta_dtype(ph))

        stub = _StubKernel(self.cse)
        outputs = None
        with V.set_kernel_handler(stub), V.set_ops_handler(TritonOverrides()):
            for node in self.gm.graph.nodes:
                if node.op == "placeholder":
                    continue
                if node.op == "output":
                    outputs = node.args[0]
                    continue
                if node.op != "call_function":
                    raise NotImplementedError(f"unsupported node op {node.op!r}")
                self.env[node] = self._lower(node)

        # flatten the output structure (`f` returns a tuple `(qdata, scale)`) to a flat list.
        flat_outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        if len(flat_outputs) != len(self.output_names):
            raise NotImplementedError(
                f"expected {len(self.output_names)} outputs, graph has {len(flat_outputs)}"
            )
        for out_node, out_name in zip(flat_outputs, self.output_names):
            self.body.writeline(f"{out_name} = {self._val(out_node)}")
        return self.body.getvalue(), self.group, self.reduce_kind

    # --- per-node dispatch -------------------------------------------------------

    def _lower(self, node: torch.fx.Node):
        target = node.target
        if target in (aten.view.default, aten.reshape.default, aten._unsafe_view.default):
            return self._lower_view(node)
        if target == aten.view.dtype:
            return self._lower_bitcast(node)
        if target == aten.squeeze.dim:
            return self._lower_squeeze(node)
        if target in (aten.t.default, aten.permute.default, aten.transpose.int):
            return self._lower_transpose(node)
        if target in (aten.clone.default, aten.alias.default):
            # contiguity/aliasing are memory-format hints; in registers both are no-op passthroughs.
            return self._val(node.args[0])
        if target in (aten.full_like.default, aten.full.default):
            return self._lower_full(node)
        if target == aten.slice.Tensor:
            return self._lower_slice(node)
        if target is inline_asm_elementwise:
            return self._lower_inline_asm(node)
        if target in _FUNCTION_REDUCTIONS:
            return self._lower_reduction(node)
        if target in (aten.clamp.default, aten.clamp_min.default, aten.clamp_max.default):
            return self._lower_clamp(node)
        if target in (aten._to_copy.default, prims.convert_element_type.default):
            return self._lower_to_dtype(node)
        return self._lower_pointwise(node)

    def _lower_view(self, node: torch.fx.Node):
        x = self._val(node.args[0])
        out_rank = self._meta_rank(node)
        in_rank = self._meta_rank(node.args[0])
        in_shape = tuple(int(s) for s in self._meta_val(node.args[0]).shape)
        out_shape = tuple(int(s) for s in self._meta_val(node).shape)
        if out_rank == 4:
            # 2D block reduction (mxfp8 32x32): a square-block cast splits BOTH dims into
            # (num_blocks, B). Two shapes, distinguished by the input rank:
            #   in_rank 2: (BM, BN)      -> (BM//B, B, BN//B, B)   [the forward split]
            #   in_rank 3: (RB, CB, B*B) -> (RB, CB, B, B)          [the reverse, for qdata]
            # The forward split is where we detect the block_2d kind (group = the block edge B).
            if in_rank == 2:
                self.reduce_kind = "block_2d"
                self.group = out_shape[1]  # block edge (32)
                g = self.group
                shape = f"[{self.bm} // {g}, {g}, {self.bn} // {g}, {g}]"
            else:
                g = self.group
                shape = f"[{self.bm} // {g}, {self.bn} // {g}, {g}, {g}]"
        elif out_rank == 3:
            if self.reduce_kind == "block_2d":
                # combine the two block axes into one reducible axis: (RB, CB, B, B) -> (RB, CB, B*B)
                g = self.group
                shape = f"[{self.bm} // {g}, {self.bn} // {g}, {g} * {g}]"
            else:
                # A group reshape splits one 2D axis into (num_groups, group). Two variants:
                #   dim-M: split the FIRST dim (BM, BN) -> (BM//G, G, BN)   [out_shape[0] != in]
                #   dim-K: split the LAST dim  (BM, BN) -> (BM, BN//G, G)   [out_shape[0] == in]
                # dim-M reduces down rows and transposes outputs; dim-K reduces along columns in
                # G-groups with no transpose (the nvfp4/mxfp8 1xG-along-columns shape).
                if out_shape[0] == in_shape[0]:
                    self.reduce_kind = "dim_k"
                    self.group = out_shape[2]
                    shape = f"[{self.bm}, {self.bn} // {self.group}, {self.group}]"
                else:
                    self.reduce_kind = "dim_m"
                    self.group = out_shape[1]
                    shape = f"[{self.bm} // {self.group}, {self.group}, {self.bn}]"
        elif out_rank == 2:
            if self.reduce_kind == "dim_k":
                # only rank-2 view on the dim-K path is the fp4-packed qdata flatten:
                # (BM, BN//G, G//2) fp4x2 -> (BM, BN//2). Two fp4 per byte halves the columns.
                shape = f"[{self.bm}, {self.bn} // 2]"
            else:
                # dim-M / block_2d: flatten back to the full tile (..., G, ...) -> (BM, BN)
                shape = f"[{self.bm}, {self.bn}]"
        else:
            raise NotImplementedError(f"view to rank {out_rank} unsupported")
        return self.cse.generate(f"tl.reshape({x}, {shape})", dtype=self._meta_dtype(node))

    def _lower_squeeze(self, node: torch.fx.Node):
        # squeeze the size-1 keepdim axis left by the reduction, back to a 2D scale tile.
        #   dim-M: (NG, 1, BN) -> (NG, BN);  dim-K: (BM, BN//G, 1) -> (BM, BN//G)
        x = self._val(node.args[0])
        if self.group is None:
            raise NotImplementedError("squeeze before any group reshape")
        if self.reduce_kind == "block_2d":
            # (RB, CB, 1) -> (RB, CB) = (BM//B, BN//B): one scale per 32x32 block.
            shape = f"[{self.bm} // {self.group}, {self.bn} // {self.group}]"
        elif self.reduce_kind == "dim_k":
            shape = f"[{self.bm}, {self.bn} // {self.group}]"
        else:
            shape = f"[{self.bm} // {self.group}, {self.bn}]"
        return self.cse.generate(f"tl.reshape({x}, {shape})", dtype=self._meta_dtype(node))

    def _lower_reduction(self, node: torch.fx.Node):
        x = self._val(node.args[0])
        fn = _FUNCTION_REDUCTIONS[node.target]
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        if not (isinstance(dims, (list, tuple)) and len(dims) == 1):
            raise NotImplementedError(f"only single-axis reductions supported, got dim={dims}")
        in_rank = self._meta_rank(node.args[0])
        axis = dims[0] % in_rank  # normalize -1 -> last static (group) axis
        reduced = self.cse.generate(f"{fn}({x}, axis={axis})", dtype=self._meta_dtype(node))
        if keepdim:
            # re-insert the reduced axis as size 1 so the following broadcast lines up.
            #   dim-M (axis 1): (NG, 1, BN);  dim-K (axis 2): (BM, BN//G, 1)
            #   block_2d (axis 2): (RB, CB, 1) = (BM//B, BN//B, 1)
            if self.reduce_kind == "block_2d":
                shape = f"[{self.bm} // {self.group}, {self.bn} // {self.group}, 1]"
            elif self.reduce_kind == "dim_k":
                shape = f"[{self.bm}, {self.bn} // {self.group}, 1]"
            else:
                shape = f"[{self.bm} // {self.group}, 1, {self.bn}]"
            reduced = self.cse.generate(
                f"tl.reshape({reduced}, {shape})", dtype=self._meta_dtype(node)
            )
        return reduced

    def _lower_clamp(self, node: torch.fx.Node):
        # clamp(x, min, max) / clamp_min(x, lo) / clamp_max(x, hi) -> maximum/minimum chain.
        x = self._val(node.args[0])
        if node.target == aten.clamp_min.default:
            lo = node.args[1] if len(node.args) > 1 else node.kwargs.get("min")
            hi = None
        elif node.target == aten.clamp_max.default:
            lo = None
            hi = node.args[1] if len(node.args) > 1 else node.kwargs.get("max")
        else:
            lo = node.args[1] if len(node.args) > 1 else node.kwargs.get("min")
            hi = node.args[2] if len(node.args) > 2 else node.kwargs.get("max")
        cur = x
        if lo is not None:
            cur = self.cse.generate(TritonOverrides.maximum(cur, lo), dtype=self._meta_dtype(node))
        if hi is not None:
            cur = self.cse.generate(TritonOverrides.minimum(cur, hi), dtype=self._meta_dtype(node))
        return cur

    def _lower_to_dtype(self, node: torch.fx.Node):
        x = self._val(node.args[0])
        dtype = node.kwargs.get("dtype")
        if dtype is None and len(node.args) > 1:
            dtype = node.args[1]
        # src_dtype omitted on purpose: passing it is what makes to_dtype touch min_elem_per_thread.
        expr = TritonOverrides.to_dtype(x, dtype)
        return self.cse.generate(expr, dtype=dtype)

    def _lower_bitcast(self, node: torch.fx.Node):
        # `x.view(dtype)` REINTERPRETS the bits (e.g. fp32<->int32, uint8<->e8m0) -- the mxfp8
        # e8m0 scale math extracts the exponent via int bit-ops then bitcasts back. Triton spells
        # this `x.to(triton_dtype, bitcast=True)`; TritonOverrides.to_dtype_bitcast needs the source
        # dtype (unlike the value-preserving to_dtype, here src_dtype is mandatory).
        x = self._val(node.args[0])
        dtype = node.args[1] if len(node.args) > 1 else node.kwargs.get("dtype")
        src_dtype = self._meta_dtype(node.args[0])
        expr = TritonOverrides.to_dtype_bitcast(x, dtype, src_dtype)
        return self.cse.generate(expr, dtype=dtype)

    def _lower_full(self, node: torch.fx.Node):
        # full_like(x, fill) / full(size, fill) with a CONSTANT fill collapses to the scalar itself
        # -- Triton broadcasts it against the other branch of the `where` it feeds (the e8m0 NaN
        # sentinel). Both overloads carry the fill at args[1] (full_like: (x, fill); full: (size,
        # fill)). A non-constant fill would need a real broadcast op; unsupported for now.
        fill = node.args[1] if len(node.args) > 1 else node.kwargs.get("fill_value")
        if isinstance(fill, torch.fx.Node):
            raise NotImplementedError("full/full_like with a non-constant fill is unsupported")
        return fill

    def _lower_slice(self, node: torch.fx.Node):
        # The fp4 pack deinterleaves each G-group into even/odd nibbles: `x[..., 0::2]` and
        # `x[..., 1::2]`. Both slices share one source; emit ONE reshape (BM, BN//G, G//2, 2) and
        # tl.split (which returns the (even, odd) halves along the trailing size-2 axis), then index
        # per slice start. Only this step-2 last-dim pattern is supported (raises otherwise).
        src = node.args[0]
        dim = node.args[1]
        start = node.args[2]
        step = node.args[4] if len(node.args) > 4 else 1
        in_rank = self._meta_rank(src)
        if step != 2 or (dim % in_rank) != in_rank - 1 or start not in (0, 1):
            raise NotImplementedError(
                "only the fp4 even/odd deinterleave (last-dim step-2 slice) is supported, got "
                f"slice(dim={dim}, start={start}, step={step})"
            )
        if self.group is None:
            raise NotImplementedError("deinterleave slice before any group reshape")
        reshaped = self._split_cache.get(src)
        if reshaped is None:
            x = self._val(src)
            shape = f"[{self.bm}, {self.bn} // {self.group}, {self.group // 2}, 2]"
            reshaped = self.cse.generate(f"tl.reshape({x}, {shape})", dtype=self._meta_dtype(src))
            self._split_cache[src] = reshaped
        # start 0 -> even half (split[0]), start 1 -> odd half (split[1]).
        return self.cse.generate(f"tl.split({reshaped})[{start}]", dtype=self._meta_dtype(node))

    def _lower_inline_asm(self, node: torch.fx.Node):
        # SM100 hardware fp4 pack: the traced InlineAsmElementwiseOp carries the PTX
        # (`cvt.rn.satfinite.e2m1x2.f32`) + constraints + dtype/pack; map it 1:1 onto Triton's
        # tl.inline_asm_elementwise. `node.args` are the (odd, even) fp32 halves to pack.
        args = [self._val(a) for a in node.args]
        asm = node.kwargs.get("asm_str", node.kwargs.get("asm"))
        constraints = node.kwargs["constraints"]
        dtype = node.kwargs["dtype"]
        is_pure = node.kwargs.get("is_pure", True)
        pack = node.kwargs.get("pack", 1)
        arg_list = "[" + ", ".join(str(a) for a in args) + "]"
        expr = (
            f"tl.inline_asm_elementwise({asm!r}, {constraints!r}, {arg_list}, "
            f"dtype={triton_type(dtype)}, is_pure={is_pure}, pack={pack})"
        )
        return self.cse.generate(expr, dtype=dtype)

    def _lower_transpose(self, node: torch.fx.Node):
        # dim-M outputs are `.t()`-ed (2D): a register-level tl.trans of the tile. The block_2d
        # (mxfp8 32x32) path swaps two axes of a rank-4 block tile -- traced as a `permute`
        # ([0, 2, 1, 3]) -- which lowers to the same explicit `tl.trans(x, *perm)` permutation.
        x = self._val(node.args[0])
        rank = self._meta_rank(node.args[0])
        if node.target == aten.transpose.int:
            d0, d1 = node.args[1] % rank, node.args[2] % rank
            perm = list(range(rank))
            perm[d0], perm[d1] = perm[d1], perm[d0]
        elif node.target == aten.permute.default:
            perm = [d % rank for d in node.args[1]]
        else:  # aten.t.default (2D)
            perm = [1, 0]
        dims_str = ", ".join(str(d) for d in perm)
        return self.cse.generate(f"tl.trans({x}, {dims_str})", dtype=self._meta_dtype(node))

    def _lower_pointwise(self, node: torch.fx.Node):
        name = self._op_name(node.target)
        fn = getattr(TritonOverrides, name, None)
        if fn is None:
            raise NotImplementedError(f"unsupported op {node.target} (no TritonOverrides.{name})")
        args = [self._val(a) for a in node.args]
        return self.cse.generate(fn(*args), dtype=self._meta_dtype(node))
