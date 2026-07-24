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
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.triton import TritonOverrides
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
        body_str, group = emitter.emit()

    `body_str` assumes the template already loaded the input tile into `input_var` (default
    "x_var") with shape `(BLOCK_M, BLOCK_N)`, and defines the block-size symbols `BLOCK_M`,
    `BLOCK_N`. It ends by aliasing each graph output to the corresponding `output_names` entry.
    `group` is the detected reduction group width (e.g. 128), or None if `f` had no group reshape.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        output_names: list[str],
        input_var: str = "x_var",
        block_m: str = "BLOCK_M",
        block_n: str = "BLOCK_N",
    ) -> None:
        self.gm = graph_module
        self.output_names = output_names
        self.input_var = input_var
        self.bm = block_m
        self.bn = block_n
        self.body = _Body()
        self.cse = _CSE(self.body)
        self.env: dict[torch.fx.Node, object] = {}
        self.group: int | None = None  # group width, set when the split reshape is seen

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
        # aten's `div` maps to python `truediv` in the op-overrides tables.
        return {"div": "truediv"}.get(name, name)

    # --- top-level walk ----------------------------------------------------------

    def emit(self) -> tuple[str, int | None]:
        placeholders = [n for n in self.gm.graph.nodes if n.op == "placeholder"]
        if len(placeholders) != 1:
            raise NotImplementedError(
                f"single-input tile only, got {len(placeholders)} placeholders"
            )
        self.env[placeholders[0]] = CSEVariable(
            self.input_var, ValueRanges.unknown(), dtype=self._meta_dtype(placeholders[0])
        )

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
        return self.body.getvalue(), self.group

    # --- per-node dispatch -------------------------------------------------------

    def _lower(self, node: torch.fx.Node):
        target = node.target
        if target in (aten.view.default, aten.reshape.default, aten._unsafe_view.default):
            return self._lower_view(node)
        if target == aten.squeeze.dim:
            return self._lower_squeeze(node)
        if target in (aten.t.default, aten.permute.default):
            return self._lower_transpose(node)
        if target == aten.clone.default:
            # contiguity is a memory-format hint; in registers a clone is a no-op passthrough.
            return self._val(node.args[0])
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
        v = self._meta_val(node)
        if out_rank == 3:
            in_shape = tuple(int(s) for s in self._meta_val(node.args[0]).shape)
            out_shape = tuple(int(s) for s in v.shape)
            # dim-M only: split the FIRST dim into (num_groups, group): (BM, BN) -> (BM//G, G, BN).
            # A preserved first dim (out_shape[0] == in_shape[0]) is the dim-K variant (split the LAST
            # dim, reduce columns) -- unsupported (no template), so reject it here rather than emit a
            # misclassified reshape. Under torch.compile this NotImplementedError falls back to eager.
            if out_shape[0] == in_shape[0]:
                raise NotImplementedError(
                    "reduction emitter supports only the dim-M variant (split dim0 into row-groups); "
                    "got a dim-K split (last dim reshaped into groups)"
                )
            self.group = out_shape[1]
            shape = f"[{self.bm} // {self.group}, {self.group}, {self.bn}]"
        elif out_rank == 2:
            # flatten the group axis back to the full tile: (..., G, ...) -> (BM, BN)
            shape = f"[{self.bm}, {self.bn}]"
        else:
            raise NotImplementedError(f"view to rank {out_rank} unsupported")
        return self.cse.generate(f"tl.reshape({x}, {shape})", dtype=self._meta_dtype(node))

    def _lower_squeeze(self, node: torch.fx.Node):
        # squeeze the size-1 keepdim axis left by the reduction, back to a 2D scale tile.
        #   dim-M: (NG, 1, BN) -> (NG, BN)
        x = self._val(node.args[0])
        if self.group is None:
            raise NotImplementedError("squeeze before any group reshape")
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
            #   dim-M (axis 1): (NG, 1, BN)
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

    def _lower_transpose(self, node: torch.fx.Node):
        # dim-M outputs are `.t()`-ed (2D): a register-level tl.trans of the tile. permute is only
        # supported when it's a plain 2D transpose (the only shape the dim-M recipe produces).
        x = self._val(node.args[0])
        if node.target == aten.permute.default:
            dims = node.args[1]
            if list(dims) != [1, 0]:
                raise NotImplementedError(f"only 2D transpose permute supported, got {dims}")
        return self.cse.generate(f"tl.trans({x})", dtype=self._meta_dtype(node))

    def _lower_pointwise(self, node: torch.fx.Node):
        name = self._op_name(node.target)
        fn = getattr(TritonOverrides, name, None)
        if fn is None:
            raise NotImplementedError(f"unsupported op {node.target} (no TritonOverrides.{name})")
        args = [self._val(a) for a in node.args]
        return self.cse.generate(fn(*args), dtype=self._meta_dtype(node))
