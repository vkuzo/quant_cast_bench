"""mm + flex_tile_map_ref -> flex_gemm fusion, in BOTH forward and backward.

A self-contained Inductor post-grad pass, auto-installed when the flex_tile_map package is
imported. It matches ONLY the reference-path HOP (``flex_tile_map_ref_hop``, see
``reference_hop.py``); the hand-rolled Triton-template HOP in ``hop/`` is deliberately NOT fused.

The user writes two separate ops -- ``c = a @ b`` then ``d = flex_tile_map(c, f)`` -- so the
autograd boundary sits around the epilogue ``f`` alone. Under ``torch.compile`` this pass
re-fuses the pair into a single ``flex_gemm_hop(aten.mm, body, (a, b, *aux), {}, {})``, recovering
the fused kernel. Because the pass runs on the forward and backward post-grad graphs separately,
the fusion fires in each (the backward's ``grad_d = grad_e @ w.T`` feeds its own epilogue).

Two stages run per graph:
  1. FUSION: rewrite every ``flex_tile_map_ref_hop(sub, mm(a, b), *aux)`` into a ``flex_gemm_hop``.
  2. INLINE: any ``flex_tile_map_ref_hop`` that survived (its operand is NOT an mm -- e.g. a
     standalone cast with no preceding gemm, as in the benchmark's REFERENCE+compile path) is
     spliced back into the parent graph as ordinary nodes, so regular Inductor lowers it. This is
     necessary because a HOP with no registered lowering hard-errors under Inductor (there is no
     graceful fallback to the eager body).

The fusion port (``_build_fused_body`` / ``_fuse_mm_into_flex_gemm`` / ``_resolve_gm``) is
verbatim from the working reference; only the matched HOP name differs.
"""

import operator

import torch
from torch._higher_order_ops.flex_gemm import flex_gemm_hop
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from .reference_hop import flex_tile_map_ref_hop

__all__ = [
    "flex_gemm_hop",
    "install_flex_tile_map_pass",
    "count_node_targets",
    "FLEX_TILE_MAP_PASS",
]

aten = torch.ops.aten


FLEX_TILE_MAP_PASS = PatternMatcherPass(pass_name="flex_tile_map_fusion")


def _resolve_gm(owning_gm, arg):
    """The HOP subgraph arg is a get_attr node (or, defensively, a GraphModule)."""
    if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
        return getattr(owning_gm, arg.target)
    if isinstance(arg, torch.fx.GraphModule):
        return arg
    raise AssertionError(f"unexpected flex_tile_map_ref subgraph arg: {arg!r}")


def _register_submodule(owning_gm, submod) -> str:
    """Register ``submod`` under a fresh qualname and return it."""
    i = 0
    while hasattr(owning_gm, f"flex_tile_map_fused_body_{i}"):
        i += 1
    name = f"flex_tile_map_fused_body_{i}"
    owning_gm.register_module(name, submod)
    return name


def _build_fused_body(epilogue_gm, tile_idx, mm_val, out_val, aux_vals):
    """Build a GraphModule ``(a, b, *aux) -> epilogue(...)`` for flex_gemm.

    This matches what flex_gemm's own body looks like: the GEMM operands ``a, b`` are the leading
    placeholders, any captured epilogue tensors follow as trailing placeholders (flex_gemm appends
    captured tensors to the gemm-args tuple), the body contains an ``aten.mm.default`` node, and
    the epilogue is inlined on top.

    The epilogue's placeholders correspond positionally to the flex_tile_map operands. ``tile_idx``
    is the operand index that is the mm tile (NOT necessarily 0 -- e.g. in a backward graph Dynamo
    may order a captured activation before the gradient tile). The tile placeholder maps to the mm
    result; every other epilogue placeholder maps to a fresh aux placeholder, in order.
    """
    g = torch.fx.Graph()
    # All placeholders MUST come first and contiguously: process_subgraph_nodes (the flex_gemm
    # default-backend lowering) binds placeholder node at absolute graph index i to args[i], so a
    # placeholder appearing after the mm node would read an out-of-range arg. Order: gemm operands
    # (a, b), then captured aux.
    pa = g.placeholder("a")
    pb = g.placeholder("b")
    aux_placeholders = []
    for i, aux_val in enumerate(aux_vals):
        p = g.placeholder(f"aux{i}")
        p.meta["val"] = aux_val
        aux_placeholders.append(p)

    mm_node = g.call_function(aten.mm.default, (pa, pb))
    mm_node.meta["val"] = mm_val

    remap = {}
    aux_iter = iter(aux_placeholders)
    output_val = None
    ph_idx = 0
    for node in epilogue_gm.graph.nodes:
        if node.op == "placeholder":
            if ph_idx == tile_idx:
                remap[node] = mm_node  # the tile -> mm(a, b)
            else:
                remap[node] = next(aux_iter)  # a captured operand -> aux input
            ph_idx += 1
        elif node.op == "output":
            output_val = node
        else:
            new_node = g.node_copy(node, lambda x: remap[x])
            remap[node] = new_node

    assert output_val is not None, "epilogue graph has no output"
    out_args = torch.fx.map_arg(output_val.args, lambda x: remap[x])
    out_node = g.output(out_args[0])
    out_node.meta["val"] = out_val

    fused = torch.fx.GraphModule(epilogue_gm, g)
    fused.recompile()
    return fused


@register_graph_pattern(
    CallFunctionVarArgs(flex_tile_map_ref_hop), pass_dict=FLEX_TILE_MAP_PASS
)
def _fuse_mm_into_flex_gemm(match: Match, *args, **kwargs):
    ftm_node = match.nodes[-1]
    # node is flex_tile_map_ref_hop(subgraph, *operands); operands correspond positionally to the
    # epilogue subgraph placeholders. Exactly one operand is the mm "tile"; the rest are captured
    # aux tensors (which may come BEFORE the tile -- e.g. in a backward graph Dynamo can order a
    # saved activation ahead of the grad tile).
    subgraph_arg = ftm_node.args[0]
    operands = list(ftm_node.args[1:])

    def is_fusible_mm(n):
        return (
            isinstance(n, torch.fx.Node)
            and n.op == "call_function"
            and n.target is aten.mm.default
        )

    # pick the tile operand: the (first) operand that is an aten.mm. It need NOT be single-use --
    # in a joint fwd+bwd graph the mm output commonly also feeds a saved-for-backward use; we
    # recompute the mm inside the fused body and only erase the original mm if nothing else reads it.
    tile_idx = next((i for i, n in enumerate(operands) if is_fusible_mm(n)), None)
    if tile_idx is None:
        return

    input_node = operands[tile_idx]
    aux_nodes = [n for i, n in enumerate(operands) if i != tile_idx]

    a, b = input_node.args
    graph = match.graph
    owning_gm = graph.owning_module

    epilogue_gm = _resolve_gm(owning_gm, subgraph_arg)
    mm_val = input_node.meta["val"]
    out_val = ftm_node.meta["val"]
    aux_vals = [n.meta["val"] for n in aux_nodes]
    fused_body = _build_fused_body(epilogue_gm, tile_idx, mm_val, out_val, aux_vals)

    body_name = _register_submodule(owning_gm, fused_body)

    with graph.inserting_before(ftm_node):
        body_attr = graph.get_attr(body_name)
        # Exactly the node the public flex_gemm(torch.mm, (a, b), epilogue_fn) builds: the gemm op,
        # a body graph (a, b, *aux) -> epilogue_fn(..., mm(a, b), ...), the gemm args (with any
        # captured epilogue tensors appended, as flex_gemm does), empty gemm_kwargs, empty
        # kernel_options (flex_gemm's default TRITON backend).
        new_node = graph.call_function(
            flex_gemm_hop,
            args=(aten.mm.default, body_attr, (a, b, *aux_nodes), {}, {}),
        )
    new_node.meta.update(ftm_node.meta)

    ftm_node.replace_all_uses_with(new_node)
    graph.erase_node(ftm_node)
    if len(input_node.users) == 0:
        graph.erase_node(input_node)


# ---------------------------------------------------------------------------
# Stage 2: inline a surviving (non-fused) reference HOP back into the graph
# ---------------------------------------------------------------------------


def _fresh_attr_name(owning_gm, base) -> str:
    name, i = base, 0
    while hasattr(owning_gm, name):
        name = f"{base}_{i}"
        i += 1
    return name


def _inline_reference_hop(graph, ftm_node):
    """Splice ``flex_tile_map_ref_hop``'s epilogue subgraph into the parent graph as plain nodes.

    Runs only on HOP nodes that did NOT fuse (no mm operand). Copies the subgraph body before the
    HOP node, binding its placeholders to the HOP's operands, then rewires the HOP's outputs to the
    copied output. Regular Inductor then lowers the inlined ops -- reproducing the generic kernel
    the REFERENCE backend produced before it routed through a HOP (and handling reductions, unlike
    a pointwise-only subgraph lowering).
    """
    owning_gm = graph.owning_module
    subgraph_arg = ftm_node.args[0]
    operands = list(ftm_node.args[1:])
    epilogue_gm = _resolve_gm(owning_gm, subgraph_arg)

    remap = {}
    ph_iter = iter(operands)
    epilogue_out = None
    with graph.inserting_before(ftm_node):
        for node in epilogue_gm.graph.nodes:
            if node.op == "placeholder":
                remap[node] = next(ph_iter)
            elif node.op == "output":
                epilogue_out = node
            elif node.op == "get_attr":
                # copy the referenced constant/attribute onto the parent module under a fresh name
                # so the copied get_attr resolves there.
                new_name = _fresh_attr_name(owning_gm, f"flex_tile_map_inlined_{node.target}")
                setattr(owning_gm, new_name, getattr(epilogue_gm, node.target))
                new_node = graph.get_attr(new_name)
                new_node.meta.update(node.meta)
                remap[node] = new_node
            else:
                remap[node] = graph.node_copy(node, lambda n: remap[n])

    assert epilogue_out is not None, "epilogue graph has no output"
    out_val = epilogue_out.args[0]

    if isinstance(out_val, (list, tuple)):
        mapped = [torch.fx.map_arg(v, lambda n: remap[n]) for v in out_val]
        # tuple-returning epilogue: downstream reads via getitem on the HOP node.
        for user in list(ftm_node.users):
            assert user.op == "call_function" and user.target is operator.getitem, (
                f"expected getitem on a tuple-returning flex_tile_map_ref, got {user.format_node()}"
            )
            user.replace_all_uses_with(mapped[user.args[1]])
            graph.erase_node(user)
    else:
        ftm_node.replace_all_uses_with(torch.fx.map_arg(out_val, lambda n: remap[n]))

    graph.erase_node(ftm_node)


# ---------------------------------------------------------------------------
# Pass installation
# ---------------------------------------------------------------------------


def _run_pass(graph: torch.fx.Graph) -> None:
    """Stage 1 (fuse mm+HOP -> flex_gemm), then stage 2 (inline any surviving HOP)."""
    FLEX_TILE_MAP_PASS.apply(graph)
    survivors = [
        n
        for n in graph.nodes
        if n.op == "call_function" and n.target is flex_tile_map_ref_hop
    ]
    for n in survivors:
        _inline_reference_hop(graph, n)
    if survivors:
        graph.owning_module.recompile()


class _FlexTileMapPass(CustomGraphPass):
    """Runs the fusion+inline and (optionally) records the resulting post-grad graph(s).

    The recorded graphs are the *actual* post-grad FX graphs (one per compiled graph -- forward and
    backward are separate), so a test can iterate ``graph.nodes`` and count real node targets
    (``flex_gemm_hop``, ``flex_tile_map_ref_hop``, ``aten.mm``).
    """

    def __init__(self, recorded_graphs=None):
        self._recorded = recorded_graphs

    def __call__(self, graph: torch.fx.Graph) -> None:
        _run_pass(graph)
        if self._recorded is not None:
            self._recorded.append(graph)

    def uuid(self):
        # depends on external captured state / an out-of-tree pass, so opt out of fx graph caching.
        return None


def _auto_install():
    """Install the fusion pass as Inductor's post-grad custom post pass on import.

    NOTE: post_grad_custom_post_pass is a single global slot, so auto-installing here stomps any
    pass a user already set. Acceptable for now (this package owns that slot).
    """
    torch._inductor.config.post_grad_custom_post_pass = _FlexTileMapPass()


def install_flex_tile_map_pass():
    """Re-install the pass in a *recording* variant and return the list it appends graphs to.

    After compilation the list holds the post-grad FX graph(s) the pass ran on -- inspect their
    ``.nodes`` directly (see :func:`count_node_targets`). Used by tests.
    """
    recorded_graphs: list = []
    torch._inductor.config.post_grad_custom_post_pass = _FlexTileMapPass(recorded_graphs)
    return recorded_graphs


def count_node_targets(graph, target) -> int:
    """Count call_function nodes in an fx graph whose target is ``target``."""
    return sum(n.op == "call_function" and n.target is target for n in graph.nodes)
