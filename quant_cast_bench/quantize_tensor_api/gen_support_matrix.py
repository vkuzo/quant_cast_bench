"""Generate the `quantize_tensor*` support matrices in README.md by PROBING the real API.

This does NOT hardcode the supported rows -- it inspects the actual dispatch in `api.py` by running
every argument combination through each of the four entry points (`quantize_tensor`,
`quantize_tensor_dual`, `quantize_tensor_grouped`,
`quantize_tensor_grouped_dual`) on real CUDA tensors and keeping the ones the frontend
accepts (i.e. that don't raise). For each accepted combination it also records the function the call
dispatched to (kernel = a `quant_cast_triton` recipe, reference = a `quant_cast_gold` one), captured
by wrapping those recipe callables in the `api` and `moe_utils` module namespaces.

    python quant_cast_bench/quantize_tensor_api/gen_support_matrix.py

Requires a CUDA device. It rewrites the region between each BEGIN/END GENERATED marker pair in
README.md in place. Rows are emitted in deterministic lexicographic order, so the output is
byte-stable (rerunning without an `api.py` dispatch change is a no-op).
"""

import itertools
from pathlib import Path

import torch
import torch.func._random as prng

from quant_cast_bench.quantize_tensor_api import api, moe_utils
from quant_cast_bench.quantize_tensor_api.api import (
    InnerScaleCalc,
    RoundingMode,
    ScalingType,
    SwizzleType,
    quantize_tensor,
    quantize_tensor_dual,
    quantize_tensor_grouped,
    quantize_tensor_grouped_dual,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    hadamard_rht_matrix,
    nvfp4_gs_per_token_scale,
    nvfp4_gs_scale,
)

# --- per-entry-point column layouts (status is always second-to-last, dispatch last) --------------
COLUMNS = [
    "format", "scl_tp", "orient", "swizzle_type", "sqex", "rnd_md", "rht", "status", "dispatches to",
]
COLUMNS_BI = [
    "format", "scl_tp", "swizzle_type", "skip_tr", "sqex", "rnd_md", "status", "dispatches to",
]
COLUMNS_GROUPED = [
    "format", "scl_tp", "orient", "swizzle_type", "rnd_md", "status", "dispatches to",
]
COLUMNS_GROUPED_BI = [
    "format", "scl_tp", "swizzle_type", "skip_tr", "rnd_md", "status", "dispatches to",
]

# Abbreviations applied to keep the tables dense; rendered as a legend under the first table.
HEADER_ABBR = {
    "scl_tp": "scaling_type",
    "orient": "orientation",
    "rnd_md": "qdata_rounding_mode",
    "rht": "rht_tensor",
    "skip_tr": "skip_transposed_qdata",
    "sqex": "scaling_type_square_block_and_expand",
}
VALUE_ABBR = {
    "1x16": "BlockWise1x16",
    "1x32": "BlockWise1x32",
    "TW": "TensorWise",
    "RW": "RowWise",
    "NT": "NATURAL",
    "TR": "TRANSPOSED",
    "NONE": "NO_SWIZZLE",
    "32_4_4": "SWIZZLE_32_4_4",
    "RS": "STOCHASTIC",
}

STATUS_EMOJI = {"kernel": "🟢", "reference": "🟡"}


def _begin(tag: str) -> str:
    return f"<!-- BEGIN GENERATED: {tag} (python gen_support_matrix.py) -->"


def _end(tag: str) -> str:
    return f"<!-- END GENERATED: {tag} -->"


# --- per-axis value grids (the argument cross-product we probe) -----------------------------------
QDATA = [torch.float8_e4m3fn, torch.float4_e2m1fn_x2]
INNER = [InnerScaleCalc.RCEIL_E8M0, InnerScaleCalc.NVFP4_E4M3]
SCALING = [ScalingType.BlockWise1x16, ScalingType.BlockWise1x32]
ORIENT = ["dim_k", "dim_m"]  # dim_k = contiguous input; dim_m = a transposed view of it
SWIZZLE = [SwizzleType.NO_SWIZZLE, SwizzleType.SWIZZLE_32_4_4]
SKIP = [False, True]
EXPAND = [False, True]  # scaling_type_square_block_and_expand (the 32x32 square-block mxfp8 cast)
ROUNDING = [RoundingMode.RTNE, RoundingMode.STOCHASTIC]
OUTER = ["none", "scalar", "per_token"]  # None / per-tensor scalar / per-token (M, 1)
OUTER_BI = ["none", "pair"]  # None / (dim_k, dim_m) tuple of per-tensor scalars
RHT = ["none", "rht"]  # None / 16x16 Hadamard
RHT_BI = ["none", "dim_m"]  # None / (None, rht) -- RHT applies to the dim-m operand only

# --- how each axis value renders in the table -----------------------------------------------------
SCALING_R = {ScalingType.BlockWise1x16: "1x16", ScalingType.BlockWise1x32: "1x32"}
ORIENT_R = {"dim_k": "NT", "dim_m": "TR"}
SWIZZLE_R = {SwizzleType.NO_SWIZZLE: "NONE", SwizzleType.SWIZZLE_32_4_4: "32_4_4"}
SKIP_R = {False: "no", True: "yes"}
EXPAND_R = {False: "no", True: "yes"}
ROUNDING_R = {RoundingMode.RTNE: "RTNE", RoundingMode.STOCHASTIC: "RS"}
RHT_R = {"none": "None", "rht": "16×16"}


def _format_label(qdata_dtype, inner_scale_calc, outer_variant, rht_variant):
    """Derive the human format name from the arguments of an ACCEPTED `quantize_tensor` combination."""
    if qdata_dtype == torch.float4_e2m1fn_x2:
        if inner_scale_calc == InnerScaleCalc.RCEIL_E8M0:
            return "mxfp4"
        if outer_variant == "per_token":
            return "nvfp4 (per-token)"
        if rht_variant == "rht":
            return "nvfp4 (per-tensor, RHT)"
        return "nvfp4 (per-tensor)"
    return "mxfp8"


_last = {}  # set by the dispatch probes below to the recipe the last call routed to


def _install_dispatch_probes():
    """Wrap every recipe callable imported into `api` / `moe_utils` so a call records (name, status)
    in `_last`.

    This is how the "dispatches to" / "status" columns are derived from the actual code path rather
    than hardcoded: whichever recipe the entry point calls first sets `_last` before returning. Both
    module namespaces are patched because the grouped entry points reach `mxfp8_f` through
    `moe_utils.quantize_2d_act`, which holds its own binding to it.

    We record the FIRST recipe entered, not the last: the quantizer runs before any scale-blocking
    helper (`_to_blocked_*`, which is itself a gold recipe), so first-wins pins "dispatches to" to the
    quantizer rather than the post-quantize swizzle.
    """
    for module in (api, moe_utils):
        for name, obj in list(vars(module).items()):
            mod = getattr(obj, "__module__", "") or ""
            is_kernel = "quant_cast_triton" in mod
            is_ref = "quant_cast_gold" in mod
            if not (callable(obj) and (is_kernel or is_ref)):
                continue
            status = "kernel" if is_kernel else "reference"

            def make(orig, nm, st):
                def wrapper(*args, **kwargs):
                    if "name" not in _last:  # first recipe entered wins (the quantizer)
                        _last["name"], _last["status"] = nm, st
                    return orig(*args, **kwargs)

                return wrapper

            setattr(module, name, make(obj, name, status))


def _probe(thunk):
    """Run one probe call; return (status, dispatch) if the API accepted it, else None."""
    _last.clear()
    try:
        thunk()
    except Exception:
        return None  # frontend (or a downstream guard) rejected this combination -> unsupported
    return _last["status"], f"`{_last['name']}`"


def _build_inputs():
    """A real 2D CUDA input tensor plus the aux tensors (outer scale / RHT) keyed by variant."""
    dev = "cuda"
    x = torch.randn(256, 512, dtype=torch.bfloat16, device=dev)
    sign = torch.tensor([1, -1] * 8, device=dev, dtype=x.dtype)  # fixed +/-1 sign vector
    outer = {
        "none": None,
        "scalar": nvfp4_gs_scale(x),
        "per_token": nvfp4_gs_per_token_scale(x),
    }
    rht = {"none": None, "rht": hadamard_rht_matrix(sign, x.device, x.dtype)}
    key = prng.key(0, device=dev)
    return x, outer, rht, key


def _build_grouped_inputs():
    """A 2D grouped `(total_M, C)` input plus block-aligned `offs` (two 128-row token groups)."""
    dev = "cuda"
    gx = torch.randn(256, 512, dtype=torch.bfloat16, device=dev)
    offs = torch.tensor([128, 256], dtype=torch.int32, device=dev)
    return gx, offs


def collect_quantize_tensor() -> list[tuple[str, ...]]:
    x, outer, rht, key = _build_inputs()
    rows = set()
    for qd, isc, st, orient, sw, ex, rm, osv, rhv in itertools.product(
        QDATA, INNER, SCALING, ORIENT, SWIZZLE, EXPAND, ROUNDING, OUTER, RHT
    ):
        # The outer scaling LEVEL is named explicitly now: bare (single-level) / [inner, TensorWise]
        # (per-tensor) / [inner, RowWise] (per-token), keyed off the OUTER axis 1:1 with outer_scale.
        st_arg = st if osv == "none" else [st, ScalingType.TensorWise if osv == "scalar" else ScalingType.RowWise]
        # dim-m is requested by passing a transposed view of the contiguous input (no orientation arg).
        probe_in = x if orient == "dim_k" else x.transpose(-2, -1)
        res = _probe(lambda: quantize_tensor(
            probe_in,
            qdata_dtype=qd,
            inner_scale_calc=isc,
            scaling_type=st_arg,
            swizzle_type=sw,
            scaling_type_square_block_and_expand=ex,
            qdata_rounding_mode=rm,
            random_key=key if rm == RoundingMode.STOCHASTIC else None,
            outer_scale=outer[osv],
            rht_tensor=rht[rhv],
        ))
        if res is None:
            continue
        scl_tp = SCALING_R[st] if osv == "none" else f"{SCALING_R[st]}+{'TW' if osv == 'scalar' else 'RW'}"
        rows.add((
            _format_label(qd, isc, osv, rhv),
            scl_tp, ORIENT_R[orient], SWIZZLE_R[sw], EXPAND_R[ex], ROUNDING_R[rm], RHT_R[rhv], *res,
        ))
    return sorted(rows)  # lexicographic order -> byte-stable output


def collect_dual() -> list[tuple[str, ...]]:
    x, outer, rht, key = _build_inputs()
    # outer_scale / rht_tensor are per-orientation (dim_k, dim_m) tuples here. Without an RHT both
    # orientations share the scalar (|input.t()| == |input|); the RHT applies to dim-m only.
    outer_bi = {"none": None, "pair": (outer["scalar"], outer["scalar"])}
    rht_bi = {"none": None, "dim_m": (None, rht["rht"])}
    rows = set()
    for qd, isc, st, sw, skip, ex, rm, osv, rhv in itertools.product(
        QDATA, INNER, SCALING, SWIZZLE, SKIP, EXPAND, ROUNDING, OUTER_BI, RHT_BI
    ):
        # dual nvfp4 is per-tensor only; name the outer level TensorWise when outer_scale is set.
        st_arg = st if osv == "none" else [st, ScalingType.TensorWise]
        res = _probe(lambda: quantize_tensor_dual(
            x,
            qdata_dtype=qd,
            inner_scale_calc=isc,
            scaling_type=st_arg,
            swizzle_type=sw,
            skip_transposed_qdata=skip,
            scaling_type_square_block_and_expand=ex,
            qdata_rounding_mode=rm,
            random_key=key if rm == RoundingMode.STOCHASTIC else None,
            outer_scale=outer_bi[osv],
            rht_tensor=rht_bi[rhv],
        ))
        if res is None:
            continue
        fmt = "mxfp8" if qd == torch.float8_e4m3fn else (
            "nvfp4 (per-tensor, RHT)" if rhv == "dim_m" else "nvfp4 (per-tensor)"
        )
        scl_tp = SCALING_R[st] if osv == "none" else f"{SCALING_R[st]}+TW"
        rows.add((fmt, scl_tp, SWIZZLE_R[sw], SKIP_R[skip], EXPAND_R[ex], ROUNDING_R[rm], *res))
    return sorted(rows)


def collect_grouped() -> list[tuple[str, ...]]:
    _, outer, rht, key = _build_inputs()  # reuse the 2D-shaped scalar/per-token/RHT aux tensors
    gx, offs = _build_grouped_inputs()
    rows = set()
    for qd, isc, st, orient, sw, rm, osv, rhv in itertools.product(
        QDATA, INNER, SCALING, ORIENT, SWIZZLE, ROUNDING, OUTER, RHT
    ):
        probe_in = gx if orient == "dim_k" else gx.transpose(-2, -1)
        res = _probe(lambda: quantize_tensor_grouped(
            probe_in, offs,
            qdata_dtype=qd,
            inner_scale_calc=isc,
            scaling_type=st,
            swizzle_type=sw,
            qdata_rounding_mode=rm,
            random_key=key if rm == RoundingMode.STOCHASTIC else None,
            outer_scale=outer[osv],
            rht_tensor=rht[rhv],
        ))
        if res is None:
            continue
        # grouped is mxfp8-only (fp4 qdata is rejected), so the format is always mxfp8.
        rows.add(("mxfp8", SCALING_R[st], ORIENT_R[orient], SWIZZLE_R[sw], ROUNDING_R[rm], *res))
    return sorted(rows)


def collect_grouped_dual() -> list[tuple[str, ...]]:
    _, outer, rht, key = _build_inputs()
    gx, offs = _build_grouped_inputs()
    rows = set()
    for qd, isc, st, sw, skip, rm, osv, rhv in itertools.product(
        QDATA, INNER, SCALING, SWIZZLE, SKIP, ROUNDING, OUTER, RHT
    ):
        res = _probe(lambda: quantize_tensor_grouped_dual(
            gx, offs,
            qdata_dtype=qd,
            inner_scale_calc=isc,
            scaling_type=st,
            swizzle_type=sw,
            skip_transposed_qdata=skip,
            qdata_rounding_mode=rm,
            random_key=key if rm == RoundingMode.STOCHASTIC else None,
            outer_scale=outer[osv],
            rht_tensor=rht[rhv],
        ))
        if res is None:
            continue
        rows.add(("mxfp8", SCALING_R[st], SWIZZLE_R[sw], SKIP_R[skip], ROUNDING_R[rm], *res))
    return sorted(rows)


def render_table(columns: list[str], rows: list[tuple[str, ...]], with_legend: bool) -> str:
    status_idx = len(columns) - 2  # status is always the second-to-last column
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = "\n".join(
        "| " + " | ".join(f"{STATUS_EMOJI[c]} {c}" if i == status_idx else c for i, c in enumerate(row)) + " |"
        for row in rows
    )
    parts = [header, sep, body]
    if with_legend:
        hdr_legend = ", ".join(f"`{abbr}` = `{full}`" for abbr, full in HEADER_ABBR.items())
        val_legend = ", ".join(f"`{abbr}` = `{full}`" for abbr, full in VALUE_ABBR.items())
        parts += ["", f"**Header abbreviations:** {hdr_legend}.", "", f"**Value abbreviations:** {val_legend}."]
    return "\n".join(parts)


def main() -> None:
    assert torch.cuda.is_available(), "gen_support_matrix probes the real API on CUDA; no device found"
    _install_dispatch_probes()
    sections = [
        # (marker tag, columns, collector, emit the abbreviation legend under this table)
        ("support-matrix", COLUMNS, collect_quantize_tensor, True),
        ("support-matrix-dual", COLUMNS_BI, collect_dual, False),
        ("support-matrix-grouped", COLUMNS_GROUPED, collect_grouped, False),
        ("support-matrix-grouped-dual", COLUMNS_GROUPED_BI, collect_grouped_dual, False),
    ]
    readme = Path(__file__).with_name("README.md")
    text = readme.read_text()
    total = 0
    for tag, columns, collect, with_legend in sections:
        rows = collect()
        total += len(rows)
        begin, end = _begin(tag), _end(tag)
        start, stop = text.index(begin), text.index(end) + len(end)
        block = f"{begin}\n\n{render_table(columns, rows, with_legend)}\n\n{end}"
        text = text[:start] + block + text[stop:]
    readme.write_text(text)
    print(f"wrote {total} supported combinations across {len(sections)} entry points to {readme.name}")


if __name__ == "__main__":
    main()
