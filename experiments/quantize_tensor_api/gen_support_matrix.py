"""Generate the `quantize_tensor` support matrix in README.md by PROBING the real API.

This does NOT hardcode the supported rows -- it inspects the actual dispatch in `api.py` by running
every argument combination through `quantize_tensor` on real CUDA tensors and keeping the ones the
frontend accepts (i.e. that don't raise). For each accepted combination it also records the function
the call dispatched to (kernel = a `quant_cast_triton` recipe, reference = a `quant_cast_gold` one),
captured by wrapping those recipe callables in the `api` module namespace.

    python experiments/quantize_tensor_api/gen_support_matrix.py

Requires a CUDA device. It rewrites the region between the BEGIN/END GENERATED markers in README.md
in place. Rows are emitted in deterministic lexicographic order, so the output is byte-stable
(rerunning without an `api.py` dispatch change is a no-op).
"""

import itertools
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; render straight to a PNG
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.func._random as prng  # noqa: E402

# Put the repo root on sys.path so `experiments.*` resolves when run as a script (mirrors api.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.quantize_tensor_api import api  # noqa: E402
from experiments.quantize_tensor_api.api import (
    InnerScaleCalc,
    QuantOrientation,
    RoundingMode,
    ScalingType,
    SwizzleType,
    quantize_tensor,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    hadamard_rht_matrix,
    nvfp4_gs_per_token_scale,
    nvfp4_gs_scale,
)

COLUMNS = [
    "format",
    "scaling_type",
    "orientation",
    "swizzle_type",
    "rounding_mode",
    "outer_scale",
    "rht_tensor",
    "input",
    "status",
    "dispatches to",
]

BEGIN = "<!-- BEGIN GENERATED: support-matrix (python gen_support_matrix.py) -->"
END = "<!-- END GENERATED: support-matrix -->"

# --- per-axis value grids (the argument cross-product we probe) -----------------------------------
QDATA = [torch.float8_e4m3fn, torch.float4_e2m1fn_x2]
INNER = [InnerScaleCalc.E8M0_RCEIL, InnerScaleCalc.E4M3_NVFP4]
SCALING = [ScalingType.BlockWise1x16, ScalingType.BlockWise1x32, ScalingType.BlockWise32x32]
ORIENT = [QuantOrientation.NATURAL, QuantOrientation.TRANSPOSED]
SWIZZLE = [SwizzleType.NO_SWIZZLE, SwizzleType.SWIZZLE_32_4_4]
ROUNDING = [RoundingMode.RTNE, RoundingMode.STOCHASTIC]
OUTER = ["none", "scalar", "per_token"]  # None / per-tensor scalar / per-token (M, 1)
RHT = ["none", "rht"]  # None / 16x16 Hadamard
INPUT_DIM = ["2d", "3d"]

# --- how each axis value renders in the table -----------------------------------------------------
SCALING_R = {ScalingType.BlockWise1x16: "1x16", ScalingType.BlockWise1x32: "1x32", ScalingType.BlockWise32x32: "32x32"}
SWIZZLE_R = {SwizzleType.NO_SWIZZLE: "NO_SWIZZLE", SwizzleType.SWIZZLE_32_4_4: "32_4_4"}
OUTER_R = {"none": "None", "scalar": "scalar", "per_token": "`(M,1)`"}
RHT_R = {"none": "None", "rht": "16×16"}
INPUT_R = {"2d": "2D", "3d": "3D `(E,N,K)`"}


def _format_label(qdata_dtype, inner_scale_calc, outer_variant, rht_variant):
    """Derive the human format name from the arguments of an ACCEPTED combination."""
    if qdata_dtype == torch.float4_e2m1fn_x2:
        if inner_scale_calc == InnerScaleCalc.E8M0_RCEIL:
            return "mxfp4"
        if outer_variant == "per_token":
            return "nvfp4 (per-token)"
        if rht_variant == "rht":
            return "nvfp4 (per-tensor, RHT)"
        return "nvfp4 (per-tensor)"
    return "mxfp8"


_last = {}  # set by the dispatch probes below to the recipe the last call routed to


def _install_dispatch_probes():
    """Wrap every recipe callable imported into `api` so a call records (name, status) in `_last`.

    This is how the "dispatches to" / "status" columns are derived from the actual code path rather
    than hardcoded: whichever recipe `quantize_tensor` calls sets `_last` before returning.
    """
    for name, obj in list(vars(api).items()):
        mod = getattr(obj, "__module__", "") or ""
        is_kernel = "quant_cast_triton" in mod
        is_ref = "quant_cast_gold" in mod
        if not (callable(obj) and (is_kernel or is_ref)):
            continue
        status = "kernel" if is_kernel else "reference"

        def make(orig, nm, st):
            def wrapper(*args, **kwargs):
                _last["name"], _last["status"] = nm, st
                return orig(*args, **kwargs)

            return wrapper

        setattr(api, name, make(obj, name, status))


def _build_inputs():
    """Real CUDA tensors, one per input dim, plus the aux tensors keyed by (variant, dim)."""
    dev = "cuda"
    inputs = {
        "2d": torch.randn(256, 512, dtype=torch.bfloat16, device=dev),
        "3d": torch.randn(4, 256, 512, dtype=torch.bfloat16, device=dev),
    }
    outer = {}
    rht = {}
    for dim, x in inputs.items():
        outer[("none", dim)] = None
        outer[("scalar", dim)] = nvfp4_gs_scale(x)
        # per-token (M, 1) only makes sense for 2D; a 3D probe is rejected on the dim check first, so
        # any (E, 1) placeholder is fine there.
        outer[("per_token", dim)] = (
            nvfp4_gs_per_token_scale(x) if x.dim() == 2 else torch.ones(x.shape[0], 1, dtype=torch.float32, device=dev)
        )
        rht[("none", dim)] = None
        sign = torch.tensor([1, -1] * 8, device=dev, dtype=x.dtype)  # fixed +/-1 sign vector
        rht[("rht", dim)] = hadamard_rht_matrix(sign, x.device, x.dtype)
    key = prng.key(0, device=dev)
    return inputs, outer, rht, key


def collect_rows() -> list[tuple[str, ...]]:
    assert torch.cuda.is_available(), "gen_support_matrix probes the real API on CUDA; no device found"
    _install_dispatch_probes()
    inputs, outer, rht, key = _build_inputs()

    rows = set()
    for qd, isc, st, orient, sw, rm, osv, rhv, dim in itertools.product(
        QDATA, INNER, SCALING, ORIENT, SWIZZLE, ROUNDING, OUTER, RHT, INPUT_DIM
    ):
        _last.clear()
        try:
            quantize_tensor(
                inputs[dim],
                qdata_dtype=qd,
                inner_scale_calc=isc,
                scaling_type=st,
                orientation=orient,
                swizzle_type=sw,
                rounding_mode=rm,
                random_key=key if rm == RoundingMode.STOCHASTIC else None,
                outer_scale=outer[(osv, dim)],
                rht_tensor=rht[(rhv, dim)],
            )
        except Exception:
            continue  # frontend (or a downstream guard) rejected this combination -> unsupported
        rows.add((
            _format_label(qd, isc, osv, rhv),
            SCALING_R[st],
            orient.name,
            SWIZZLE_R[sw],
            rm.name,
            OUTER_R[osv],
            RHT_R[rhv],
            INPUT_R[dim],
            _last["status"],
            f"`{_last['name']}`",
        ))
    return sorted(rows)  # lexicographic order -> byte-stable output


def render_chart(rows: list[tuple[str, ...]]) -> Path:
    """Render the probed rows as a monochrome matplotlib table image with vertical column headers
    and a small font, and return the PNG path (next to this file / the README)."""
    ncols, nrows = len(COLUMNS), len(rows)
    fig, ax = plt.subplots(figsize=(ncols * 1.0, nrows * 0.3 + 2.0))
    ax.axis("off")

    tbl = ax.table(
        cellText=[list(r) for r in rows],
        colLabels=COLUMNS,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.auto_set_column_width(range(ncols))

    # Rotate the header row (keyed (0, col)) 90 degrees and give it room to fit the upright labels.
    for col in range(ncols):
        cell = tbl[0, col]
        cell.set_height(cell.get_height() * 4)
        text = cell.get_text()
        text.set_rotation(90)
        text.set_verticalalignment("bottom")
        text.set_horizontalalignment("center")

    png_path = Path(__file__).with_name("support_matrix.png")
    # dpi + tight bbox for a crisp crop; drop the version-stamped metadata so re-renders are stable.
    fig.savefig(png_path, dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return png_path


def main() -> None:
    rows = collect_rows()
    png_path = render_chart(rows)
    readme = Path(__file__).with_name("README.md")
    text = readme.read_text()
    start, end = text.index(BEGIN), text.index(END) + len(END)
    block = f"{BEGIN}\n\n![quantize_tensor support matrix]({png_path.name})\n\n{END}"
    readme.write_text(text[:start] + block + text[end:])
    print(f"wrote {len(rows)} supported combinations to {png_path.name} + {readme.name}")


if __name__ == "__main__":
    main()
