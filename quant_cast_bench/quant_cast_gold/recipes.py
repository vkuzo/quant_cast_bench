"""Golden single-kernel quant-cast reference recipes, decoupled from flexquant_v3.

Each `QuantCastSingleKernelGold` pairs a plain-PyTorch reference kernel (`pt_ref_fn`,
the function that would be handed to flex_tile_map as `f`) with a `correctness_fn`
that asserts a candidate set of outputs is "close enough" to `pt_ref_fn`'s own
semantics. This package is intentionally independent of flexquant_v3 -- it must not
import from it, since it exists to grade it. flexquant_v3/recipes.py imports these
gold objects and wraps them as `RecipeV2` (adding the flex_tile_map tiling metadata).
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.func._random as prng

from .utils import f32_to_f4_unpacked, f4_unpacked_to_f32, pack_uint4, unpack_uint4

# Optional hardware fp4 encode: PyTorch core exposes an `inline_asm_elementwise` higher-order op
# (torch >= 2.12) that can emit `cvt.rn.satfinite.e2m1x2.f32` on Blackwell (SM100+). It only works
# under torch.compile / fake-tensor tracing (the eager JITerator backend rejects the fp32->uint8
# dtype change), so recipes gate its use behind `torch.compiler.is_compiling() or is_fake(...)` and
# fall back to the pure-PyTorch bit-math path otherwise. Mirrors torchao's `_to_mx_rceil`.
try:
    from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
    from torch._subclasses.fake_tensor import is_fake

    _HAS_INLINE_ASM = True
except Exception:  # pragma: no cover - older torch
    _HAS_INLINE_ASM = False

_SM100 = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
)


def _f32_to_packed_fp4(data_scaled):
    """fp32 (already clamped to +-6) -> packed nvfp4 bytes (two e2m1 per byte; even index -> low
    nibble, odd -> high nibble, matching `pack_uint4(f32_to_f4_unpacked(...))`).

    Under torch.compile on SM100+, uses the hardware `cvt.rn.satfinite.e2m1x2.f32` via the core
    `inline_asm_elementwise` HOP; otherwise falls back to the pure-PyTorch bit-math encode.
    """
    if _HAS_INLINE_ASM and _SM100 and data_scaled.is_cuda and (
        torch.compiler.is_compiling() or is_fake(data_scaled)
    ):
        even = data_scaled[..., 0::2].contiguous().to(torch.float32)  # -> low nibble ($2)
        odd = data_scaled[..., 1::2].contiguous().to(torch.float32)   # -> high nibble ($1)
        # cvt d, HI, LO : first source packs into the high nibble, second into the low nibble.
        packed_u16 = inline_asm_elementwise(
            odd,
            even,
            asm_str=(
                "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $1, $2; cvt.u16.u8 $0, t; }"
            ),
            constraints="=h,r,r",
            dtype=torch.uint16,
        )
        return packed_u16.to(torch.uint8)
    return pack_uint4(f32_to_f4_unpacked(data_scaled))


@dataclass(frozen=True)
class QuantCastSingleKernelGold:
    """A golden single-kernel quant-cast reference.

    `pt_ref_fn(*inputs, **kwargs) -> outputs` is a plain-PyTorch reference function 

    `correctness_fn(inputs, outputs) -> None`
      - inputs - inputs to pt_ref_fn
      - outputs - outputs from an implementation of pt_ref_fn

      The function checks that the outputs are valid, and asserts with an error
      message if they are not. For example, if `pt_ref_fn` quantizes a tensor,
      `correctness_fn` could check SQNR between ref and quantized outputs.

    `example_input_fn(M, K) -> (x, *aux)` builds one representative set of positional
      inputs for `pt_ref_fn` at the given (rows, cols): the tensor `x` plus any extra args
      the recipe takes (a precalculated scale, a bias, an RHT matrix, a PRNG key).

    `perf_description` is a free-form note about the recipe's performance characteristics
      (filled in manually per recipe); surfaced in the benchmark results.
    """

    pt_ref_fn: Callable
    correctness_fn: Callable
    example_input_fn: Callable[[int, int], Tuple[torch.Tensor, ...]]
    perf_description: str


def _compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # torchao's `compute_error` (quantization/utils.py:63) -- SQNR in dB. Duplicated
    # locally (not imported from flexquant_v3/test.py) so gold has no dependency on the
    # thing it grades.
    return 20 * torch.log10(
        torch.linalg.vector_norm(x) / torch.linalg.vector_norm(x - y)
    )


# ---------------------------------------------------------------------------
# Golden recipe: deepseek fp8 1x128.
# ---------------------------------------------------------------------------
def deepseek_1x128_f(x, **kwargs):  # kwargs: framework-supplied global_row/global_col/num_col (unused)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    *lead, last = x.shape
    x_b = x.reshape(*lead, last // 128, 128)
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    scale = (amax / fp8_max).to(torch.float32)  # forward scale
    qdata = (x_b.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return qdata.reshape(*lead, last), scale.squeeze(-1)


def deepseek_1x128_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _deepseek_1x128_correctness, and importable
    # directly by consumers (e.g. flexquant_v3's own Recipe.dequant) that need the inverse.
    M, N = q.shape
    nb = N // 128
    return (q.float().reshape(M, nb, 128) * scale.reshape(M, nb, 1)).reshape(M, N)


def _deepseek_1x128_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = deepseek_1x128_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"deepseek_1x128: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Deepseek1x128Gold = QuantCastSingleKernelGold(
    pt_ref_fn=deepseek_1x128_f,
    correctness_fn=_deepseek_1x128_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(1,128) block",
)


# ---------------------------------------------------------------------------
# Golden recipe: deepseek fp8 128x128.
# ---------------------------------------------------------------------------
def deepseek_128x128_f(x, **kwargs):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    *lead, d1, d2 = x.shape
    n1, n2 = d1 // 128, d2 // 128
    x_b = (
        x.reshape(*lead, n1, 128, n2, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, n1, n2, 128 * 128)
    )
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    scale = (amax / fp8_max).to(torch.float32)  # forward scale
    qdata_b = (x_b.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    qdata = (
        qdata_b.reshape(*lead, n1, n2, 128, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, d1, d2)
    )
    return qdata, scale.squeeze(-1)


def deepseek_128x128_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _deepseek_128x128_correctness, and importable
    # directly by consumers (e.g. flexquant_v3's own Recipe.dequant) that need the inverse.
    M, N = q.shape
    n1, n2 = M // 128, N // 128
    return (q.float().reshape(n1, 128, n2, 128) * scale.reshape(n1, 1, n2, 1)).reshape(M, N)


def _deepseek_128x128_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = deepseek_128x128_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"deepseek_128x128: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Deepseek128x128Gold = QuantCastSingleKernelGold(
    pt_ref_fn=deepseek_128x128_f,
    correctness_fn=_deepseek_128x128_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(128,128) block",
)


# ---------------------------------------------------------------------------
# Golden recipe: deepseek fp8 1x128, reduced across M (128x1 blocks), transposed output.
# ---------------------------------------------------------------------------
def deepseek_1x128_dim_m_f(x, **kwargs):
    """dim-M deepseek: reduce over dim0 (128x1 blocks along rows), then write the tile's outputs
    TRANSPOSED locally. Pair with OutputKind.SWAP_TILE_INDEX on both outputs so the framework
    places each transposed tile at the swapped grid position -> the full (K, M) layout.

    Inlined from deepseek_1x128_f but reducing the other axis: reshape rows into 128-blocks and
    amax over dim1 (the 128 within-block dim), giving a (M//128, N) scale; transpose both outputs
    to (N, M) / (N, M//128) so a tile computed at grid [m, n] carries (bn, bm)-shaped data.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    M, N = x.shape
    x_b = x.reshape(M // 128, 128, N)
    amax = x_b.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    scale = (amax / fp8_max).to(torch.float32)  # forward scale, (M//128, 1, N)
    qdata = (x_b.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn).reshape(M, N)
    # write outputs transposed locally; the framework's SWAP_TILE_INDEX handles the grid swap.
    return qdata.t().contiguous(), scale.squeeze(1).t().contiguous()


def _deepseek_1x128_dim_m_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs), transposed back to (M, N), recovers `x` with SQNR above
    threshold. Reuses deepseek_1x128_dq_f -- it works in the (K, M) transposed frame."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = deepseek_1x128_dq_f(qdata, scale).t()
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"deepseek_1x128_dim_m: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Deepseek1x128DimMGold = QuantCastSingleKernelGold(
    pt_ref_fn=deepseek_1x128_dim_m_f,
    correctness_fn=_deepseek_1x128_dim_m_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(128,1) block, t-contig",
)


# ---------------------------------------------------------------------------
# Golden recipe: deepseek fp8 1x128 in BOTH directions in one pass. Reads x as-is (no input
# transpose/contiguous) and reduces it two ways -- dim-K (1x128 blocks along columns) and dim-M
# (128x1 blocks along rows) -- returning FOUR outputs. Models a fused kernel that reads x once and
# emits both the rowwise (dim-K) and transposed (dim-M) fp8 quantizations. NOT built by calling the
# two existing recipe fns (which would read/reshape x twice); the reductions share one fp32 view of x.
# ---------------------------------------------------------------------------
def deepseek_1x128_dim_km_f(x, **kwargs):
    """One pass over x, reducing in both directions. Returns
    (qdata_k (M,N), scale_k (M,N//128), qdata_m (N,M), scale_m (N,M//128)):
    dim-K matches deepseek_1x128_f, dim-M matches deepseek_1x128_dim_m_f (transposed outputs).
    Requires M % 128 == 0 and N % 128 == 0.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    M, N = x.shape
    xf = x.to(torch.float32)  # read x once; both reshapes below are views of this buffer
    # dim-K: 1x128 blocks along the last dim
    xk = xf.reshape(M, N // 128, 128)
    scale_k = xk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fp8_max
    qdata_k = (xk * (1.0 / scale_k)).to(torch.float8_e4m3fn).reshape(M, N)
    # dim-M: 128x1 blocks along rows (reduce over the 128 within-block rows)
    xm = xf.reshape(M // 128, 128, N)
    scale_m = xm.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fp8_max
    qdata_m = (xm * (1.0 / scale_m)).to(torch.float8_e4m3fn).reshape(M, N)
    # dim-M outputs in transposed (N,M)/(N,M//128) layout (matches Deepseek1x128DimMGold);
    # .t().contiguous() is applied to OUTPUTS only, never to the input.
    return (
        qdata_k,
        scale_k.squeeze(-1),
        qdata_m.t().contiguous(),
        scale_m.squeeze(1).t().contiguous(),
    )


def _deepseek_1x128_dim_km_correctness(
    inputs: Tuple[torch.Tensor, ...],
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Both quantizations must dequant back to `x` with SQNR above threshold: the dim-K pair
    directly, the dim-M pair in the transposed frame (reusing deepseek_1x128_dq_f, then .t())."""
    (x,) = inputs
    qk, sk, qm, sm = outputs
    sqnr_k = _compute_error(x.float(), deepseek_1x128_dq_f(qk, sk).float())
    sqnr_m = _compute_error(x.float(), deepseek_1x128_dq_f(qm, sm).t().float())
    threshold = 20.0
    assert sqnr_k > threshold, (
        f"deepseek_1x128_dim_km (dim-k): sqnr={sqnr_k.item():.2f} dB below {threshold} dB"
    )
    assert sqnr_m > threshold, (
        f"deepseek_1x128_dim_km (dim-m): sqnr={sqnr_m.item():.2f} dB below {threshold} dB"
    )


Deepseek1x128DimKmGold = QuantCastSingleKernelGold(
    pt_ref_fn=deepseek_1x128_dim_km_f,
    correctness_fn=_deepseek_1x128_dim_km_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(1,128) dim-k + (128,1) dim-m, one pass, t-contig",
)


# ---------------------------------------------------------------------------
# Golden recipe: rowwise fp8 (one scale per row, amax over all columns).
# ---------------------------------------------------------------------------
def rowwise_fp8_f(x, **kwargs):
    """Rowwise fp8: one fp32 scale per row (amax over all columns). Tile must span all columns."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).to(torch.float32)  # (M, 1)
    scale = (amax / fp8_max).to(torch.float32)
    qdata = (x.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return qdata, scale  # scale shape (M, 1)


def rowwise_fp8_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _rowwise_fp8_correctness, and importable
    # directly by consumers that need the inverse.
    return q.float() * scale  # scale (M, 1) broadcasts over columns


def _rowwise_fp8_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = rowwise_fp8_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"rowwise_fp8: sqnr={sqnr.item():.2f} dB below {threshold} dB"


RowwiseFp8Gold = QuantCastSingleKernelGold(
    pt_ref_fn=rowwise_fp8_f,
    correctness_fn=_rowwise_fp8_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(1,-1) block",
)


# ---------------------------------------------------------------------------
# Golden recipe: colwise fp8 (one scale per column, amax over all rows).
# ---------------------------------------------------------------------------
def colwise_fp8_f(x, **kwargs):
    """Colwise fp8: one fp32 scale per column (amax over all rows). Tile must span all rows.

    Writes both outputs transposed locally (q -> (N, M), scale -> (N, 1)); pair with
    output_kinds=SWAP_TILE_INDEX so the framework's grid swap yields the transposed (N, M) layout.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12).to(torch.float32)  # (1, N)
    scale = (amax / fp8_max).to(torch.float32)
    qdata = (x.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return qdata.t().contiguous(), scale.t().contiguous()  # (N, M), (N, 1)


def colwise_fp8_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _colwise_fp8_correctness, and importable
    # directly by consumers that need the inverse. q is (N, M) transposed frame; scale
    # (N, 1) broadcasts over q's columns (the original rows).
    return q.float() * scale


def _colwise_fp8_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs), transposed back to (M, N), recovers `x` with SQNR above
    threshold. outputs are in the transposed (N, M) frame."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = colwise_fp8_dq_f(qdata, scale).t()
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"colwise_fp8: sqnr={sqnr.item():.2f} dB below {threshold} dB"


ColwiseFp8Gold = QuantCastSingleKernelGold(
    pt_ref_fn=colwise_fp8_f,
    correctness_fn=_colwise_fp8_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(-1,1) block, t-contig",
)


# ---------------------------------------------------------------------------
# Golden recipe: rowwise fp8 with a PRECALCULATED per-row scale (AuxKind.ROW).
#
# Unlike rowwise_fp8_f (which reduces the row itself), here the (M, 1) per-row scale is
# computed OUTSIDE and passed as an explicit second positional arg (an AuxKind.ROW aux
# input under flex_tile_map). `pt_ref_fn` only does the divide; `rowwise_precalc_scale`
# is the row-reduction that lives outside flex_tile_map, so `inputs = (x, scale)` carries
# the precomputed scale through to correctness_fn (outputs has no scale to recover it from).
# ---------------------------------------------------------------------------
def rowwise_precalc_scale(x):
    """Per-row fp32 scale (row reduction; computed outside flex_tile_map). Returns (M, 1)."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).to(torch.float32)
    return (amax / fp8_max).to(torch.float32)  # (M, 1)


def rowwise_precalc_f(x, scale, **kwargs):
    """Rowwise fp8 cast given a precalculated (M, 1) per-row `scale` (AuxKind.ROW aux input)."""
    qdata = (x.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return (qdata,)  # scale is an input, not a returned output


def rowwise_precalc_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _rowwise_precalc_correctness, and importable
    # directly by consumers that need the inverse.
    return q.float() * scale  # (M, 1) broadcasts over columns


def _rowwise_precalc_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor]
) -> None:
    """Assert dequant(outputs, using the precalculated scale from `inputs`) recovers `x`
    with SQNR above threshold."""
    x, scale = inputs
    (qdata,) = outputs
    x_hat = rowwise_precalc_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"rowwise_precalc: sqnr={sqnr.item():.2f} dB below {threshold} dB"


def _rowwise_precalc_inputs(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    return (x, rowwise_precalc_scale(x))


RowwisePrecalcGold = QuantCastSingleKernelGold(
    pt_ref_fn=rowwise_precalc_f,
    correctness_fn=_rowwise_precalc_correctness,
    example_input_fn=_rowwise_precalc_inputs,
    perf_description="elementwise",
)


# ---------------------------------------------------------------------------
# Golden recipe: colwise fp8 with a PRECALCULATED per-column scale (AuxKind.COL), transposed
# output. The symmetric partner of the ROW precalc recipe above.
# ---------------------------------------------------------------------------
def colwise_precalc_scale(x):
    """Per-column fp32 scale (col reduction; computed outside flex_tile_map). Returns (1, N)."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12).to(torch.float32)
    return (amax / fp8_max).to(torch.float32)  # (1, N)


def colwise_precalc_f(x, scale, **kwargs):
    """Colwise fp8 cast given a precalculated (1, N) per-column `scale` (AuxKind.COL aux input);
    writes the tile output transposed-contiguous (pair with output_kinds=SWAP_TILE_INDEX)."""
    qdata = (x.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return (qdata.t().contiguous(),)  # (Ntile, Mtile); scale is an input, not a returned output


def colwise_precalc_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _colwise_precalc_correctness, and importable
    # directly by consumers that need the inverse. q is (N, M) transposed frame; scale
    # (1, N) -> transpose to (N, 1) to broadcast over q's cols.
    return q.float() * scale.t()


def _colwise_precalc_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor]
) -> None:
    """Assert dequant(outputs, using the precalculated scale from `inputs`), transposed back
    to (M, N), recovers `x` with SQNR above threshold."""
    x, scale = inputs
    (qdata,) = outputs
    x_hat = colwise_precalc_dq_f(qdata, scale).t()
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"colwise_precalc: sqnr={sqnr.item():.2f} dB below {threshold} dB"


def _colwise_precalc_inputs(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    return (x, colwise_precalc_scale(x))


ColwisePrecalcGold = QuantCastSingleKernelGold(
    pt_ref_fn=colwise_precalc_f,
    correctness_fn=_colwise_precalc_correctness,
    example_input_fn=_colwise_precalc_inputs,
    perf_description="elementwise",
)


# ---------------------------------------------------------------------------
# Golden recipe: mxfp8 with FLOOR rounding (1x32 blocks, e8m0 power-of-two scale).
#
# Tile-invariant like deepseek_1x128 (reduce over 32-element groups along N, no transpose),
# but the scale is an e8m0 (float8_e8m0fnu) power-of-two rather than fp32: derived by
# extracting amax's fp32 exponent via integer bit-ops (FLOOR, no log2), and the cast
# reconstructs the pow2 factor by shifting the biased exponent back into the fp32 exponent
# field.
# ---------------------------------------------------------------------------
def _amax_to_e8m0_floor(amax):
    """amax (any shape) -> e8m0 (float8_e8m0fnu) power-of-two block scale via FLOOR: extract
    amax's fp32 exponent by integer bit-ops (no log2). Shared by the mxfp8-floor recipes."""
    e8m0_exponent_bias = 127
    f32_exp_bias = 127
    mbits_f32 = 23
    f8e4m3_max_pow2 = 8
    e8m0_nan = 255

    max_abs = amax.to(torch.float32)
    max_abs_int32 = max_abs.view(torch.int32)
    extracted_pow2 = ((max_abs_int32 >> mbits_f32) & 0xFF) - f32_exp_bias
    scale_unbiased = extracted_pow2 - f8e4m3_max_pow2
    scale_unbiased = torch.clamp(
        scale_unbiased, -e8m0_exponent_bias, e8m0_exponent_bias + 1
    )
    scale_biased = (scale_unbiased + e8m0_exponent_bias).to(torch.uint8)
    scale_biased = torch.where(
        torch.isnan(max_abs), torch.full_like(scale_biased, e8m0_nan), scale_biased
    )
    return scale_biased.view(torch.float8_e8m0fnu)


def _e8m0_to_fp32(scale):
    # inverse of the e8m0 cast: e8m0 biased exponent -> fp32 pow2 factor.
    biased_i32 = scale.contiguous().view(torch.uint8).to(torch.int32)
    scale_fp32 = (biased_i32 << 23).view(torch.float32)
    return torch.clamp(scale_fp32, min=2.0**-126)


def mxfp8_floor_f(x, **kwargs):
    *lead, last = x.shape
    x_b = x.reshape(*lead, last // 32, 32)
    amax = x_b.abs().amax(dim=-1, keepdim=True)
    scale_e8m0 = _amax_to_e8m0_floor(amax)
    # cast: reconstruct the fp32 pow2 factor from the e8m0 scale, then divide.
    qdata = (x_b.to(torch.float32) / _e8m0_to_fp32(scale_e8m0)).to(torch.float8_e4m3fn)
    return qdata.reshape(*lead, last), scale_e8m0.squeeze(-1)


def mxfp8_floor_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _mxfp8_floor_correctness, and importable
    # directly by consumers (e.g. flexquant_v3's mxfp8_floor_swizzle_dq_f/MXFP8_BIAS) that
    # need the inverse.
    M, N = q.shape
    nb = N // 32
    s = _e8m0_to_fp32(scale).reshape(M, nb, 1)
    return (q.float().reshape(M, nb, 32) * s).reshape(M, N)


def _mxfp8_floor_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold. The e8m0 pow2 scale
    is coarser than fp32, so the threshold is lower than the fp8 recipes' (20 dB)."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = mxfp8_floor_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 15.0
    assert sqnr > threshold, f"mxfp8_floor: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Mxfp8FloorGold = QuantCastSingleKernelGold(
    pt_ref_fn=mxfp8_floor_f,
    correctness_fn=_mxfp8_floor_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(1,32) block",
)


# ---------------------------------------------------------------------------
# Golden recipe: mxfp8 FLOOR, reduced across M (32x1 blocks), transposed output.
# Same math as mxfp8_floor_f but the 1x32 e8m0 block runs down M; mirrors deepseek_1x128_dim_m_f.
# ---------------------------------------------------------------------------
def mxfp8_floor_dim_m_f(x, **kwargs):
    """dim-M mxfp8 floor: reshape rows into 32-blocks and reduce down M (dim1 of the block view),
    then write both outputs transposed-contiguous (pair with OutputKind.SWAP_TILE_INDEX on both).
    Inlined from mxfp8_floor_f reducing the other axis, like deepseek_1x128_dim_m_f relative to
    deepseek_1x128_f."""
    M, N = x.shape
    x_b = x.reshape(M // 32, 32, N)
    amax = x_b.abs().amax(dim=1, keepdim=True)  # (M//32, 1, N), reduce down M
    scale_e8m0 = _amax_to_e8m0_floor(amax)  # (M//32, 1, N)
    qdata = (
        (x_b.to(torch.float32) / _e8m0_to_fp32(scale_e8m0)).to(torch.float8_e4m3fn).reshape(M, N)
    )
    # write outputs transposed locally; the framework's SWAP_TILE_INDEX handles the grid swap.
    return qdata.t().contiguous(), scale_e8m0.squeeze(1).t().contiguous()  # (N, M), (N, M//32)


def _mxfp8_floor_dim_m_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """dequant works in the (N, M) transposed frame (mxfp8_floor_dq_f reduces the last dim in
    32-blocks); transpose back before comparing to `x`."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = mxfp8_floor_dq_f(qdata, scale).t()
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 15.0
    assert sqnr > threshold, f"mxfp8_floor_dim_m: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Mxfp8FloorDimMGold = QuantCastSingleKernelGold(
    pt_ref_fn=mxfp8_floor_dim_m_f,
    correctness_fn=_mxfp8_floor_dim_m_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(32,1) block, t-contig",
)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR in BOTH directions in one pass. Reads x as-is and reduces it two ways -- dim-K (1x32
# blocks along columns) and dim-M (32x1 blocks along rows) -- returning FOUR outputs. Models a fused
# kernel that reads x once and emits both the rowwise (dim-K) and transposed (dim-M) mxfp8-floor
# quantizations. The mxfp8 analog of deepseek_1x128_dim_km_f (block 32, e8m0 FLOOR scales, divide).
# ---------------------------------------------------------------------------
def mxfp8_floor_dim_km_f(x, **kwargs):
    """One pass over x, reducing in both directions. Returns
    (qdata_k (M,N), scale_k (M,N//32), qdata_m (N,M), scale_m (N,M//32)) with e8m0 scales:
    dim-K matches mxfp8_floor_f, dim-M matches mxfp8_floor_dim_m_f (transposed outputs).
    Requires M % 32 == 0 and N % 32 == 0.
    """
    M, N = x.shape
    xf = x.to(torch.float32)  # read x once; both reshapes below are views of this buffer
    # dim-K: 1x32 blocks along the last dim
    xk = xf.reshape(M, N // 32, 32)
    sk = _amax_to_e8m0_floor(xk.abs().amax(dim=-1, keepdim=True))
    qk = (xk / _e8m0_to_fp32(sk)).to(torch.float8_e4m3fn).reshape(M, N)
    # dim-M: 32x1 blocks along rows (reduce over the 32 within-block rows)
    xm = xf.reshape(M // 32, 32, N)
    sm = _amax_to_e8m0_floor(xm.abs().amax(dim=1, keepdim=True))
    qm = (xm / _e8m0_to_fp32(sm)).to(torch.float8_e4m3fn).reshape(M, N)
    # dim-M outputs in transposed (N,M)/(N,M//32) layout (matches Mxfp8FloorDimMGold);
    # .t().contiguous() is applied to OUTPUTS only, never to the input.
    return qk, sk.squeeze(-1), qm.t().contiguous(), sm.squeeze(1).t().contiguous()


def _mxfp8_floor_dim_km_correctness(
    inputs: Tuple[torch.Tensor, ...],
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Both quantizations must dequant back to `x` with SQNR above threshold: the dim-K pair
    directly, the dim-M pair in the transposed frame (reusing mxfp8_floor_dq_f, then .t())."""
    (x,) = inputs
    qk, sk, qm, sm = outputs
    sqnr_k = _compute_error(x.float(), mxfp8_floor_dq_f(qk, sk).float())
    sqnr_m = _compute_error(x.float(), mxfp8_floor_dq_f(qm, sm).t().float())
    threshold = 15.0
    assert sqnr_k > threshold, (
        f"mxfp8_floor_dim_km (dim-k): sqnr={sqnr_k.item():.2f} dB below {threshold} dB"
    )
    assert sqnr_m > threshold, (
        f"mxfp8_floor_dim_km (dim-m): sqnr={sqnr_m.item():.2f} dB below {threshold} dB"
    )


Mxfp8FloorDimKmGold = QuantCastSingleKernelGold(
    pt_ref_fn=mxfp8_floor_dim_km_f,
    correctness_fn=_mxfp8_floor_dim_km_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(1,32) dim-k + (32,1) dim-m, one pass, t-contig",
)


# ---------------------------------------------------------------------------
# Golden recipe: mxfp8 FLOOR with square 32x32 blocks (one e8m0 scale per 32x32 block).
# Block structure mirrors deepseek_128x128_f (32 instead of 128); scale logic is mxfp8_floor's.
# ---------------------------------------------------------------------------
def mxfp8_32x32_floor_f(x, **kwargs):
    *lead, d1, d2 = x.shape
    n1, n2 = d1 // 32, d2 // 32
    x_b = (
        x.reshape(*lead, n1, 32, n2, 32)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, n1, n2, 32 * 32)
    )
    amax = x_b.abs().amax(dim=-1, keepdim=True)  # (..., n1, n2, 1)
    scale_e8m0 = _amax_to_e8m0_floor(amax)  # e8m0, (..., n1, n2, 1)
    qdata_b = (x_b.to(torch.float32) / _e8m0_to_fp32(scale_e8m0)).to(torch.float8_e4m3fn)
    qdata = (
        qdata_b.reshape(*lead, n1, n2, 32, 32)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, d1, d2)
    )
    return qdata, scale_e8m0.squeeze(-1)  # scale (M//32, N//32)


def mxfp8_32x32_floor_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # inverse: un-block the e8m0 scale over the 32x32 grid (mirrors deepseek_128x128_dq_f).
    M, N = q.shape
    n1, n2 = M // 32, N // 32
    s = _e8m0_to_fp32(scale).reshape(n1, 1, n2, 1)
    return (q.float().reshape(n1, 32, n2, 32) * s).reshape(M, N)


def _mxfp8_32x32_floor_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold. e8m0 pow2 scale is coarse,
    so the threshold matches mxfp8_floor's (15 dB), not the fp8 recipes' (20 dB)."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = mxfp8_32x32_floor_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 15.0
    assert sqnr > threshold, f"mxfp8_32x32_floor: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Mxfp832x32FloorGold = QuantCastSingleKernelGold(
    pt_ref_fn=mxfp8_32x32_floor_f,
    correctness_fn=_mxfp8_32x32_floor_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(32,32) block",
)


# ---------------------------------------------------------------------------
# Golden recipe: mxfp8 FLOOR with swizzled (NVIDIA 32x4x4 blocked) scale.
#
# Same quantization as mxfp8_floor_f, but the e8m0 scale is emitted in the blocked layout
# `_scaled_mm` consumes. The swizzle is a LOCAL, tile-invariant transform when tiles are
# whole 128x128 hp units (= 128x4 e8m0 tiles): each 128x4 scale tile swizzles independently
# into a 32x16 block. The scale is returned as the 4D block grid `(n_row_blocks,
# n_col_blocks, 32, 16)` (see `_to_blocked_4d`): keeping the block axes separate makes it
# tile-invariant under BOTH row and column splits -- MANUAL_TILE reassembles a column tile
# with cat(dim=1) and a row tile with cat(dim=0). The final serialization to the flat
# `_scaled_mm` buffer (a global, grid-shape-dependent step) is `.reshape(-1)`, done once
# outside `f` after reassembly.
# ---------------------------------------------------------------------------
def _to_blocked_4d(scale):
    """Swizzle a row-major block-scale (H, W) into NVIDIA's blocked layout, kept as an
    explicit 4D block grid `(n_row_blocks, n_col_blocks, 32, 16)`.

    Ported from flexquant v1 swizzle.py:11-46 (itself a port of torchao's `to_blocked`,
    torchao/prototype/mx_formats/utils.py), but the final `(n_row_blocks*32,
    n_col_blocks*16)` reshape is NOT applied here. Serializing to that 2D buffer folds the
    (row-block, col-block) walk order into the axes, which makes the result depend on the
    GLOBAL grid shape -- a column split then reorders the buffer, so `f` composed with the
    2D swizzle is not tile-invariant. Keeping the two block axes separate makes the swizzle
    tile-invariant: a column tile concatenates on `dim=1` (n_col_blocks), a row tile on
    `dim=0` (n_row_blocks), and `.reshape(-1)` still equals torchao's `to_blocked` buffer
    (do that serialization once, outside `f`, after tiles are reassembled).

    Each 128x4 scale block swizzles independently into a 32x16 block, so this is a LOCAL
    (per-atom) transform: valid only when tiles are whole 128x4 scale atoms.
    """
    def _ceil_div(a, b):
        return (a + b - 1) // b

    rows, cols = scale.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = scale
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=scale.device, dtype=scale.dtype
        )
        padded[:rows, :cols] = scale

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    # (n_row_blocks, n_col_blocks, 128, 4) -> (n_row_blocks, n_col_blocks, 32, 16), keeping
    # the two block axes intact (no reshape across them).
    rearranged = blocks.reshape(n_row_blocks, n_col_blocks, 4, 32, 4).transpose(-3, -2)
    return rearranged.reshape(n_row_blocks, n_col_blocks, 32, 16)


def _from_blocked_4d(blocked, rows, cols):
    """Inverse of `_to_blocked_4d` for the exact case rows % 128 == 0, cols % 4 == 0.

    `blocked` is the 4D block grid `(n_row_blocks, n_col_blocks, 32, 16)`.
    """
    nrb, ncb = rows // 128, cols // 4
    x = blocked.reshape(nrb, ncb, 32, 4, 4).transpose(-3, -2)
    x = x.reshape(nrb, ncb, 128, 4).permute(0, 2, 1, 3)
    return x.reshape(rows, cols)


def mxfp8_floor_swizzle_f(x, **kwargs):
    qdata, scale_e8m0 = mxfp8_floor_f(x)
    return qdata, _to_blocked_4d(scale_e8m0)


def mxfp8_floor_swizzle_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _mxfp8_floor_swizzle_correctness, and importable
    # directly by consumers that need the inverse. Un-swizzle the 4D block grid back to
    # (M, N//32) e8m0, then dequant as mxfp8.
    M, N = q.shape
    rows, cols = M, N // 32
    scale_e8m0 = _from_blocked_4d(scale, rows, cols)
    return mxfp8_floor_dq_f(q, scale_e8m0)


def _mxfp8_floor_swizzle_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs) recovers `x` with SQNR above threshold."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = mxfp8_floor_swizzle_dq_f(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 15.0
    assert sqnr > threshold, f"mxfp8_floor_swizzle: sqnr={sqnr.item():.2f} dB below {threshold} dB"


Mxfp8FloorSwizzleGold = QuantCastSingleKernelGold(
    pt_ref_fn=mxfp8_floor_swizzle_f,
    correctness_fn=_mxfp8_floor_swizzle_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(1,32) block, swizzle",
)


# ---------------------------------------------------------------------------
# Golden recipe: mxfp8 FLOOR reduced across M (32x1 blocks), transposed output, with the e8m0
# scale in the swizzled (NVIDIA 32x4x4 blocked) layout. Combines mxfp8_floor_dim_m_f (dim-M
# reduction -> transposed (N, M) qdata + (N, M//32) scale) with the _to_blocked_4d swizzle applied
# to that transposed scale -- i.e. mxfp8_floor_swizzle_f in the dim-M / transposed frame.
# ---------------------------------------------------------------------------
def mxfp8_floor_dim_m_swizzle_f(x, **kwargs):
    qdata, scale_e8m0 = mxfp8_floor_dim_m_f(x)  # (N, M), (N, M//32)
    return qdata, _to_blocked_4d(scale_e8m0)


def mxfp8_floor_dim_m_swizzle_dq_f(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- inverse for the correctness check / consumers. `q` is (N, M) in
    # the transposed dim-M frame, so its 32-blocks run along the last (M) axis; reuse
    # mxfp8_floor_swizzle_dq_f, which un-swizzles the 4D scale grid and dequants last-dim 32-blocks.
    return mxfp8_floor_swizzle_dq_f(q, scale)


def _mxfp8_floor_dim_m_swizzle_correctness(
    inputs: Tuple[torch.Tensor, ...], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """dequant works in the (N, M) transposed frame (like _mxfp8_floor_dim_m_correctness); transpose
    back before comparing to `x`."""
    (x,) = inputs
    qdata, scale = outputs
    x_hat = mxfp8_floor_dim_m_swizzle_dq_f(qdata, scale).t()
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 15.0
    assert sqnr > threshold, (
        f"mxfp8_floor_dim_m_swizzle: sqnr={sqnr.item():.2f} dB below {threshold} dB"
    )


Mxfp8FloorDimMSwizzleGold = QuantCastSingleKernelGold(
    pt_ref_fn=mxfp8_floor_dim_m_swizzle_f,
    correctness_fn=_mxfp8_floor_dim_m_swizzle_correctness,
    example_input_fn=lambda M, K: (torch.randn(M, K, dtype=torch.bfloat16, device="cuda"),),
    perf_description="(32,1) block, t-contig, swizzle",
)


# ---------------------------------------------------------------------------
# Golden recipe: float8 tensorwise (per-tensor) scaling.
#
# Unlike the block recipes, the scale is a single per-tensor value that needs a GLOBAL
# reduction over the whole tensor -- that reduction is NOT tile-invariant, so it lives
# OUTSIDE flex_tile_map (`float8_tensorwise_scale`). Given that precomputed scale,
# quantization is just dividing every element by one fixed scalar -- identical across
# tiles, hence tile-invariant -- so it runs INSIDE flex_tile_map via an `f` that takes the
# scale as an explicit REPLICATE aux input (handed whole to every tile).
# ---------------------------------------------------------------------------
def float8_tensorwise_scale(x):
    """Per-tensor scale (global reduction; computed outside flex_tile_map)."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    amax = x.abs().amax().clamp(min=1e-12).to(torch.float32)
    return (amax / fp8_max).to(torch.float32)  # scalar forward scale


def float8_tensorwise_f(x, scale, **kwargs):
    """Tile-invariant `f` taking the precomputed per-tensor `scale` as an explicit aux input
    (REPLICATE: the same scalar scale is used for every tile). `scale` is an input, not a
    returned output, so `f` returns a 1-tuple `(qdata,)`."""
    qdata = (x.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return (qdata,)


def dq_tensorwise(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _float8_tensorwise_correctness, and importable
    # directly by consumers that need the inverse.
    return q.float() * scale


def _float8_tensorwise_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor]
) -> None:
    """Assert dequant(outputs, using the precalculated scale from `inputs`) recovers `x`
    with SQNR above threshold."""
    x, scale = inputs
    (qdata,) = outputs
    x_hat = dq_tensorwise(qdata, scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 20.0
    assert sqnr > threshold, f"float8_tensorwise: sqnr={sqnr.item():.2f} dB below {threshold} dB"


def _float8_tensorwise_inputs(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    return (x, float8_tensorwise_scale(x))


Float8TensorwiseGold = QuantCastSingleKernelGold(
    pt_ref_fn=float8_tensorwise_f,
    correctness_fn=_float8_tensorwise_correctness,
    example_input_fn=_float8_tensorwise_inputs,
    perf_description="elementwise",
)


# nvfp4 format constants (fp4 e2m1 max value + e4m3 scale range).
F4_E2M1_MAX = 6.0
F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny


# ---------------------------------------------------------------------------
# Golden recipe: nvfp4 with global scale (two-level) + swizzled inner scale.
#
# A per-tensor fp32 OUTER scale plus a per-16-element e4m3 INNER scale, with fp4-packed
# qdata. The outer scale is a GLOBAL amax reduction -- not tile-invariant -- so it is
# computed OUTSIDE flex_tile_map (`nvfp4_gs_scale`) and passed in as a REPLICATE aux
# input. The inner scale is additionally swizzled into the NVIDIA blocked layout and
# returned as the 4D block grid `(n_row_blocks, n_col_blocks, 32, 16)`, so MANUAL_TILE's
# cat recomposition reproduces the full-tensor swizzle under both row and column splits
# (cf. mxfp8 swizzle).
# ---------------------------------------------------------------------------
def nvfp4_gs_scale(x):
    """Per-tensor fp32 outer scale (global reduction; computed outside flex_tile_map)."""
    outer_amax = x.abs().to(torch.float32).amax()
    return outer_amax / (F8E4M3_MAX * F4_E2M1_MAX)


def nvfp4_gs_swizzle_f(x, outer_scale, **kwargs):
    """Tile-invariant `f` taking the precomputed per-tensor `outer_scale` as an explicit aux
    input (REPLICATE: the same scalar outer scale is used for every tile)."""
    *lead, last = x.shape
    x_b = x.reshape(*lead, last // 16, 16)
    local_amax = x_b.abs().amax(dim=-1, keepdim=True)
    # inner e4m3 block scale, relative to the outer scale.
    inner = torch.clamp(
        (local_amax.to(torch.float32) / F4_E2M1_MAX) / outer_scale,
        min=E4M3_EPS, max=F8E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    # cast: divide by (outer * inner), clamp to fp4 range, pack two per byte.
    reciprocal = (1.0 / outer_scale) / inner.to(torch.float32)
    data_scaled = torch.clamp(x_b.to(torch.float32) * reciprocal, -F4_E2M1_MAX, F4_E2M1_MAX)
    qdata_b = _f32_to_packed_fp4(data_scaled).view(torch.float4_e2m1fn_x2)
    qdata = qdata_b.reshape(*lead, last // 2)
    inner_swizzled = _to_blocked_4d(inner.squeeze(-1))
    return qdata, inner_swizzled


def nvfp4_gs_swizzle_dq_f(q: torch.Tensor, inner_swizzled: torch.Tensor, outer_scale: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _nvfp4_gs_swizzle_correctness, and importable
    # directly by consumers that need the inverse. Takes the per-tensor `outer_scale` as an
    # explicit arg (symmetric with the lifted quant `nvfp4_gs_swizzle_f`).
    # q: (M, N//2) packed fp4; N = 2 * packed cols.
    M, half = q.shape
    N = half * 2
    cols = N // 16  # number of 16-blocks per row (== inner scale cols)
    # unpack fp4 -> fp32 in (M, N).
    unpacked = f4_unpacked_to_f32(unpack_uint4(q.view(torch.uint8))).reshape(M, N)
    # un-swizzle inner scale (4D block grid) back to (M, cols) e4m3 -> fp32.
    inner = _from_blocked_4d(inner_swizzled, M, cols)
    inner_fp32 = inner.to(torch.float32).reshape(M, cols, 1)
    return (unpacked.reshape(M, cols, 16) * inner_fp32 * outer_scale).reshape(M, N)


def _nvfp4_gs_swizzle_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs, using the precalculated outer scale from `inputs`) recovers
    `x` with SQNR above threshold. nvfp4 is 4-bit, coarser than fp8/mxfp8, so a lower floor."""
    x, outer_scale = inputs
    qdata, inner_swizzled = outputs
    x_hat = nvfp4_gs_swizzle_dq_f(qdata, inner_swizzled, outer_scale)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 12.0
    assert sqnr > threshold, f"nvfp4_gs_swizzle: sqnr={sqnr.item():.2f} dB below {threshold} dB"


def _nvfp4_gs_swizzle_inputs(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    return (x, nvfp4_gs_scale(x))


Nvfp4GsSwizzleGold = QuantCastSingleKernelGold(
    pt_ref_fn=nvfp4_gs_swizzle_f,
    correctness_fn=_nvfp4_gs_swizzle_correctness,
    example_input_fn=_nvfp4_gs_swizzle_inputs,
    perf_description="(1,16) block, fp4 qdata, swizzle",
)


# ---------------------------------------------------------------------------
# Golden recipe: nvfp4 with a 128x128-BLOCKED outer scale (instead of a global scalar).
#
# Same two-level nvfp4 as nvfp4_gs_swizzle_f, but the outer scale is one value per 128x128
# block -- shape (M//128, N//128) -- computed outside and passed as an AuxKind.TILE aux. The
# framework hands `f` the sub-block of the outer scale covering the current tile; `f`
# block-broadcasts it to per-element (option-4 pattern: expand+reshape, no materialized (M,N)
# scale). Because 128 is a multiple of the 16-element inner block, the outer is constant within
# each 16-group, so one representative per 16-group aligns with the inner scale.
# ---------------------------------------------------------------------------
def nvfp4_blocked_outer_scale(x, blk=128):
    """Per-128x128-block fp32 outer scale (block reduction; computed outside flex_tile_map).
    Returns shape (M//blk, N//blk)."""
    Mb, Nb = x.shape[0] // blk, x.shape[1] // blk
    block_amax = x.abs().to(torch.float32).reshape(Mb, blk, Nb, blk).amax(dim=(1, 3))
    return block_amax / (F8E4M3_MAX * F4_E2M1_MAX)  # (Mb, Nb)


def nvfp4_blocked_outer_f(x, outer_blocked, **kwargs):
    """Tile-invariant `f`: nvfp4 cast with a 128x128-blocked outer scale (AuxKind.TILE).

    Instead of expanding the outer scale to per-element, reshape `x` so the outer block grid is
    explicit -- (Mb, rows_per_block, Nb, n16_per_block, 16) -- and let `outer_blocked` broadcast
    against it via size-1 axes. Each outer element then maps directly to its block slice of the
    input; the full (M, N) outer scale is never materialized.
    """
    M, N = x.shape
    Mb, Nb = outer_blocked.shape
    rpb, cpb = M // Mb, N // Nb          # rows / cols per outer block (e.g. 128, 128)
    n16 = cpb // 16                      # inner 16-groups per outer block along N
    # block-grid view: last dim is the 16-element inner block.
    x_b = x.reshape(Mb, rpb, Nb, n16, 16)
    outer_b = outer_blocked[:, None, :, None, None]     # (Mb, 1, Nb, 1, 1), broadcasts
    local_amax = x_b.abs().amax(dim=-1, keepdim=True)   # (Mb, rpb, Nb, n16, 1)
    inner = torch.clamp(
        (local_amax.to(torch.float32) / F4_E2M1_MAX) / outer_b,
        min=E4M3_EPS, max=F8E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    reciprocal = (1.0 / outer_b) / inner.to(torch.float32)
    data_scaled = torch.clamp(x_b.to(torch.float32) * reciprocal, -F4_E2M1_MAX, F4_E2M1_MAX)
    qdata = pack_uint4(f32_to_f4_unpacked(data_scaled)).view(torch.float4_e2m1fn_x2).reshape(M, N // 2)
    # inner scale back to (M, N//16) row-major, then swizzle.
    inner_swizzled = _to_blocked_4d(inner.squeeze(-1).reshape(M, N // 16))
    return qdata, inner_swizzled


def nvfp4_blocked_outer_dq_f(q: torch.Tensor, inner_swizzled: torch.Tensor, outer_blocked: torch.Tensor) -> torch.Tensor:
    # not a dataclass field -- used inside _nvfp4_blocked_outer_correctness, and importable
    # directly by consumers that need the inverse. Reshapes onto the outer block grid so
    # `outer_blocked` broadcasts via size-1 axes (no materialized (M, N) outer scale),
    # mirroring the quant.
    M, half = q.shape
    N = half * 2
    Mb, Nb = outer_blocked.shape
    rpb, cpb = M // Mb, N // Nb
    n16 = cpb // 16
    unpacked = f4_unpacked_to_f32(unpack_uint4(q.view(torch.uint8))).reshape(M, N)
    inner = _from_blocked_4d(inner_swizzled, M, N // 16)
    # block-grid view: (Mb, rpb, Nb, n16, 16); outer broadcasts on the block axes.
    data = unpacked.reshape(Mb, rpb, Nb, n16, 16)
    inner_b = inner.to(torch.float32).reshape(Mb, rpb, Nb, n16, 1)
    outer_b = outer_blocked[:, None, :, None, None]
    return (data * inner_b * outer_b).reshape(M, N)


def _nvfp4_blocked_outer_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert dequant(outputs, using the precalculated blocked outer scale from `inputs`)
    recovers `x` with SQNR above threshold."""
    x, outer_blocked = inputs
    qdata, inner_swizzled = outputs
    x_hat = nvfp4_blocked_outer_dq_f(qdata, inner_swizzled, outer_blocked)
    sqnr = _compute_error(x.float(), x_hat.float())
    threshold = 12.0
    assert sqnr > threshold, f"nvfp4_blocked_outer: sqnr={sqnr.item():.2f} dB below {threshold} dB"


def _nvfp4_blocked_outer_inputs(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    return (x, nvfp4_blocked_outer_scale(x))


Nvfp4BlockedOuterGold = QuantCastSingleKernelGold(
    pt_ref_fn=nvfp4_blocked_outer_f,
    correctness_fn=_nvfp4_blocked_outer_correctness,
    example_input_fn=_nvfp4_blocked_outer_inputs,
    perf_description="(1,16) block, fp4 qdata, swizzle",
)


# ---------------------------------------------------------------------------
# Golden recipe: mxfp8 FLOOR with an elementwise bias added before quant.
#
# `bias` is the same shape as the input -> AuxKind.TILE with divisor (1, 1): the framework
# partitions it exactly like the input (one bias element per input element). `f` just adds it
# and runs the existing mxfp8 cast; dequant is the plain mxfp8 dequant (the bias is folded in).
# ---------------------------------------------------------------------------
def mxfp8_bias_f(x, bias, **kwargs):
    """Tile-invariant `f`: add an elementwise `bias` (AuxKind.TILE, per-element) then mxfp8."""
    return mxfp8_floor_f(x + bias.to(x.dtype))


def _mxfp8_bias_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Assert `outputs` has the expected shape/dtype for `x`. `bias` is a fixed ones-tensor
    (not derived from `x`), so an SQNR-against-`x` check doesn't make sense here -- adding 1
    to every element is not a quantization error, it's the recipe's definition. Shape/dtype is
    the only invariant this recipe promises."""
    x, _bias = inputs
    qdata, scale = outputs
    assert qdata.shape == x.shape, f"mxfp8_bias: qdata shape {qdata.shape} != x shape {x.shape}"
    assert qdata.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float8_e8m0fnu


def _mxfp8_bias_inputs(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    return (x, torch.ones_like(x))  # bias is an arbitrary same-shape input; ones is fine


Mxfp8BiasGold = QuantCastSingleKernelGold(
    pt_ref_fn=mxfp8_bias_f,
    correctness_fn=_mxfp8_bias_correctness,
    example_input_fn=_mxfp8_bias_inputs,
    perf_description="",
)

# ---------------------------------------------------------------------------
# Golden recipe: the 16x16 randomized Hadamard transform (RHT). A non-quant example: bf16
# in, bf16 out, NO scale/aux output -- pt_ref_fn returns a 1-tuple `(out,)`. Building block
# for torchao's RHT-fused nvfp4 kernels. RHT = diag(sign) @ H, where H is the 16x16
# Sylvester-Walsh matrix / sqrt(16); mirrors torchao get_rht_matrix. RHT is orthogonal (its
# inverse is its transpose) but, with a sign vector, not an involution. The RHT matrix is
# passed to pt_ref_fn as an explicit input (a REPLICATE aux under flex_tile_map).
# ---------------------------------------------------------------------------
# 16x16 Sylvester-Walsh Hadamard values (torchao hadamard_utils.py get_hadamard_matrix).
_HADAMARD_16 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
    [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
    [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
    [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
    [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
    [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
    [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
    [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
    [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
]


def _hadamard_16_matrix(device, dtype):
    """16x16 Sylvester-Walsh Hadamard matrix scaled by 1/sqrt(16) (orthonormal)."""
    return torch.tensor(_HADAMARD_16, dtype=dtype, device=device) / (16**0.5)


def hadamard_rht_matrix(sign_vector, device, dtype):
    """RHT = diag(sign) @ H (torchao get_rht_matrix). `sign_vector` is a length-16 tensor."""
    H = _hadamard_16_matrix(device, dtype)
    return torch.diag(sign_vector.to(device=device, dtype=dtype)) @ H


def hadamard_rht_f(x, rht, **kwargs):
    """Apply the 16x16 RHT along the last dim. `rht` is the RHT matrix (built via
    `hadamard_rht_matrix`), an explicit input. Returns a 1-tuple `(out,)` -- no scale."""
    *lead, last = x.shape
    out = (x.reshape(*lead, last // 16, 16) @ rht).reshape(*lead, last)
    return (out,)


def _hadamard_rht_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor]
) -> None:
    """RHT is orthogonal, so its transpose inverts it: recover `x` from the transformed output
    (NOT by applying RHT twice) and assert high SQNR. There's no scale to dequant here."""
    x, rht = inputs
    (y,) = outputs
    M, N = x.shape
    x_rec = (y.reshape(M, N // 16, 16) @ rht.t()).reshape(M, N)
    sqnr = _compute_error(x.float(), x_rec.float())
    threshold = 25.0
    assert sqnr > threshold, f"hadamard_rht: roundtrip sqnr={sqnr.item():.2f} dB below {threshold} dB"


def _hadamard_rht_inputs(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    # fixed +/-1 sign vector (deterministic); build the 16x16 RHT matrix pt_ref_fn transforms with.
    sign = torch.tensor([1, -1] * 8, device=x.device, dtype=x.dtype)
    return (x, hadamard_rht_matrix(sign, x.device, x.dtype))


HadamardRht = QuantCastSingleKernelGold(
    pt_ref_fn=hadamard_rht_f,
    correctness_fn=_hadamard_rht_correctness,
    example_input_fn=_hadamard_rht_inputs,
    perf_description="elementwise RHT",
)

# ---------------------------------------------------------------------------
# Golden recipe: stochastic rounding (SR) fp32 -> bf16. A non-quant example. bf16 shares
# fp32's 8-bit exponent, so this is the simplest SR target -- no exponent rebias, no packing,
# no scale, no subnormal edge case: just dither the 16 discarded mantissa bits, then truncate.
# pt_ref_fn returns a 1-tuple `(out,)`; the Philox key is passed as an explicit input (a
# REPLICATE aux under flex_tile_map). SR is unbiased: a value between two bf16 grid points
# rounds up with probability p_up = (x-lo)/(hi-lo), so E[SR(x)] = x.
# ---------------------------------------------------------------------------
def _sr_bf16_dither(x, rand16):
    """Apply a uniform 16-bit dither `rand16` to `x` (fp32) then truncate to bf16.

    dither the 16 mantissa bits fp32->bf16 drops, then truncate them (mask off the low 16
    bits). -65536 == 0xFFFF0000 as int32; .to(bfloat16) is exact since the low bits are zero.
    """
    xi = x.contiguous().view(torch.int32) + rand16
    xi = xi & -65536
    return xi.view(torch.float32).to(torch.bfloat16)


def sr_bf16_f(x, key, **kwargs):
    """fp32 -> bf16 stochastic rounding, keyed on the TILE-LOCAL element layout.

    `key` is a torch.func._random (stateless counter-based Philox) PRNG key, an explicit input
    (a REPLICATE aux under flex_tile_map). One uniform is drawn per element in tile-local order,
    so offsets repeat across tiles and tiling CHANGES the rounding -- NOT tile-invariant, kept
    as the counterexample. Returns `(out,)`.
    """
    assert x.dtype == torch.float32, f"SR bf16 expects fp32 input, got {x.dtype}"
    # uniform [0, 1) per element from the Philox key, scaled to a uniform 16-bit dither.
    u = prng.uniform(key, tuple(x.shape))
    rand16 = (u * (1 << 16)).to(torch.int32)  # uniform int in [0, 2**16)
    return (_sr_bf16_dither(x, rand16),)


def _sr_bf16_unbiased_correctness(
    inputs: Tuple[torch.Tensor, torch.Tensor], outputs: Tuple[torch.Tensor]
) -> None:
    """SR's defining property, checked on a CONSTANT input `x` (value v in [1, 2), where the
    bf16 grid spacing is 2**-7): every output lands on one of the two bracketing bf16 grid
    points, and the mean is ~= v (unbiased). `x` must be constant so the per-element draws share
    the same two neighbors and the mean estimates E[SR(v)]."""
    x, _key = inputs
    (out,) = outputs
    v = x.flatten()[0].item()
    lo = torch.tensor(v, dtype=torch.bfloat16).float().item()  # RTN neighbor (round down)
    hi = torch.tensor(v + 2**-7, dtype=torch.bfloat16).float().item()
    assert lo < v < hi, f"sr_bf16: v={v} not strictly between bf16 neighbors ({lo}, {hi})"
    uniq = set(out.float().unique().tolist())
    assert uniq <= {lo, hi}, f"sr_bf16: unexpected values {uniq - {lo, hi}}"
    assert abs(out.float().mean().item() - v) < 1e-3, "sr_bf16: mean not unbiased"


def _sr_inputs(M, K):
    # SR asserts fp32 (not bf16) and its correctness_fn checks unbiasedness on a CONSTANT value
    # strictly between two bf16 grid points (spacing 2**-7 near 1.0). Shared by both SR variants.
    x = torch.full((M, K), 1.0 + 0.003, dtype=torch.float32, device="cuda")
    return (x, prng.key(0, device=x.device))


SrF32ToBf16 = QuantCastSingleKernelGold(
    pt_ref_fn=sr_bf16_f,
    correctness_fn=_sr_bf16_unbiased_correctness,
    example_input_fn=_sr_inputs,
    perf_description="",
)


def sr_bf16_global_f(x, key, **kwargs):
    """Tiling-INVARIANT fp32 -> bf16 stochastic rounding: keys the dither on each element's
    GLOBAL position in the parent tensor, so the draws don't shift with tiling (the tile-invariant
    counterpart to `sr_bf16_f`, which keys on tile-local order and so is NOT tile-invariant).

    The framework supplies the tile's global origin and row stride via kwargs, read here as
    `global_row`, `global_col`, `num_col`. Each element's global flat index is
    `(global_row + i) * num_col + (global_col + j)`; we build a per-element Philox key
    `[seed, global_index]` (vectorized, no host sync) and draw one uniform each. Because the
    index is global, element (i, j) gets the same draw regardless of which tile it lands in, so
    REFERENCE == MANUAL_TILE bit-for-bit. Returns `(out,)`.
    """
    assert x.dtype == torch.float32, f"SR bf16 expects fp32 input, got {x.dtype}"
    global_row = kwargs["global_row"]
    global_col = kwargs["global_col"]
    num_col = kwargs["num_col"]
    M, N = x.shape
    # per-element global flat index (int64 arithmetic; uint64 mul is unsupported on cuda).
    i = (global_row + torch.arange(M, device=x.device)).view(-1, 1)
    j = (global_col + torch.arange(N, device=x.device)).view(1, -1)
    gidx = (i * num_col + j).reshape(-1).to(torch.int64)
    # per-element Philox key [seed, global_index]; seed = key[0:1] (a slice, not .item(), so this
    # stays traceable / survives the FakeTensor shape-probe).
    seed = key[0:1].to(torch.int64).expand(gidx.numel())
    keys = torch.stack([seed, gidx], dim=-1).to(torch.uint64)
    u = prng.uniform(keys, (gidx.numel(),)).reshape(M, N)
    rand16 = (u * (1 << 16)).to(torch.int32)
    return (_sr_bf16_dither(x, rand16),)


# same unbiasedness check as SrF32ToBf16 -- the two variants differ only in RNG keying (global
# position vs tile-local order), not in the SR property each output must satisfy.
SrF32ToBf16Global = QuantCastSingleKernelGold(
    pt_ref_fn=sr_bf16_global_f,
    correctness_fn=_sr_bf16_unbiased_correctness,
    example_input_fn=_sr_inputs,
    perf_description="elementwise SR with stateless RNG",
)


# (name, gold) index of every golden recipe. Consumed by quant_cast_gold/test.py (correctness)
# and quant_cast_bench/benchmark.py (bandwidth sweep).
ALL_RECIPES = [
    # elementwise
    ("fp8_tensorwise_precalc_scale", Float8TensorwiseGold),
    ("fp8_rowwise_precalc_scale", RowwisePrecalcGold),
    ("fp8_colwise_precalc_scale", ColwisePrecalcGold),
    # 8-bit 1D, dim-k reduction
    ("mxfp8_floor", Mxfp8FloorGold),
    ("mxfp8_floor_swizzle", Mxfp8FloorSwizzleGold),
    ("fp8_deepseek_1x128", Deepseek1x128Gold),
    # 8-bit 1D, dim-m reduction
    ("mxfp8_floor_dim_m", Mxfp8FloorDimMGold),
    ("mxfp8_floor_dim_m_swizzle", Mxfp8FloorDimMSwizzleGold),
    ("fp8_deepseek_1x128_dim_m", Deepseek1x128DimMGold),
    # 8-bit 1D, dim-km reduction 
    ("mxfp8_floor_dim_km", Mxfp8FloorDimKmGold),
    ("fp8_deepseek_1x128_dim_km", Deepseek1x128DimKmGold),
    # 8-bit 2D
    ("mxfp8_32x32_floor", Mxfp832x32FloorGold),
    ("fp8_deepseek_128x128", Deepseek128x128Gold),
    # 8-bit rowwise/colwise
    ("fp8_rowwise", RowwiseFp8Gold),
    ("fp8_colwise", ColwiseFp8Gold),
    # 4 bit 1D
    ("nvfp4_swizzle", Nvfp4GsSwizzleGold),
    ("nvfp4_blocked_outer", Nvfp4BlockedOuterGold),
    # RHT
    ("bf16_rht", HadamardRht),
    # stochastic rounding
    ("fp32_to_bf16_sr", SrF32ToBf16),
    ("fp32_to_bf16_sr_global_offsets", SrF32ToBf16Global),
    # debug (not real recipes)
    ("mxfp8_bias", Mxfp8BiasGold),
]
