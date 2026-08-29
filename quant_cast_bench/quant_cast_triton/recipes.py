"""Triton implementations of the quant_cast_gold recipes.

Each recipe is a `QuantCastTritonRecipe` -- it inherits the gold reference
(`pt_ref_fn`/`correctness_fn`/`example_input_fn`/`perf_description`) from a
`QuantCastSingleKernelGold` and adds `triton_fn`, a Triton-backed implementation of the same
cast. Mirrors flex_tile_map's `RecipeV2` (inherit-from-gold + `from_gold`). test.py grades each
`triton_fn` against its gold `pt_ref_fn`.
"""

import functools
from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl

from quant_cast_bench.quant_cast_gold.recipes import (
    ColwiseFp8Gold,
    ColwisePrecalcGold,
    Deepseek1x128DimKmGold,
    Deepseek1x128DimMGold,
    Deepseek1x128Gold,
    Deepseek128x128Gold,
    Float8TensorwiseGold,
    HadamardRht,
    Mxfp832x32DimKMSwizzleGold,
    Mxfp832x32DimMSwizzleGold,
    Mxfp832x32ExpandGold,
    Mxfp832x32QdataDimKScaleDimKMSwizzleGold,
    Mxfp832x32SwizzleGold,
    Mxfp8DimKmGold,
    Mxfp8DimKmSwizzleGold,
    Mxfp8DimMGold,
    Mxfp8DimMSwizzleGold,
    Mxfp8Gold,
    Mxfp8SwizzleGold,
    Nvfp4BlockedOuterGold,
    Nvfp4GsGold,
    Nvfp4GsNVIDIASRSwizzleGold,
    Nvfp4GsSRSwizzleGold,
    Nvfp4GsSwizzleGold,
    QuantCastSingleKernelGold,
    RowwiseFp8Gold,
    RowwisePrecalcGold,
    SrF32ToBf16Global,
)


@dataclass(frozen=True)
class QuantCastTritonRecipe(QuantCastSingleKernelGold):
    """A gold recipe plus a Triton implementation of its `pt_ref_fn`. Mirrors flex_tile_map's
    RecipeV2: inherits pt_ref_fn/correctness_fn/example_input_fn/perf_description from the gold,
    and adds `triton_fn` (same `(inputs) -> outputs` signature as `pt_ref_fn`)."""

    triton_fn: Callable | None = None

    @classmethod
    def from_gold(cls, gold: QuantCastSingleKernelGold, triton_fn: Callable) -> "QuantCastTritonRecipe":
        """Build a QuantCastTritonRecipe from a gold recipe, attaching its Triton implementation."""
        return cls(
            pt_ref_fn=gold.pt_ref_fn,
            correctness_fn=gold.correctness_fn,
            example_input_fn=gold.example_input_fn,
            perf_description=gold.perf_description,
            triton_fn=triton_fn,
        )


# ---------------------------------------------------------------------------
# fp8 tensorwise with a precomputed per-tensor scale. The scale is an input (a global reduction
# done outside), so the kernel is a pure elementwise cast: qdata = (x * (1/scale)).to(fp8_e4m3).
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_tensorwise_kernel(x_ptr, scale_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr)  # precomputed per-tensor scalar
    y = (x * (1.0 / scale)).to(tl.float8e4nv)  # mirror float8_tensorwise_f exactly
    tl.store(y_ptr + offs, y, mask=mask)


def float8_tensorwise_triton(x, scale, **kwargs):
    """Triton impl matching float8_tensorwise_f: elementwise (x / scale) -> fp8_e4m3. `scale` is
    the precomputed per-tensor scalar. Returns a 1-tuple `(qdata,)`."""
    assert x.is_contiguous() and x.dim() == 2
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    n = x.numel()

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK"]),)

    _fp8_tensorwise_kernel[grid](x, scale, y, n, BLOCK=1024)
    return (y,)


FP8_TENSORWISE_PRECALC_SCALE = QuantCastTritonRecipe.from_gold(
    Float8TensorwiseGold, triton_fn=float8_tensorwise_triton
)


# ---------------------------------------------------------------------------
# fp8 rowwise with a precomputed (M, 1) per-row scale (an aux input). Elementwise divide + cast;
# each tile divides its rows by the matching per-row scalar. Mirrors rowwise_precalc_f.
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_rowwise_precalc_kernel(
    x_ptr, s_ptr, y_ptr, M, N, sxm, sxn, sym, syn, BM: tl.constexpr, BN: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    mask = m_mask[:, None] & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + offs_m, mask=m_mask)  # (BM,) per-row scale, scale is (M, 1) contiguous
    y = (x / s[:, None]).to(tl.float8e4nv)
    tl.store(y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn, y, mask=mask)


def fp8_rowwise_precalc_triton(x, scale, **kwargs):
    """Matches rowwise_precalc_f: (x / per-row-scale) -> fp8_e4m3. `scale` is (M, 1). Returns (qdata,)."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)

    def grid(meta):
        return (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))

    _fp8_rowwise_precalc_kernel[grid](
        x, scale, y, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1), BM=64, BN=64
    )
    return (y,)


FP8_ROWWISE_PRECALC_SCALE = QuantCastTritonRecipe.from_gold(
    RowwisePrecalcGold, triton_fn=fp8_rowwise_precalc_triton
)


# ---------------------------------------------------------------------------
# fp8 colwise with a precomputed (1, N) per-column scale (aux). Elementwise divide + cast, then a
# TRANSPOSED-contiguous store: output is (N, M). Mirrors colwise_precalc_f.
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_colwise_precalc_kernel(
    x_ptr, s_ptr, y_ptr, M, N, sxm, sxn, sym, syn, BM: tl.constexpr, BN: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=m_mask[:, None] & n_mask[None, :]
    ).to(tl.float32)
    s = tl.load(s_ptr + offs_n, mask=n_mask)  # (BN,) per-col scale, scale is (1, N) contiguous
    y = (x / s[None, :]).to(tl.float8e4nv)  # (BM, BN)
    # transposed store into (N, M): out[n, m] = y[m, n]
    out_off = offs_n[:, None] * sym + offs_m[None, :] * syn
    tl.store(y_ptr + out_off, tl.trans(y), mask=n_mask[:, None] & m_mask[None, :])


def fp8_colwise_precalc_triton(x, scale, **kwargs):
    """Matches colwise_precalc_f: (x / per-col-scale) -> fp8_e4m3, transposed-contiguous (N, M).
    `scale` is (1, N). Returns (qdata,)."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)

    def grid(meta):
        return (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))

    _fp8_colwise_precalc_kernel[grid](
        x, scale, y, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1), BM=64, BN=64
    )
    return (y,)


FP8_COLWISE_PRECALC_SCALE = QuantCastTritonRecipe.from_gold(
    ColwisePrecalcGold, triton_fn=fp8_colwise_precalc_triton
)


# ---------------------------------------------------------------------------
# deepseek fp8 1x128: one fp32 scale per (row, 128-col-block). amax over the 128 group; multiply
# by 1/scale and cast. Mirrors deepseek_1x128_f. Grid: (cdiv(M, BM), N // 128).
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_deepseek_1x128_kernel(
    x_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, ssm, ssn, BM: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_b * 128 + tl.arange(0, 128)
    m_mask = offs_m < M
    mask = m_mask[:, None] & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=mask).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12)  # (BM,)
    scale = amax / 448.0
    y = (x * (1.0 / scale)[:, None]).to(tl.float8e4nv)
    tl.store(y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn, y, mask=mask)
    tl.store(s_ptr + offs_m * ssm + pid_b * ssn, scale, mask=m_mask)


def fp8_deepseek_1x128_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, N // 128, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, 64), N // 128)
    _fp8_deepseek_1x128_kernel[grid](
        x, y, s, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        s.stride(0), s.stride(1), BM=64,
    )
    return y, s


FP8_DEEPSEEK_1X128 = QuantCastTritonRecipe.from_gold(
    Deepseek1x128Gold, triton_fn=fp8_deepseek_1x128_triton
)


# ---------------------------------------------------------------------------
# deepseek fp8 128x128: one fp32 scale per 128x128 block (amax over the whole block).
# Mirrors deepseek_128x128_f. Grid: (M // 128, N // 128).
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_deepseek_128x128_kernel(x_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, ssm, ssn):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=mask).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-12)  # scalar over the whole 128x128 tile
    scale = amax / 448.0
    y = (x * (1.0 / scale)).to(tl.float8e4nv)
    tl.store(y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn, y, mask=mask)
    tl.store(s_ptr + pid_m * ssm + pid_n * ssn, scale)


def fp8_deepseek_128x128_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty(M // 128, N // 128, dtype=torch.float32, device=x.device)
    grid = (M // 128, N // 128)
    _fp8_deepseek_128x128_kernel[grid](
        x, y, s, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1), s.stride(0), s.stride(1)
    )
    return y, s


FP8_DEEPSEEK_128X128 = QuantCastTritonRecipe.from_gold(
    Deepseek128x128Gold, triton_fn=fp8_deepseek_128x128_triton
)


# ---------------------------------------------------------------------------
# deepseek fp8 1x128 dim-M: reduce 128-row blocks down M, one fp32 scale per (128-row-block, col);
# transposed-contiguous outputs (N, M) / (N, M//128). Mirrors deepseek_1x128_dim_m_f.
# Grid: (M // 128, cdiv(N, BN)).
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_deepseek_1x128_dim_m_kernel(
    x_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, ssm, ssn, BN: tl.constexpr
):
    pid_rb = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_rb * 128 + tl.arange(0, 128)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    # Keep the loaded tile in bf16; the amax reduction is exact in bf16 (abs just clears the sign
    # bit, max is a comparison), so we only promote to fp32 for the short-lived reduction accumulator
    # and for the final quantize. This halves the tile's register footprint versus promoting the
    # whole (128, BN) tile up front, which lifts occupancy on these tiny 1-warp CTAs.
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=n_mask[None, :])
    amax = tl.maximum(tl.max(tl.abs(x).to(tl.float32), axis=0), 1e-12)  # (BN,) per column
    scale = amax / 448.0
    y = (x.to(tl.float32) * (1.0 / scale)[None, :]).to(tl.float8e4nv)  # (128, BN)
    # transposed store into (N, M): out[n, m] = y[row_in_block, n]
    out_off = offs_n[:, None] * sym + offs_m[None, :] * syn
    tl.store(y_ptr + out_off, tl.trans(y), mask=n_mask[:, None])
    # scale (N, M//128): out_scale[n, pid_rb] = scale[n]
    tl.store(s_ptr + offs_n * ssm + pid_rb * ssn, scale, mask=n_mask)


def fp8_deepseek_1x128_dim_m_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((N, M // 128), dtype=torch.float32, device=x.device)
    # BN=32, single warp: the transposed fp8 store is bottlenecked by the tl.trans smem roundtrip, so
    # small high-occupancy CTAs (many resident, 128B contiguous store rows) beat larger tiles here.
    grid = (M // 128, triton.cdiv(N, 32))
    _fp8_deepseek_1x128_dim_m_kernel[grid](
        x, y, s, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        s.stride(0), s.stride(1), BN=32, num_warps=1,
    )
    return y, s


FP8_DEEPSEEK_1X128_DIM_M = QuantCastTritonRecipe.from_gold(
    Deepseek1x128DimMGold, triton_fn=fp8_deepseek_1x128_dim_m_triton
)


# ---------------------------------------------------------------------------
# deepseek fp8 1x128 in BOTH directions, ONE pass. Each program owns a 128x128 tile of x (read
# once): dim-K reduces the 128 columns (one 1x128 block per row) and dim-M reduces the 128 rows
# (one 128x1 block per column), so a single tile aligns both block reductions. Emits 4 outputs:
# qdata_k (M,N)/scale_k (M,N//128) like fp8_deepseek_1x128, and qdata_m (N,M)/scale_m (N,M//128)
# like fp8_deepseek_1x128_dim_m (transposed store). Requires M%128==0 and N%128==0.
# Grid: (M // 128, N // 128).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (2, 4, 8)], key=["M", "N"]
)
@triton.jit
def _fp8_deepseek_1x128_dim_km_kernel(
    x_ptr, yk_ptr, sk_ptr, ym_ptr, sm_ptr, M, N,
    sxm, sxn, sykm, sykn, sskm, sskn, symn, symm, ssmn, ssmm,
):
    pid_m = tl.program_id(0)  # 128-row block
    pid_n = tl.program_id(1)  # 128-col block
    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn).to(tl.float32)  # (128,128)
    # dim-K: one 1x128 block per row -> reduce over the 128 columns (axis=1).
    amax_k = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12)  # (128,) per row
    scale_k = amax_k / 448.0
    yk = (x * (1.0 / scale_k)[:, None]).to(tl.float8e4nv)
    tl.store(yk_ptr + offs_m[:, None] * sykm + offs_n[None, :] * sykn, yk)
    tl.store(sk_ptr + offs_m * sskm + pid_n * sskn, scale_k)
    # dim-M: one 128x1 block per column -> reduce over the 128 rows (axis=0); transposed store.
    amax_m = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)  # (128,) per column
    scale_m = amax_m / 448.0
    ym = (x * (1.0 / scale_m)[None, :]).to(tl.float8e4nv)  # (128,128) in (row, col)
    # out[n, m] = ym[row, col] with n=offs_n[col], m=offs_m[row] -> store tl.trans(ym) into (N, M).
    tl.store(ym_ptr + offs_n[:, None] * symn + offs_m[None, :] * symm, tl.trans(ym))
    tl.store(sm_ptr + offs_n * ssmn + pid_m * ssmm, scale_m)


def fp8_deepseek_1x128_dim_km_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0, "dim_km kernel needs M%128==0 and N%128==0"
    yk = torch.empty_like(x, dtype=torch.float8_e4m3fn)          # (M, N)
    sk = torch.empty(M, N // 128, dtype=torch.float32, device=x.device)
    ym = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)  # (N, M) transposed
    sm = torch.empty(N, M // 128, dtype=torch.float32, device=x.device)
    grid = (M // 128, N // 128)
    _fp8_deepseek_1x128_dim_km_kernel[grid](
        x, yk, sk, ym, sm, M, N,
        x.stride(0), x.stride(1), yk.stride(0), yk.stride(1), sk.stride(0), sk.stride(1),
        ym.stride(0), ym.stride(1), sm.stride(0), sm.stride(1),
    )
    return yk, sk, ym, sm


FP8_DEEPSEEK_1X128_DIM_KM = QuantCastTritonRecipe.from_gold(
    Deepseek1x128DimKmGold, triton_fn=fp8_deepseek_1x128_dim_km_triton
)


# ---------------------------------------------------------------------------
# fp8 rowwise (full-span): one fp32 scale per row, amax over ALL columns. Two passes over N
# (accumulate amax, then quant) so any N works. Mirrors rowwise_fp8_f. Grid: (cdiv(M, BM),).
# Perf (matched to Inductor's codegen for this reduction): autotune (BM, BN) and use eviction
# hints so the amax pass keeps rows resident (evict_last) for the quant pass to re-read (evict_first).
# ---------------------------------------------------------------------------
_ROWWISE_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn}, num_warps=w)
    for bm in (1, 2, 4, 8)
    for bn in (1024, 2048, 4096)
    for w in (4, 8)
]


@triton.autotune(configs=_ROWWISE_CONFIGS, key=["M", "N"])
@triton.jit
def _fp8_rowwise_kernel(x_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    amax = tl.zeros((BM,), dtype=tl.float32)
    for j in range(0, tl.cdiv(N, BN)):
        offs_n = j * BN + tl.arange(0, BN)
        n_mask = offs_n < N
        x = tl.load(
            x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn,
            mask=m_mask[:, None] & n_mask[None, :], other=0.0, eviction_policy="evict_last",
        ).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
    amax = tl.maximum(amax, 1e-12)
    scale = amax / 448.0  # mirror gold: scale then 1/scale (two roundings), not 448/amax
    inv = 1.0 / scale
    for j in range(0, tl.cdiv(N, BN)):
        offs_n = j * BN + tl.arange(0, BN)
        n_mask = offs_n < N
        mask = m_mask[:, None] & n_mask[None, :]
        x = tl.load(
            x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=mask,
            eviction_policy="evict_first",
        ).to(tl.float32)
        y = (x * inv[:, None]).to(tl.float8e4nv)
        tl.store(y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn, y, mask=mask)
    tl.store(s_ptr + offs_m, scale, mask=m_mask)  # scale (M, 1) contiguous


def fp8_rowwise_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, 1, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)  # noqa: E731
    _fp8_rowwise_kernel[grid](
        x, y, s, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1)
    )
    return y, s


FP8_ROWWISE = QuantCastTritonRecipe.from_gold(RowwiseFp8Gold, triton_fn=fp8_rowwise_triton)


# ---------------------------------------------------------------------------
# fp8 colwise (full-span): one fp32 scale per column, amax over ALL rows; transposed-contiguous
# output (N, M) and scale (N, 1). Mirrors colwise_fp8_f.
#
# Perf: the scale is a full-column (dim-M) reduction, so the cast is inherently reduce-then-quantize
# and reads x twice from DRAM (the reload misses L2 -- unlike rowwise, many concurrent full-column
# strips thrash the cache). A single kernel is forced into narrow, *strided* reads (the reduction
# axis M is the strided one in row-major x), which caps DRAM utilization ~50% -> ~37% of peak.
# Splitting into two kernels lets BOTH reads be *coalesced* (row-major), lifting DRAM utilization:
#   (A) `_fp8_colwise_amax_kernel`: coalesced wide (BM, BN) tiles, partial per-column amax, combined
#       across the M-grid with `tl.atomic_max` into a per-column scratch buffer.
#   (B) `_fp8_colwise_quant_kernel`: reads x once (coalesced), quantizes with the precomputed amax,
#       and writes the transposed (N, M) output + the (N, 1) scale.
# ~37% -> ~46% of peak. (True read-once needs staging the column in SMEM, which Triton can't express
# -- that's a CuTeDSL/CUDA optimization.)
# ---------------------------------------------------------------------------
_COLWISE_AMAX_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn}, num_warps=w)
    for bm in (128, 256) for bn in (128, 256) for w in (4, 8)
]
_COLWISE_QUANT_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn}, num_warps=w)
    for bm in (256, 512) for bn in (32, 64) for w in (4, 8)
]


@triton.autotune(configs=_COLWISE_AMAX_CONFIGS, key=["M", "N"])
@triton.jit
def _fp8_colwise_amax_kernel(x_ptr, a_ptr, M, N, sxm, sxn, BM: tl.constexpr, BN: tl.constexpr):
    # coalesced (BM, BN) row-major tile -> partial per-column amax -> atomic_max into a_ptr[N].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn,
        mask=m_mask[:, None] & n_mask[None, :], other=0.0,
    ).to(tl.float32)
    tl.atomic_max(a_ptr + offs_n, tl.max(tl.abs(x), axis=0), mask=n_mask)


@triton.autotune(configs=_COLWISE_QUANT_CONFIGS, key=["M", "N"])
@triton.jit
def _fp8_colwise_quant_kernel(
    x_ptr, a_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, BM: tl.constexpr, BN: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    amax = tl.maximum(tl.load(a_ptr + offs_n, mask=n_mask, other=1e-12), 1e-12)
    scale = amax / 448.0  # (BN,); mirror gold: scale then 1/scale
    inv = 1.0 / scale
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=m_mask[:, None] & n_mask[None, :]
    ).to(tl.float32)
    y = (x * inv[None, :]).to(tl.float8e4nv)  # (BM, BN)
    out_off = offs_n[:, None] * sym + offs_m[None, :] * syn  # transposed (N, M)
    tl.store(y_ptr + out_off, tl.trans(y), mask=n_mask[:, None] & m_mask[None, :])
    if pid_m == 0:
        tl.store(s_ptr + offs_n, scale, mask=n_mask)  # scale (N, 1), written once per column


def fp8_colwise_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(N, 1, dtype=torch.float32, device=x.device)
    a = torch.zeros(N, dtype=torch.float32, device=x.device)  # per-column amax scratch (>=0)
    grid_a = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
    grid_q = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
    _fp8_colwise_amax_kernel[grid_a](x, a, M, N, x.stride(0), x.stride(1))
    _fp8_colwise_quant_kernel[grid_q](
        x, a, y, s, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1)
    )
    return y, s


FP8_COLWISE = QuantCastTritonRecipe.from_gold(ColwiseFp8Gold, triton_fn=fp8_colwise_triton)


# ---------------------------------------------------------------------------
# e8m0 device helpers (mxfp8). Exact ports of _amax_to_e8m0 / _e8m0_scale_to_reciprocal_fp32 (recipes.py)
# so the scale matches the reference bit-for-bit. e8m0 is stored as its uint8 biased-exponent
# byte (the wrapper .view()s it as float8_e8m0fnu).
# ---------------------------------------------------------------------------
@triton.jit
def _amax_to_e8m0_tl(amax):
    # amax: fp32. RCEIL: descale = amax / 448, then round UP to the next power-of-two e8m0
    # exponent (whenever descale's fp32 mantissa is nonzero). Returns the biased exponent as
    # int32 (caller stores it as uint8). Mirrors the gold _amax_to_e8m0_rceil bit-math.
    i = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)  # descale bits
    biased_exponent = (i >> 23) & 0xFF
    mantissa = i & 0x7FFFFF
    # normal fp32 rounds up on any set mantissa bit; fp32 subnormals (biased_exp == 0) only above 2^-127.
    needs_round_up = tl.where(biased_exponent == 0, mantissa > 0x400000, mantissa != 0)
    return biased_exponent + needs_round_up.to(tl.int32)


@triton.jit
def _amax_to_e8m0_cvt(amax):
    # Blackwell (SM100+) hardware e8m0 RCEIL: `cvt.rp.ue8m0x2.f32` rounds toward +inf, i.e. rounds
    # the descale = amax / 448 up to the next power-of-two e8m0 exponent -- matching
    # _amax_to_e8m0_tl / the gold cvt path without the register-heavy bit math.
    # The x2 op packs two e8m0 into a .b16; we feed 0.0 as the high lane and keep the low byte.
    a = (amax * (1.0 / 448.0)).to(tl.float32)  # descale
    packed = tl.inline_asm_elementwise(
        asm="cvt.rp.ue8m0x2.f32 $0, 0f00000000, $1;",
        constraints="=h,f",
        args=[a],
        dtype=tl.int16,
        is_pure=True,
        pack=1,
    )
    return packed.to(tl.int32) & 0xFF


@triton.jit
def _e8m0_to_reciprocal_fp32_tl(biased):
    # biased: int32 e8m0 exponent -> fp32 reciprocal pow2 factor 2^(127-e), matching the gold
    # _e8m0_scale_to_reciprocal_fp32 (reciprocal biased exponent = 254 - e). The cast multiplies
    # data by this (torchao _to_mx_rceil) instead of dividing by the reconstructed scale.
    return ((254 - biased) << 23).to(tl.float32, bitcast=True)


# ---------------------------------------------------------------------------
# mxfp8 1x32: one e8m0 scale per (row, 32-col-block). Mirrors mxfp8_f (plain, SWIZZLE=False) and
# mxfp8_swizzle_f (SWIZZLE=True). SWIZZLE=True writes the scale straight into the NVIDIA-swizzled
# (nrb, ncb, 32, 16) block layout: pre-swizzle (row, col) lands at ((br*ncb+bc)*32+b)*16 + (a*4+c4),
# where br=row//128, r128=row%128, a=r128//32, b=r128%32, bc=col//4, c4=col%4 (per _to_blocked_4d).
#
# Perf: one 1-D persistent-reduction grid (mirrors Inductor's codegen). Slot g maps to a pre-swizzle
# (row, col-block) and reads a 32-contiguous chunk of the row-major input (flat [row*N+col*32, +32)),
# which coalesces. BOTH paths use the SAME grid: the qdata padded up to whole 128x128 tiles -- each
# 128x128 tile is exactly one swizzle atom (128 rows x 4 col-blocks x 32 = 128 cols -> 512 scale
# slots), so the padded grid never splits an atom. Padded slots (real == False) are masked out of the
# qdata store; SWIZZLE=True writes their scale as literal 0 (its (nrb, ncb, 32, 16) buffer is exactly
# n_slots and stays torch.empty-allocatable), SWIZZLE=False skips them (its scale is a plain, exactly
# sized (M, N//32) buffer). Aligned shapes (M,N % 128 == 0) have pm,pn == M,N, i.e. the exact grid.
# Grid: (cdiv(n_slots, GBLOCK),).
# ---------------------------------------------------------------------------
_MXFP8_1X32_CONFIGS = [
    triton.Config({"GBLOCK": g}, num_warps=w)
    for g in (32, 64, 128, 256, 512, 1024)
    for w in (2, 4, 8)
]


@triton.autotune(configs=_MXFP8_1X32_CONFIGS, key=["n_slots"])
@triton.jit
def _mxfp8_kernel(x_ptr, y_ptr, s_ptr, n_slots, M, N, NGC, NCB,
                  SWIZZLE: tl.constexpr, GBLOCK: tl.constexpr):
    # Per-row col-block stride of the 1-D flattening: padded to a full swizzle atom (NCB*4 == pn//32)
    # for BOTH paths, so the grid is identical for swizzle and non-swizzle.
    PCOLS = NCB * 4
    pid = tl.program_id(0)
    g = pid * GBLOCK + tl.arange(0, GBLOCK)
    g_mask = g < n_slots
    row = g // PCOLS
    col = g % PCOLS
    real = (row < M) & (col < NGC)  # masks padded slots (and, since row >= pm >= M there, the g >= n_slots tail)
    off = row[:, None] * N + col[:, None] * 32 + tl.arange(0, 32)[None, :]  # (GBLOCK, 32)
    x = tl.load(x_ptr + off, mask=real[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)  # (GBLOCK,) -- mxfp8 does NOT clamp amax
    biased = _amax_to_e8m0_tl(amax)
    rcp = _e8m0_to_reciprocal_fp32_tl(biased)
    y = (x * rcp[:, None]).to(tl.float8e4nv)
    tl.store(y_ptr + off, y, mask=real[:, None])  # y is exactly (M, N): only real slots written
    if SWIZZLE:
        # scatter the e8m0 byte into the swizzled (nrb, ncb, 32, 16) layout; padded slots
        # (real == False) write literal 0 to match gold's zero-pad.
        br = row // 128
        r128 = row % 128
        a = r128 // 32
        b = r128 % 32
        bc = col // 4
        c4 = col % 4
        flat = ((br * NCB + bc) * 32 + b) * 16 + (a * 4 + c4)
        tl.store(s_ptr + flat, tl.where(real, biased.to(tl.uint8), 0), mask=g_mask)
    else:
        # plain contiguous (M, NGC) scale at (row, col-block); mask=real skips padded slots (and tail).
        tl.store(s_ptr + row * NGC + col, biased.to(tl.uint8), mask=real)


def mxfp8_triton(x, swizzle, **kwargs):
    """mxfp8 1x32 cast via _mxfp8_kernel. swizzle=False: the e8m0 scale is a plain contiguous
    (M, N//32) buffer. swizzle=True: the scale is written into the NVIDIA-swizzled (nrb, ncb, 32, 16)
    block layout (mirrors mxfp8_swizzle_f); the grid then walks the qdata padded to whole 128x128
    tiles so every swizzled slot is written (torch.empty stays valid). Returns (qdata, scale)."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    ngc = N // 32  # real 32-col-groups per row
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # ONE grid for both paths: walk the qdata padded to whole 128x128 tiles (a 128x128 tile == one
    # swizzle atom). Aligned shapes (M,N % 128 == 0) => pm,pn == M,N, i.e. the exact grid.
    pm = ((M + 127) // 128) * 128  # padded qdata rows
    pn = ((N + 127) // 128) * 128  # padded qdata cols
    ncb = pn // 128  # padded col-blocks; PCOLS = ncb*4 == pn//32 in the kernel
    n_slots = pm * (pn // 32)  # one slot per padded (row, 32-col-group); == nrb*ncb*512 when swizzled
    if swizzle:
        # torch.empty (not zeros): the grid walks every padded slot, writing a real e8m0 byte or 0.
        s_u8 = torch.empty(pm // 128, ncb, 32, 16, dtype=torch.uint8, device=x.device)
    else:
        s_u8 = torch.empty(M, ngc, dtype=torch.uint8, device=x.device)  # plain 2-D scale, exactly sized
    grid = lambda meta: (triton.cdiv(n_slots, meta["GBLOCK"]),)  # noqa: E731
    _mxfp8_kernel[grid](x, y, s_u8, n_slots, M, N, ngc, ncb, SWIZZLE=swizzle)
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8 = QuantCastTritonRecipe.from_gold(
    Mxfp8Gold, triton_fn=functools.partial(mxfp8_triton, swizzle=False)
)
MXFP8_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Mxfp8SwizzleGold, triton_fn=functools.partial(mxfp8_triton, swizzle=True)
)


# ---------------------------------------------------------------------------
# mxfp8 32x32: one e8m0 scale per 32x32 block. Mirrors mxfp8_32x32_expand_f / mxfp8_32x32_swizzle_f.
# Perf: one 32x32 block per program is tiny/low-intensity. Batch CB col-blocks per program
# (32 rows x CB*32 cols), reshaping to (32, CB, 32) and reducing the row + within-block dims;
# autotune CB and num_warps.
# SWIZZLE=False expands each block scale over its 32 rows -> a plain, exactly-sized (M, N//32) 2D
# buffer (the 1x32 layout a gemm consumes, strides ss*). SWIZZLE=True expands the same way -> a (M, N//32)
# grid scattered into the NVIDIA-swizzled 4D block grid (nrb, ncb, 32, 16) via _mxfp8_kernel's flat
# formula; that swizzled buffer is a bijection over the PADDED grid (row in [0, nrb*128), col in
# [0, ncb*4)), real scale at row<M and col<N//32, the rest padding gold's _to_blocked_4d zero-fills,
# so the kernel writes EVERY slot itself (real byte or literal 0) and the buffer stays torch.empty.
# Grid (BOTH paths): the input padded to whole 128x128 blocks -- (mpad//32, cdiv(npad, CB*32)). When
# M%128==0 and N%128==0 this equals the old exact/real-only grid of each path. RAGGED=(M%128!=0 or
# N%128!=0): padding only occurs when RAGGED, so the plain path only needs its extra m_mask/row guard
# then (aligned -> no padded row-blocks, old n_mask-only behavior); the swizzle path always masks.
# ---------------------------------------------------------------------------
_MXFP8_32X32_CONFIGS = [
    triton.Config({"CB": cb}, num_warps=w) for cb in (2, 4, 8, 16) for w in (2, 4, 8)
]


@triton.autotune(configs=_MXFP8_32X32_CONFIGS, key=["M", "N"])
@triton.jit
def _mxfp8_32x32_kernel(
    x_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, ssm, ssn, NCB,
    SWIZZLE: tl.constexpr, RAGGED: tl.constexpr, CB: tl.constexpr,
):
    pid_rb = tl.program_id(0)  # 32-row block (over the padded rows mpad)
    pid_cb = tl.program_id(1)  # group of CB 32-col blocks (over the padded cols npad)
    offs_m = pid_rb * 32 + tl.arange(0, 32)
    offs_n = pid_cb * (CB * 32) + tl.arange(0, CB * 32)
    m_mask = offs_m < M
    n_mask = offs_n < N
    # SWIZZLE always predicates rows (its grid spans the padded rows). The plain path predicates rows
    # only when RAGGED: aligned M%128==0 -> the grid has no padded 32-row block, so m_mask is all-true
    # and the old n_mask-only path is preserved (n_mask is always needed: CB*32 rarely divides N).
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn,
        mask=(m_mask[:, None] & n_mask[None, :]) if (SWIZZLE or RAGGED) else n_mask[None, :],
        other=0.0,
    ).to(tl.float32)  # (32, CB*32); padded rows/cols read as 0
    xr = tl.reshape(x, (32, CB, 32))
    amax = tl.max(tl.max(tl.abs(xr), axis=2), axis=0)  # (CB,): within-block cols, then 32 rows
    biased = _amax_to_e8m0_tl(amax)  # (CB,)
    rcp = _e8m0_to_reciprocal_fp32_tl(biased)
    y = tl.reshape((xr * rcp[None, :, None]).to(tl.float8e4nv), (32, CB * 32))
    tl.store(y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn, y,
             mask=(m_mask[:, None] & n_mask[None, :]) if (SWIZZLE or RAGGED) else n_mask[None, :])
    if SWIZZLE:
        # swizzled scale store: expand the single block scale over all 32 rows of the block, so each
        # (row = offs_m, col = 32-col-block) gets the same biased[cb] byte at its swizzled slot. The
        # grid tiles the padded grid, so padded slots (row >= M or col >= N//32) get literal 0; mask is
        # OOB-safety only (col >= ncb*4 -> bc >= NCB -> out-of-buffer flat).
        row = offs_m[:, None]                                # (32, 1)
        col = (pid_cb * CB + tl.arange(0, CB))[None, :]      # (1, CB)
        r128 = row % 128
        flat = (((row // 128) * NCB + col // 4) * 32 + r128 % 32) * 16 + (r128 // 32 * 4 + col % 4)
        s_bytes = tl.where((row < M) & (col < (N // 32)),
                           tl.broadcast_to(biased.to(tl.uint8)[None, :], (32, CB)), 0)
        tl.store(s_ptr + flat, s_bytes, mask=(col < (NCB * 4)))
    else:
        # plain 2D scale store, EXPANDED to the 1x32 layout: repeat the single block scale over all 32
        # rows of the block -> a (M, N//32) grid a gemm consumes directly (no swizzle). Buffer is
        # exactly (M, N//32) -> skip padded block-cols (>= N//32) and, when RAGGED, padded rows.
        s_rows = offs_m[:, None]                             # (32, 1)
        s_cols = (pid_cb * CB + tl.arange(0, CB))[None, :]   # (1, CB)
        col_ok = s_cols < (N // 32)
        tl.store(s_ptr + s_rows * ssm + s_cols * ssn,
                 tl.broadcast_to(biased.to(tl.uint8)[None, :], (32, CB)),
                 mask=(col_ok & m_mask[:, None]) if RAGGED else col_ok)


def mxfp8_32x32_triton(x, swizzle, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0 and N % 32 == 0, "mxfp8_32x32 kernel needs M%32==0 and N%32==0"
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # ONE grid for swizzle and non-swizzle: tile the input padded to whole 128x128 blocks. 32 divides
    # 128 and CB*32 with cdiv covers npad, so aligned M,N%128==0 reduce to each path's old grid.
    mpad = ((M + 127) // 128) * 128
    npad = ((N + 127) // 128) * 128
    grid = lambda meta: (mpad // 32, triton.cdiv(npad, meta["CB"] * 32))  # noqa: E731
    ragged = M % 128 != 0 or N % 128 != 0
    if swizzle:
        ncb = ((N // 32) + 3) // 4  # scale col-blocks (over N//32)
        # torch.empty (not zeros): the kernel writes every slot of the padded (nrb, ncb, 32, 16) grid
        # itself -- real e8m0 bytes plus literal 0 for the padded rows/cols.
        s_u8 = torch.empty(mpad // 128, ncb, 32, 16, dtype=torch.uint8, device=x.device)
        _mxfp8_32x32_kernel[grid](
            x, y, s_u8, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1), 0, 0, ncb,
            SWIZZLE=True, RAGGED=ragged,  # ss* unused on swizzle path
        )
    else:
        s_u8 = torch.empty(M, N // 32, dtype=torch.uint8, device=x.device)  # expanded (1x32) scale
        _mxfp8_32x32_kernel[grid](
            x, y, s_u8, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
            s_u8.stride(0), s_u8.stride(1), 0,  # NCB unused when not swizzled
            SWIZZLE=False, RAGGED=ragged,
        )
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_32X32 = QuantCastTritonRecipe.from_gold(
    Mxfp832x32ExpandGold, triton_fn=functools.partial(mxfp8_32x32_triton, swizzle=False)
)
MXFP8_32X32_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Mxfp832x32SwizzleGold, triton_fn=functools.partial(mxfp8_32x32_triton, swizzle=True)
)


# ---------------------------------------------------------------------------
# mxfp8 32x32 dim-M / transposed swizzle: same square-block quant as _mxfp8_32x32_kernel
# (one e8m0 scale per 32x32 block, via the (32, CB, 32) double-max reduction), but the outputs are
# written in the dim-M / transposed frame -- qdata (N, M) and the scale into the NVIDIA-swizzled 4D
# block grid of the TRANSPOSED (N, M//32) scale. The block is square, so the scale VALUES are
# identical to _mxfp8_32x32_kernel; only the output layout changes. Mirrors
# mxfp8_32x32_dim_m_swizzle_f. The per-block scale is expanded over the 32 columns of its col-block
# (which become 32 transposed rows over N), and its swizzle pre-position is (row = n in [0, N),
# col = 32-row-block index in [0, M//32)) -- reusing _mxfp8_dim_m_kernel's SWIZZLE flat formula.
# Requires M%32==0 and N%32==0. The scale grid spans the PADDED extents (rows N->nrb*128, cols
# M//32->ncb*4) so every slot of the (nrb, ncb, 32, 16) buffer is written -- real e8m0 bytes or
# literal 0 (matching gold's zero-pad) -- letting the wrapper use torch.empty. Grid:
# (ncb*4, cdiv(nrb*128, CB*32)).
# ---------------------------------------------------------------------------
@triton.autotune(configs=_MXFP8_32X32_CONFIGS, key=["M", "N"])
@triton.jit
def _mxfp8_32x32_dim_m_swizzle_kernel(
    x_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, NCB, NRB, CB: tl.constexpr
):
    pid_rb = tl.program_id(0)  # single 32-row block (= transposed scale col, over the padded ncb*4)
    pid_cb = tl.program_id(1)  # group of CB 32-col blocks (= transposed scale rows, over padded nrb*128)
    offs_m = pid_rb * 32 + tl.arange(0, 32)
    offs_n = pid_cb * (CB * 32) + tl.arange(0, CB * 32)
    m_mask = offs_m < M
    n_mask = offs_n < N
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn,
        mask=m_mask[:, None] & n_mask[None, :], other=0.0,
    ).to(tl.float32)  # (32, CB*32); padded rows/cols read as 0
    xr = tl.reshape(x, (32, CB, 32))
    amax = tl.max(tl.max(tl.abs(xr), axis=2), axis=0)  # (CB,): within-block cols, then 32 rows
    biased = _amax_to_e8m0_tl(amax)  # (CB,)
    rcp = _e8m0_to_reciprocal_fp32_tl(biased)
    y = tl.reshape((xr * rcp[None, :, None]).to(tl.float8e4nv), (32, CB * 32))  # (32, CB*32)
    # transposed qdata store into (N, M): out[n, m] = y[m_in_tile, n_in_tile]. qdata is exactly
    # (N, M) with no padding, so gate on both real transposed-row (offs_n < N) and real col (offs_m < M).
    out_off = offs_n[:, None] * sym + offs_m[None, :] * syn
    tl.store(y_ptr + out_off, tl.trans(y), mask=n_mask[:, None] & m_mask[None, :])
    # swizzled scale store into the 4D grid (nrb, ncb, 32, 16). Pre-swizzle position in the transposed
    # frame: row = n (over N), col = pid_rb (this program's single 32-row-block, over M//32). The one
    # block scale is expanded over the 32 transposed rows of each col-block -- biased[cb] to every
    # offs_n in that col-block. Padded transposed rows (n >= N) / cols (pid_rb >= M//32) are written 0.
    row = offs_n                                    # (CB*32,)  transposed row over N
    br = row // 128
    r128 = row % 128
    a = r128 // 32
    b = r128 % 32
    bc = pid_rb // 4                                # col = pid_rb (32-row-block index)
    c4 = pid_rb % 4
    flat = ((br * NCB + bc) * 32 + b) * 16 + (a * 4 + c4)  # (CB*32,)
    # biased is (CB,); expand each col-block's scale over its 32 transposed rows -> (CB*32,).
    biased_exp = tl.reshape(tl.broadcast_to(biased[:, None], (CB, 32)), (CB * 32,))
    s_bytes = tl.where(n_mask & (pid_rb < (M // 32)), biased_exp.to(tl.uint8), 0)
    # mask is OOB-safety only (offs_n >= NRB*128 -> br >= NRB -> out-of-buffer flat); padded rows in
    # [N, NRB*128) are valid buffer positions and get the 0 written above.
    tl.store(s_ptr + flat, s_bytes, mask=offs_n < (NRB * 128))


def mxfp8_32x32_dim_m_swizzle_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0 and N % 32 == 0, "mxfp8_32x32_dim_m_swizzle kernel needs M%32==0 and N%32==0"
    y = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    nrb = (N + 127) // 128            # transposed rows = N
    ncb = ((M // 32) + 3) // 4        # transposed cols = M//32 (padded up to a multiple of 4)
    # torch.empty (not zeros): the kernel writes every slot of the padded (nrb, ncb, 32, 16) grid
    # itself -- real e8m0 bytes plus literal 0 for the padded transposed rows [N, nrb*128) and cols.
    s_u8 = torch.empty(nrb, ncb, 32, 16, dtype=torch.uint8, device=x.device)
    # grid dim0 = ncb*4 tiles the transposed scale col (32-row-block index) over [0, ncb*4) exactly;
    # dim1 covers the padded transposed rows (nrb*128), not just N -- so every swizzle slot is visited
    # exactly once. offs_m may overshoot M (masked on the qdata store).
    grid = lambda meta: (ncb * 4, triton.cdiv(nrb * 128, meta["CB"] * 32))  # noqa: E731
    _mxfp8_32x32_dim_m_swizzle_kernel[grid](
        x, y, s_u8, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1), ncb, nrb,
    )
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_32X32_DIM_M_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Mxfp832x32DimMSwizzleGold, triton_fn=mxfp8_32x32_dim_m_swizzle_triton
)


# ---------------------------------------------------------------------------
# mxfp8 dim-M: 32-row blocks down M, one e8m0 scale per (32-row-block, col); transposed qdata
# output (N, M). SWIZZLE=False writes the scale as a plain transposed 2D (N, M//32) buffer (strides
# ssm,ssn); SWIZZLE=True scatters each e8m0 byte into the NVIDIA-swizzled (nrb, ncb, 32, 16) block
# layout (NCB,NRB), acting on the TRANSPOSED-frame scale (pre-swizzle row = n over N, col = 32-row-
# block over M//32) -- reuses _mxfp8_kernel's SWIZZLE flat formula. Mirrors mxfp8_dim_m_f /
# mxfp8_dim_m_swizzle_f. Requires M % 32 == 0.
# Perf: process RB 32-row blocks x BN cols per program; reshape (RB*32, BN) -> (RB, 32, BN) and
# reduce the within-block 32. This kernel is memory-bound and OCCUPANCY-limited: the fp32 tile is
# register-heavy, so a large (RB*32, BN) tile spills registers and collapses occupancy (ncu: RB=4
# BN=128 -> 210 reg/thread, 12% warps active, 30% DRAM). Bandwidth here comes from device-wide
# TMA/load parallelism = occupancy (see the tma_occupancy_not_pipelining note), NOT from wider
# coalesced stores -- shrinking the tile *worsens* store coalescing yet nearly doubles BW (RB=1
# BN=64 W=1 -> 69 reg/thread, 40% warps active, 57% DRAM). So we autotune RB (not fix it) and
# include few-warp configs. Grid (both paths, from the wrapper): the input padded to whole 128x128
# blocks -- (mpad//(RB*32), cdiv(npad, BN)). A 128(M)x128(N) input block == one swizzle atom of the
# transposed scale, so the padded grid aligns the swizzle scatter; on the plain path the extra
# padded programs are just masked out. MRAG=(M%128!=0): when False the row grid divides M exactly
# (m_mask/s_cols all-true), so we drop those masks for cleaner codegen; n_mask stays always.
# ---------------------------------------------------------------------------
_DIM_M_CONFIGS = [
    triton.Config({"BN": bn, "RB": rb}, num_warps=w)
    for rb in (1, 2, 4)
    for bn in (32, 64, 128, 256)
    for w in (1, 2, 4)
]


@triton.autotune(configs=_DIM_M_CONFIGS, key=["M", "N"])
@triton.jit
def _mxfp8_dim_m_kernel(
    x_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, ssm, ssn, NCB, NRB,
    SWIZZLE: tl.constexpr, MRAG: tl.constexpr, BN: tl.constexpr, RB: tl.constexpr,
):
    pid_rb = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_rb * (RB * 32) + tl.arange(0, RB * 32)  # 128 rows
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    # SWIZZLE grid spans padded rows so m_mask is always needed; the plain aligned path (MRAG False)
    # drops m_mask (n_mask alone) to keep the unpredicated codegen. n_mask stays always.
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn,
        mask=(m_mask[:, None] & n_mask[None, :]) if (SWIZZLE or MRAG) else n_mask[None, :], other=0.0,
    ).to(tl.float32)  # (RB*32, BN); padded rows (offs_m >= M) read as 0
    xr = tl.reshape(x, (RB, 32, BN))
    amax = tl.max(tl.abs(xr), axis=1)  # (RB, BN): per (row-block, col)
    biased = _amax_to_e8m0_cvt(amax)  # (RB, BN); hardware cvt.rp e8m0
    rcp = _e8m0_to_reciprocal_fp32_tl(biased)
    y = tl.reshape((xr * rcp[:, None, :]).to(tl.float8e4nv), (RB * 32, BN))  # (128, BN)
    # transposed qdata store into (N, M): out[n, m] = y[m_in_tile, n]; 128-wide contiguous per row.
    out_off = offs_n[:, None] * sym + offs_m[None, :] * syn
    tl.store(y_ptr + out_off, tl.trans(y),
             mask=(n_mask[:, None] & m_mask[None, :]) if (SWIZZLE or MRAG) else n_mask[:, None])
    if SWIZZLE:
        # swizzled scale scatter into (nrb, ncb, 32, 16). Pre-swizzle transposed-frame position:
        # row = n (over N), col = 32-row-block index (over M//32). Padded transposed rows (n >= N) and
        # padded cols (col >= M//32) are written literal 0 to match gold's zero-pad; the store mask is
        # OOB-safety only (offs_n >= NRB*128 -> out-of-buffer flat).
        row = offs_n[None, :]                             # (1, BN)  transposed row
        col = (pid_rb * RB + tl.arange(0, RB))[:, None]   # (RB, 1)  32-row-block index
        br = row // 128
        r128 = row % 128
        a = r128 // 32
        b = r128 % 32
        bc = col // 4
        c4 = col % 4
        flat = ((br * NCB + bc) * 32 + b) * 16 + (a * 4 + c4)  # (RB, BN)
        s_bytes = tl.where(n_mask[None, :] & (col < (M // 32)), biased.to(tl.uint8), 0)
        tl.store(s_ptr + flat, s_bytes, mask=(offs_n < (NRB * 128))[None, :])
    else:
        # transposed scale store into (N, M//32): out_scale[n, pid_rb*RB + rb] = biased[rb, n]. The
        # (N, M//32) buffer is exactly sized, so no zeroing: just skip padded block-cols (s_cols >= M//32).
        s_cols = pid_rb * RB + tl.arange(0, RB)
        tl.store(
            s_ptr + offs_n[:, None] * ssm + s_cols[None, :] * ssn, tl.trans(biased.to(tl.uint8)),
            mask=(n_mask[:, None] & (s_cols < (M // 32))[None, :]) if MRAG else n_mask[:, None],
        )


def mxfp8_dim_m_triton(x, swizzle, **kwargs):
    """mxfp8 dim-M cast via _mxfp8_dim_m_kernel (transposed (N, M) qdata). swizzle=False: plain
    transposed (N, M//32) e8m0 scale. swizzle=True: scale written into the NVIDIA-swizzled
    (nrb, ncb, 32, 16) block layout (mirrors mxfp8_dim_m_swizzle_f). ONE grid for both paths: tile the
    input padded to whole 128x128 blocks (a 128(M)x128(N) input block == one swizzle atom of the
    transposed scale). Requires M % 32 == 0. Returns (qdata, scale)."""
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0, "mxfp8_dim_m fast kernel needs M%32==0"
    y = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    xsm, xsn, ysm, ysn = x.stride(0), x.stride(1), y.stride(0), y.stride(1)
    mpad = ((M + 127) // 128) * 128
    npad = ((N + 127) // 128) * 128
    grid = lambda meta: (mpad // (meta["RB"] * 32), triton.cdiv(npad, meta["BN"]))  # noqa: E731
    mrag = M % 128 != 0
    if swizzle:
        nrb = npad // 128  # swizzle scale rows (over N)
        ncb = mpad // 128  # swizzle scale col-blocks (over M//32); == ceil((M//32)/4)
        # torch.empty (not zeros): the padded grid writes every slot -- real e8m0 bytes or literal 0.
        s_u8 = torch.empty(nrb, ncb, 32, 16, dtype=torch.uint8, device=x.device)
        _mxfp8_dim_m_kernel[grid](
            x, y, s_u8, M, N, xsm, xsn, ysm, ysn, 0, 0, ncb, nrb,
            SWIZZLE=True, MRAG=mrag,  # ssm/ssn unused; MRAG unused on the swizzle path
        )
    else:
        s_u8 = torch.empty((N, M // 32), dtype=torch.uint8, device=x.device)
        _mxfp8_dim_m_kernel[grid](
            x, y, s_u8, M, N, xsm, xsn, ysm, ysn, s_u8.stride(0), s_u8.stride(1), 0, 0,
            SWIZZLE=False, MRAG=mrag,  # NCB/NRB unused when not swizzled
        )
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_DIM_M = QuantCastTritonRecipe.from_gold(
    Mxfp8DimMGold, triton_fn=functools.partial(mxfp8_dim_m_triton, swizzle=False)
)
MXFP8_DIM_M_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Mxfp8DimMSwizzleGold, triton_fn=functools.partial(mxfp8_dim_m_triton, swizzle=True)
)


# ---------------------------------------------------------------------------
# mxfp8 in BOTH directions, ONE pass. Each program owns a (RB*32) x BN tile of x (read once)
# and reduces it both ways: dim-K = 1x32 blocks along columns (reshape (BM, BN//32, 32), reduce the
# 32), dim-M = 32x1 blocks along rows (reshape (RB, 32, BN), reduce the 32). Emits 4 outputs: qdata_k
# (M,N)/scale_k (M,N//32) like mxfp8, and qdata_m (N,M)/scale_m (N,M//32) like mxfp8_dim_m
# (transposed store). Uses the bit-math e8m0 (bit-exact vs gold). Requires M%32==0 and N%32==0.
# SWIZZLE=False writes both scales as plain, exactly-sized 2D buffers (strides ss*); SWIZZLE=True
# scatters both into NVIDIA-swizzled (nrb, ncb, 32, 16) layouts (NCB_K, NCB_M): dim-K pre-swizzle
# (row=m over M, col=32-col-block over N//32), dim-M transposed frame (row=n over N, col=32-row-block
# over M//32), reusing _mxfp8_kernel's flat formula. Mirrors mxfp8_dim_km_f / mxfp8_dim_km_swizzle_f.
# Perf: like mxfp8_dim_m, the transposed dim-M store is the binding cost -- taller tiles (larger RB)
# widen its contiguous runs, wider BN raises work/occupancy; autotune RB/BN/num_warps to trade off.
# Grid (BOTH paths): the input padded to whole 128x128 blocks -- (mpad//(RB*32), npad//BN). RB*32 and
# BN both divide 128 and mpad,npad are multiples of 128, so the grid exactly tiles the padded extent;
# the swizzle scale buffers (sized == the padded slot count) are thus fully written (real e8m0 bytes
# or literal 0 for padding, matching gold's zero-pad) and stay torch.empty-allocatable. RAGGED =
# (M%128!=0 or N%128!=0): when False the grid divides M,N exactly (no padding) so every mask is
# all-true and we pass mask=None for unpredicated codegen; padding only ever occurs when RAGGED, and
# the plain 2D scale buffers are exactly sized so those padded block-cols are just skipped.
# ---------------------------------------------------------------------------
_DIM_KM_CONFIGS = [
    triton.Config({"BN": bn, "RB": rb}, num_warps=w)
    for rb in (1, 2, 4)
    for bn in (32, 64, 128)
    for w in (1, 2, 4)
]


@triton.autotune(configs=_DIM_KM_CONFIGS, key=["M", "N"])
@triton.jit
def _mxfp8_dim_km_kernel(
    x_ptr, yk_ptr, sk_ptr, ym_ptr, sm_ptr, M, N,
    sxm, sxn, sykm, sykn, sskm, sskn, symn, symm, ssmn, ssmm, NCB_K, NCB_M,
    SWIZZLE: tl.constexpr, RAGGED: tl.constexpr, BN: tl.constexpr, RB: tl.constexpr,
):
    BM: tl.constexpr = RB * 32   # rows in the tile
    CB: tl.constexpr = BN // 32  # 32-col blocks in the tile
    pid_m = tl.program_id(0)     # row-block group (BM rows), over Mpad
    pid_n = tl.program_id(1)     # col group (BN cols), over Npad
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    # SWIZZLE always predicates (its swizzle buffers span the padded rows/cols). The plain path only
    # predicates when RAGGED: aligned M%128==0 and N%128==0 -> the grid divides M,N exactly and M//32,
    # N//32 are multiples of RB,CB, so every mask below is all-true -> pass mask=None (unpredicated).
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn,
        mask=(m_mask[:, None] & n_mask[None, :]) if (SWIZZLE or RAGGED) else None,
        other=0.0 if (SWIZZLE or RAGGED) else None,  # Triton forbids `other` without a mask
    ).to(tl.float32)  # (BM, BN); padded rows/cols read as 0. M%32==0/N%32==0 -> each 32-block is
    # wholly real or wholly padded, so a real block's amax never mixes in a padded 0.
    # dim-K: 1x32 blocks along columns -> (BM, CB, 32), reduce the 32. mxfp8 does NOT clamp amax.
    xk = tl.reshape(x, (BM, CB, 32))
    bk = _amax_to_e8m0_tl(tl.max(tl.abs(xk), axis=2))  # (BM, CB) per (row, col-block)
    yk = tl.reshape((xk * _e8m0_to_reciprocal_fp32_tl(bk)[:, :, None]).to(tl.float8e4nv), (BM, BN))
    tl.store(yk_ptr + offs_m[:, None] * sykm + offs_n[None, :] * sykn, yk,
             mask=(m_mask[:, None] & n_mask[None, :]) if (SWIZZLE or RAGGED) else None)
    if SWIZZLE:
        # swizzled sk store: scale (M, N//32); pre-swizzle row = m (over M), col = 32-col-block (over
        # N//32). The grid tiles Mpad x Npad exactly, so (row_k, col_k) covers the whole sk buffer ->
        # full store (no store mask); padded slots (m >= M or col-block >= N//32) get literal 0.
        row_k = offs_m[:, None]                              # (BM, 1)
        col_k = (pid_n * CB + tl.arange(0, CB))[None, :]     # (1, CB)
        r128k = row_k % 128
        flat_k = (((row_k // 128) * NCB_K + col_k // 4) * 32 + r128k % 32) * 16 + (r128k // 32 * 4 + col_k % 4)
        tl.store(sk_ptr + flat_k, tl.where(m_mask[:, None] & (col_k < (N // 32)), bk.to(tl.uint8), 0))
    else:
        # sk (M, N//32) is exactly sized -> no zeroing; skip padded rows and block-cols (>= N//32).
        sk_cols = pid_n * CB + tl.arange(0, CB)
        tl.store(sk_ptr + offs_m[:, None] * sskm + sk_cols[None, :] * sskn, bk.to(tl.uint8),
                 mask=(m_mask[:, None] & (sk_cols < (N // 32))[None, :]) if RAGGED else None)
    # dim-M: 32x1 blocks along rows -> (RB, 32, BN), reduce the 32; transposed store.
    xm = tl.reshape(x, (RB, 32, BN))
    bm = _amax_to_e8m0_tl(tl.max(tl.abs(xm), axis=1))  # (RB, BN) per (row-block, col)
    ym = tl.reshape((xm * _e8m0_to_reciprocal_fp32_tl(bm)[:, None, :]).to(tl.float8e4nv), (BM, BN))
    # out[n, m] = ym[row, col] with n=offs_n[col], m=offs_m[row] -> store tl.trans(ym) into (N, M).
    tl.store(ym_ptr + offs_n[:, None] * symn + offs_m[None, :] * symm, tl.trans(ym),
             mask=(n_mask[:, None] & m_mask[None, :]) if (SWIZZLE or RAGGED) else None)
    if SWIZZLE:
        # swizzled sm store: scale (N, M//32) transposed; pre-swizzle row = n (over N), col = 32-row-
        # block (over M//32). Grid tiles the whole sm buffer -> full store; padded slots get literal 0.
        row_m = offs_n[None, :]                              # (1, BN)
        col_m = (pid_m * RB + tl.arange(0, RB))[:, None]     # (RB, 1)
        r128m = row_m % 128
        flat_m = (((row_m // 128) * NCB_M + col_m // 4) * 32 + r128m % 32) * 16 + (r128m // 32 * 4 + col_m % 4)
        tl.store(sm_ptr + flat_m, tl.where(n_mask[None, :] & (col_m < (M // 32)), bm.to(tl.uint8), 0))
    else:
        # sm (N, M//32) is exactly sized -> no zeroing; skip padded rows and block-cols (>= M//32).
        sm_cols = pid_m * RB + tl.arange(0, RB)
        tl.store(sm_ptr + offs_n[:, None] * ssmn + sm_cols[None, :] * ssmm, tl.trans(bm.to(tl.uint8)),
                 mask=(n_mask[:, None] & (sm_cols < (M // 32))[None, :]) if RAGGED else None)


def mxfp8_dim_km_triton(x, swizzle, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0 and N % 32 == 0, "mxfp8_dim_km kernel needs M%32==0 and N%32==0"
    yk = torch.empty_like(x, dtype=torch.float8_e4m3fn)                    # (M, N)
    ym = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)    # (N, M) transposed
    # ONE grid for swizzle and non-swizzle: tile the input padded to whole 128x128 blocks. RB*32 and BN
    # divide 128 and mpad,npad are multiples of 128, so the grid tiles the padded extent exactly.
    mpad = ((M + 127) // 128) * 128
    npad = ((N + 127) // 128) * 128
    grid = lambda meta: (mpad // (meta["RB"] * 32), npad // meta["BN"])  # noqa: E731
    ragged = M % 128 != 0 or N % 128 != 0
    if swizzle:
        # sk: (M, N//32) swizzled; sm: (N, M//32) swizzled. torch.empty (not zeros): the kernel writes
        # every slot of both padded (nrb, ncb, 32, 16) grids itself (real e8m0 bytes or literal 0).
        ncb_k = ((N // 32) + 3) // 4  # dim-K scale col-blocks (over N//32)
        ncb_m = ((M // 32) + 3) // 4  # dim-M scale col-blocks (over M//32)
        sk = torch.empty(mpad // 128, ncb_k, 32, 16, dtype=torch.uint8, device=x.device)  # rows over M
        sm = torch.empty(npad // 128, ncb_m, 32, 16, dtype=torch.uint8, device=x.device)  # rows over N
        _mxfp8_dim_km_kernel[grid](
            x, yk, sk, ym, sm, M, N,
            x.stride(0), x.stride(1), yk.stride(0), yk.stride(1), 0, 0,
            ym.stride(0), ym.stride(1), 0, 0, ncb_k, ncb_m,  # ssk*/ssm* unused on swizzle path
            SWIZZLE=True, RAGGED=ragged,
        )
    else:
        sk = torch.empty(M, N // 32, dtype=torch.uint8, device=x.device)
        sm = torch.empty(N, M // 32, dtype=torch.uint8, device=x.device)
        _mxfp8_dim_km_kernel[grid](
            x, yk, sk, ym, sm, M, N,
            x.stride(0), x.stride(1), yk.stride(0), yk.stride(1), sk.stride(0), sk.stride(1),
            ym.stride(0), ym.stride(1), sm.stride(0), sm.stride(1), 0, 0,  # NCB_K/NCB_M unused
            SWIZZLE=False, RAGGED=ragged,
        )
    return yk, sk.view(torch.float8_e8m0fnu), ym, sm.view(torch.float8_e8m0fnu)


MXFP8_DIM_KM = QuantCastTritonRecipe.from_gold(
    Mxfp8DimKmGold, triton_fn=functools.partial(mxfp8_dim_km_triton, swizzle=False)
)
MXFP8_DIM_KM_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Mxfp8DimKmSwizzleGold, triton_fn=functools.partial(mxfp8_dim_km_triton, swizzle=True)
)


# ---------------------------------------------------------------------------
# mxfp8 32x32 both directions, one pass, with BOTH e8m0 scales in the swizzled 4D (nrb, ncb, 32, 16)
# grid. The block is SQUARE, so the quantization + single per-block scale are shared: one (RB, CB)
# scale (amax over each 32x32 block via the (RB, 32, CB, 32) double-max), one qdata reused for both
# the dim-K (M,N) store and the dim-M (N,M) transposed store. Only the two scale layouts differ --
# the same block scale expanded over its 32 rows (dim-K: sk (M, N//32)) or 32 cols then transposed
# (dim-M: sm (N, M//32)), each scattered to its swizzled slot (reusing _mxfp8_dim_km_kernel's
# flat formulas). Mirrors mxfp8_32x32_dim_km_swizzle_f. Requires M%32==0, N%32==0. Both swizzle grids
# allocate with torch.empty: the grid spans the PADDED extents Mpad=ceil(M/128)*128,
# Npad=ceil(N/128)*128 so every slot of both buffers is written (real e8m0 bytes or literal 0).
# ---------------------------------------------------------------------------
@triton.autotune(configs=_DIM_KM_CONFIGS, key=["M", "N"])
@triton.jit
def _mxfp8_32x32_dim_km_swizzle_kernel(
    x_ptr, yk_ptr, sk_ptr, ym_ptr, sm_ptr, M, N,
    sxm, sxn, sykm, sykn, symn, symm, NCB_K, NCB_M,
    BN: tl.constexpr, RB: tl.constexpr,
):
    BM: tl.constexpr = RB * 32   # rows in the tile
    CB: tl.constexpr = BN // 32  # 32-col blocks in the tile
    pid_m = tl.program_id(0)     # row-block group (BM rows), over Mpad
    pid_n = tl.program_id(1)     # col group (BN cols), over Npad
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_real = offs_m < M
    n_real = offs_n < N
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn,
        mask=m_real[:, None] & n_real[None, :], other=0.0,
    ).to(tl.float32)  # (BM, BN); padded rows/cols read as 0
    # one scale per 32x32 block: reshape (RB, 32, CB, 32) and reduce both within-block dims. M%32==0
    # and N%32==0 -> each block is wholly real or wholly padded, so a real block's amax never mixes 0.
    xr = tl.reshape(x, (RB, 32, CB, 32))
    b = _amax_to_e8m0_tl(tl.max(tl.max(tl.abs(xr), axis=3), axis=1))  # (RB, CB) per block
    rcp = _e8m0_to_reciprocal_fp32_tl(b)
    q = tl.reshape((xr * rcp[:, None, :, None]).to(tl.float8e4nv), (BM, BN))  # shared qdata
    # dim-K store: qdata (M, N) as-is.
    tl.store(yk_ptr + offs_m[:, None] * sykm + offs_n[None, :] * sykn, q,
             mask=m_real[:, None] & n_real[None, :])
    # dim-M store: qdata transposed into (N, M).
    tl.store(ym_ptr + offs_n[:, None] * symn + offs_m[None, :] * symm, tl.trans(q),
             mask=n_real[:, None] & m_real[None, :])
    # swizzled sk store: scale (M, N//32), block scale expanded over its 32 rows -> (BM, CB). Pre-swizzle
    # row = m (over M), col = 32-col-block (over N//32). Padded slots (m >= M or col-block >= N//32) -> 0.
    b_exp_k = tl.reshape(tl.broadcast_to(b[:, None, :], (RB, 32, CB)), (BM, CB))
    row_k = offs_m[:, None]                              # (BM, 1)
    col_k = (pid_n * CB + tl.arange(0, CB))[None, :]     # (1, CB)
    r128k = row_k % 128
    flat_k = (((row_k // 128) * NCB_K + col_k // 4) * 32 + r128k % 32) * 16 + (r128k // 32 * 4 + col_k % 4)
    real_k = m_real[:, None] & (col_k < (N // 32))
    tl.store(sk_ptr + flat_k, tl.where(real_k, b_exp_k.to(tl.uint8), 0))
    # swizzled sm store: scale (N, M//32) transposed, block scale expanded over its 32 cols -> (RB, BN).
    # Pre-swizzle row = n (over N), col = 32-row-block (over M//32). Padded slots -> 0.
    b_exp_m = tl.reshape(tl.broadcast_to(b[:, :, None], (RB, CB, 32)), (RB, BN))
    row_m = offs_n[None, :]                              # (1, BN)
    col_m = (pid_m * RB + tl.arange(0, RB))[:, None]     # (RB, 1)
    r128m = row_m % 128
    flat_m = (((row_m // 128) * NCB_M + col_m // 4) * 32 + r128m % 32) * 16 + (r128m // 32 * 4 + col_m % 4)
    real_m = (offs_n[None, :] < N) & (col_m < (M // 32))
    tl.store(sm_ptr + flat_m, tl.where(real_m, b_exp_m.to(tl.uint8), 0))


def mxfp8_32x32_dim_km_swizzle_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0 and N % 32 == 0, \
        "mxfp8_32x32_dim_km_swizzle kernel needs M%32==0 and N%32==0"
    yk = torch.empty_like(x, dtype=torch.float8_e4m3fn)                    # (M, N)
    ym = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)    # (N, M) transposed
    # sk: (M, N//32) swizzled; sm: (N, M//32) swizzled. torch.empty (not zeros): the kernel writes
    # every slot of both padded (nrb, ncb, 32, 16) grids itself (real e8m0 bytes or literal 0).
    nrb_k = (M + 127) // 128
    nrb_m = (N + 127) // 128
    ncb_k = ((N // 32) + 3) // 4
    ncb_m = ((M // 32) + 3) // 4
    sk = torch.empty(nrb_k, ncb_k, 32, 16, dtype=torch.uint8, device=x.device)
    sm = torch.empty(nrb_m, ncb_m, 32, 16, dtype=torch.uint8, device=x.device)
    # grid spans the padded extents (multiples of 128, hence of every RB*32 and BN), so every swizzle
    # slot is visited exactly once.
    mpad = nrb_k * 128
    npad = nrb_m * 128
    grid = lambda meta: (triton.cdiv(mpad, meta["RB"] * 32), triton.cdiv(npad, meta["BN"]))  # noqa: E731
    _mxfp8_32x32_dim_km_swizzle_kernel[grid](
        x, yk, sk, ym, sm, M, N,
        x.stride(0), x.stride(1), yk.stride(0), yk.stride(1),
        ym.stride(0), ym.stride(1), ncb_k, ncb_m,
    )
    return yk, sk.view(torch.float8_e8m0fnu), ym, sm.view(torch.float8_e8m0fnu)


MXFP8_32X32_DIM_KM_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Mxfp832x32DimKMSwizzleGold, triton_fn=mxfp8_32x32_dim_km_swizzle_triton
)


# ---------------------------------------------------------------------------
# mxfp8 32x32 like _mxfp8_32x32_dim_km_swizzle_kernel, but WITHOUT the dim-M qdata store: on
# Blackwell torch._scaled_mm takes the second operand row-major, so only the dim-K qdata (M,N) is
# needed and the dim-M frame contributes only its swizzled scale. Same square-block quantization
# (one (RB, CB) scale per 32x32 block, one shared qdata) -> three outputs: qk (M,N), sk (M, N//32)
# swizzled, sm (N, M//32) swizzled. Dropping the transposed-qdata write is the whole point (less
# store traffic). Mirrors mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_f. Requires M%32==0, N%32==0.
# Both swizzle grids allocate with torch.empty: the grid spans the PADDED extents so every slot is
# written (real e8m0 bytes or literal 0).
# ---------------------------------------------------------------------------
@triton.autotune(configs=_DIM_KM_CONFIGS, key=["M", "N"])
@triton.jit
def _mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_kernel(
    x_ptr, yk_ptr, sk_ptr, sm_ptr, M, N,
    sxm, sxn, sykm, sykn, NCB_K, NCB_M,
    BN: tl.constexpr, RB: tl.constexpr,
):
    BM: tl.constexpr = RB * 32   # rows in the tile
    CB: tl.constexpr = BN // 32  # 32-col blocks in the tile
    pid_m = tl.program_id(0)     # row-block group (BM rows), over Mpad
    pid_n = tl.program_id(1)     # col group (BN cols), over Npad
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_real = offs_m < M
    n_real = offs_n < N
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn,
        mask=m_real[:, None] & n_real[None, :], other=0.0,
    ).to(tl.float32)  # (BM, BN); padded rows/cols read as 0
    # one scale per 32x32 block: reshape (RB, 32, CB, 32) and reduce both within-block dims. M%32==0
    # and N%32==0 -> each block is wholly real or wholly padded, so a real block's amax never mixes 0.
    xr = tl.reshape(x, (RB, 32, CB, 32))
    b = _amax_to_e8m0_tl(tl.max(tl.max(tl.abs(xr), axis=3), axis=1))  # (RB, CB) per block
    rcp = _e8m0_to_reciprocal_fp32_tl(b)
    q = tl.reshape((xr * rcp[:, None, :, None]).to(tl.float8e4nv), (BM, BN))  # dim-K qdata only
    # dim-K store: qdata (M, N) as-is. No dim-M qdata store.
    tl.store(yk_ptr + offs_m[:, None] * sykm + offs_n[None, :] * sykn, q,
             mask=m_real[:, None] & n_real[None, :])
    # swizzled sk store: scale (M, N//32), block scale expanded over its 32 rows -> (BM, CB). Pre-swizzle
    # row = m (over M), col = 32-col-block (over N//32). Padded slots (m >= M or col-block >= N//32) -> 0.
    b_exp_k = tl.reshape(tl.broadcast_to(b[:, None, :], (RB, 32, CB)), (BM, CB))
    row_k = offs_m[:, None]                              # (BM, 1)
    col_k = (pid_n * CB + tl.arange(0, CB))[None, :]     # (1, CB)
    r128k = row_k % 128
    flat_k = (((row_k // 128) * NCB_K + col_k // 4) * 32 + r128k % 32) * 16 + (r128k // 32 * 4 + col_k % 4)
    real_k = m_real[:, None] & (col_k < (N // 32))
    tl.store(sk_ptr + flat_k, tl.where(real_k, b_exp_k.to(tl.uint8), 0))
    # swizzled sm store: scale (N, M//32) transposed, block scale expanded over its 32 cols -> (RB, BN).
    # Pre-swizzle row = n (over N), col = 32-row-block (over M//32). Padded slots -> 0.
    b_exp_m = tl.reshape(tl.broadcast_to(b[:, :, None], (RB, CB, 32)), (RB, BN))
    row_m = offs_n[None, :]                              # (1, BN)
    col_m = (pid_m * RB + tl.arange(0, RB))[:, None]     # (RB, 1)
    r128m = row_m % 128
    flat_m = (((row_m // 128) * NCB_M + col_m // 4) * 32 + r128m % 32) * 16 + (r128m // 32 * 4 + col_m % 4)
    real_m = (offs_n[None, :] < N) & (col_m < (M // 32))
    tl.store(sm_ptr + flat_m, tl.where(real_m, b_exp_m.to(tl.uint8), 0))


def mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 32 == 0 and N % 32 == 0, \
        "mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle kernel needs M%32==0 and N%32==0"
    yk = torch.empty_like(x, dtype=torch.float8_e4m3fn)                    # (M, N)
    # sk: (M, N//32) swizzled; sm: (N, M//32) swizzled. torch.empty (not zeros): the kernel writes
    # every slot of both padded (nrb, ncb, 32, 16) grids itself (real e8m0 bytes or literal 0).
    nrb_k = (M + 127) // 128
    nrb_m = (N + 127) // 128
    ncb_k = ((N // 32) + 3) // 4
    ncb_m = ((M // 32) + 3) // 4
    sk = torch.empty(nrb_k, ncb_k, 32, 16, dtype=torch.uint8, device=x.device)
    sm = torch.empty(nrb_m, ncb_m, 32, 16, dtype=torch.uint8, device=x.device)
    # grid spans the padded extents (multiples of 128, hence of every RB*32 and BN), so every swizzle
    # slot is visited exactly once.
    mpad = nrb_k * 128
    npad = nrb_m * 128
    grid = lambda meta: (triton.cdiv(mpad, meta["RB"] * 32), triton.cdiv(npad, meta["BN"]))  # noqa: E731
    _mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_kernel[grid](
        x, yk, sk, sm, M, N,
        x.stride(0), x.stride(1), yk.stride(0), yk.stride(1), ncb_k, ncb_m,
    )
    return yk, sk.view(torch.float8_e8m0fnu), sm.view(torch.float8_e8m0fnu)


MXFP8_32X32_QDATA_DIM_K_SCALE_DIM_KM_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Mxfp832x32QdataDimKScaleDimKMSwizzleGold,
    triton_fn=mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle_triton,
)


# ---------------------------------------------------------------------------
# nvfp4 device helper: fp32 -> fp4 e2m1 4-bit code (RNE, saturate to 6.0). Exact port of
# f32_to_f4_unpacked (utils.py) for ebits=2, mbits=1. Precomputed constants:
#   denorm_mask_float = bitcast(149<<23) = 4194304.0 ; denorm_mask_int = 1250951168
#   val_to_add = ((1-127)<<23) + ((1<<21)-1) = -1054867457
# ---------------------------------------------------------------------------
@triton.jit
def _f32_to_f4_code_tl(x):
    xu = x.to(tl.uint32, bitcast=True)
    sign = xu & 0x80000000
    absxu = xu ^ sign
    absx = absxu.to(tl.float32, bitcast=True)
    absxi = absxu.to(tl.int32, bitcast=True)

    saturate = absx >= 6.0
    is_denorm = (absx < 1.0) & (~saturate)
    is_normal = (absx >= 1.0) & (~saturate)

    denormal_code = (absx + 4194304.0).to(tl.int32, bitcast=True) - 1250951168
    mant_odd = (absxi >> 22) & 1
    normal_code = (absxi + (-1054867457) + mant_odd) >> 22

    code = tl.where(is_normal, normal_code, 7)  # saturate -> max_int (7)
    code = tl.where(is_denorm, denormal_code, code)

    sign_lp = (sign >> 28).to(tl.int32) & 8
    return (code | sign_lp) & 0xF


# --- MSLK-derived helpers (ported from meta-pytorch/MSLK mslk/quantize/triton/fp4_quantize.py) ---
@triton.jit
def _nvfp4_scale_swizzle_offsets(offs_m):
    # within-atom (128x4) swizzle offsets for rows `offs_m` (cols broadcast over arange(4)); a
    # 128x4 layout is 32 4x4 sub-layouts. Equals the 4D (32,16) flatten used by _to_blocked_4d.
    sub_layout_off = (offs_m % 32) * 16
    sub_layout_row = offs_m // 32
    return sub_layout_off + sub_layout_row * 4 + tl.arange(0, 4)[None, :]


@triton.jit
def _convert_fp32_to_fp4_packed(x_pairs):
    # hardware fp32 -> packed fp4 e2m1 (RNE, saturating), two values per byte (first->low nibble,
    # second->high nibble). Verbatim from MSLK's convert_fp32_to_fp4_packed.
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;
        cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints=("=r,r,r,r,r,r,r,r,r"),
        args=x_pairs,
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


# ---------------------------------------------------------------------------
# nvfp4 with a per-tensor (global) outer scale: 1x16 inner blocks, e4m3 inner scale, fp4-packed
# qdata. Mirrors nvfp4_gs_f / nvfp4_gs_swizzle_f, restructured after MSLK's triton_quantize_nvfp4
# kernel: each program handles one 128x4 swizzle atom = 128 rows x 64 cols (= 4 inner groups), so
# the scale store is a coherent per-atom write and the fp4 encode uses the hardware
# `cvt.rn.satfinite.e2m1x2.f32`. Requires M % 128 == 0 and N % 64 == 0.
# Numerics: bit-exact vs the gold reference. The inner e4m3 scale, reciprocal, and data-scaling use
# div.rn / reciprocal-multiply to mirror torch's per-op rounding exactly (see the note below), and
# the hardware fp4 cvt then matches gold's software f32_to_f4_unpacked on the resulting values.
# ROUNDING selects the fp4 rounding, all keyed on the gold's Philox stream so every mode is bit-exact:
#   * "rtne"      -- round-to-nearest-even (the plain cvt.rn.satfinite.e2m1x2.f32 pack).
#   * "sr"        -- software stochastic rounding: dither the 22 dropped mantissa bits + truncate,
#                    then the same RNE pack (a no-op on the now-on-grid normals).
#   * "nvidia_sr" -- hardware stochastic rounding via the Blackwell cvt.rs.satfinite.e2m1x4.f32
#                    intrinsic (one 32-bit Philox word per group of 4 elements).
# SWIZZLE=False stores the (128, 4) scale tile into its natural row-major (M, N//16) buffer (strides
# ss*); SWIZZLE=True writes it as a coherent per-atom store into the swizzled 4D grid (nrb, ncb, 32,
# 16) at flat offset (pid_m*NCB + pid_n)*512. Everything else (quant + fp4 pack + qdata store) is
# identical. Grid (BOTH paths, expressed in 128x64 atoms): (N // 64, M // 128); M%128==0 and N%64==0
# so it tiles the input exactly (no padding / masks on either path).
# ---------------------------------------------------------------------------
@triton.jit
def _nvfp4_kernel(x_ptr, outer_ptr, q_ptr, s_ptr, seed_ptr, sxm, sxn, ssm, ssn, M, N, NCB,
                  SWIZZLE: tl.constexpr, ROUNDING: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * 128 + tl.arange(0, 128)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]
    x = tl.load(x_ptr + offs_m * sxm + offs_n * sxn).to(tl.float32)  # (128, 64)
    x_blocks = x.reshape(128, 4, 16)
    amax = tl.max(tl.abs(x_blocks), axis=2)  # (128, 4)
    outer = tl.load(outer_ptr)  # per-tensor scalar, 1/S (MSLK/torchao global_scale)
    # Match the gold's fp32 ops bit-for-bit so the e4m3 scale (and downstream fp4) RNE ties resolve
    # identically. `outer` is 1/S, so the gold MULTIPLIES by it. The gold's `/ 6.0` is a
    # tensor-by-PYTHON-SCALAR divide (torch lowers it to a reciprocal-MULTIPLY `* (1/6)`); `* outer` is
    # a correctly-rounded tensor*tensor multiply; and the remaining `outer / inner` is a tensor-by-
    # tensor correctly-rounded div.rn. The default Triton `/` lowers to the approximate div.full.f32
    # (~1 ULP off) and would flip ties -> a wrong e4m3 scale byte, so use div.rn for that quotient.
    inner_val = tl.minimum(tl.maximum(amax * (1.0 / 6.0) * outer, 0.015625), 448.0)
    inner_e4 = inner_val.to(tl.float8e4nv)  # (128, 4)
    recip = tl.div_rn(outer, inner_e4.to(tl.float32))  # (128, 4)
    x_blocks = x_blocks * recip[:, :, None]  # (128, 4, 16); RNE path lets cvt saturate to +-6
    if SWIZZLE:
        # coherent swizzled scale store: atom (pid_m, pid_n) at flat offset (pid_m*NCB + pid_n)*512.
        layout_off = (pid_m * NCB + pid_n) * (128 * 4)
        scale_offs = layout_off + _nvfp4_scale_swizzle_offsets(tl.arange(0, 128)[:, None])
        tl.store(s_ptr + scale_offs, inner_e4)
    else:
        # plain row-major scale store: this atom's (128, 4) scale tile lands at rows offs_m, cols
        # pid_n*4 + [0,4) of the (M, N//16) buffer (no swizzle).
        s_offs_n = pid_n * 4 + tl.arange(0, 4)[None, :]
        tl.store(s_ptr + offs_m * ssm + s_offs_n * ssn, inner_e4)
    if ROUNDING == "nvidia_sr":
        # Hardware SR via the Blackwell `cvt.rs.satfinite.e2m1x4.f32` intrinsic (mirrors gold
        # `_f32_to_packed_fp4_nvidia_sr`): 4 f32 lanes + one 32-bit random word -> one b16 (4 packed
        # e2m1 nibbles). The intrinsic does the whole SR round + saturate itself (it slices the word
        # byte-interleaved across the 4 lanes internally), so we feed the scaled data unclamped.
        seed, base = tl.split(tl.load(seed_ptr + tl.arange(0, 2)))  # full key: key[0] seed, key[1] base
        # One random WORD per group of 4 elements (not per element as the software path). The gold uses
        # word gf -> philox(seed, key[1] + gf//4)[perm[gf%4]] with perm=[0,2,1,3]; the hardware interleave
        # `interleave(interleave(r0,r1),interleave(r2,r3))` lays exactly that perm across each block of
        # 4 groups. Global group gf = f>>2 for f = offs_m*N + col; N%64==0 => row_base = offs_m*N +
        # pid_n*64 is a multiple of 16, so gf//4 == (row_base>>4) + (grp>>2) and gf%4 == grp%4 -- i.e.
        # the 16 row-local groups split cleanly into 4 Philox counters starting at key[1] + row_base>>4.
        grp_ctr = base + (((offs_m * N + pid_n * 64) >> 4) + tl.arange(0, 4)[None, :]).to(tl.int64)  # (128, 4)
        r0, r1, r2, r3 = tl.randint4x(seed, grp_ctr)  # 4 words per counter, each (128, 4)
        rbits = tl.interleave(tl.interleave(r0, r1), tl.interleave(r2, r3))  # (128, 16) one word/group
        # Split the 64 cols into the 4 lanes of each group: reshape (128,4,16)->(128,16,2,2) so
        # [row, grp, hi, lo] is col 4*grp + 2*hi + lo, then nested splits peel cols {0,1,2,3} per group.
        g4 = x_blocks.reshape(128, 16, 2, 2)
        p_lo0, p_lo1 = g4.split()   # lo=0 -> cols {0,2}; lo=1 -> cols {1,3}; each (128,16,2)
        c0, c2 = p_lo0.split()      # (128,16): group cols 0 and 2
        c1, c3 = p_lo1.split()      # (128,16): group cols 1 and 3
        # e2m1x4 packs {$1..$4} with $1 in the HIGH nibble => little-endian b16 = [$4,$3,$2,$1], so feed
        # the cols reversed to store the quad in element order [col0,col1,col2,col3].
        q16 = tl.inline_asm_elementwise(
            asm="cvt.rs.satfinite.e2m1x4.f32 $0, {$1, $2, $3, $4}, $5;",
            constraints="=h,f,f,f,f,r", args=[c3, c2, c1, c0, rbits],
            dtype=tl.int16, is_pure=True, pack=1,
        )  # (128, 16): each int16 = 4 e2m1 nibbles = 2 packed bytes for one group
        qu = q16.to(tl.uint16, bitcast=True)
        # int16 group -> its 2 little-endian bytes [low=cols0/1, high=cols2/3]; interleave restores the
        # (128, 32) packed-byte layout the RNE/sr paths produce (byte 2g=low, 2g+1=high).
        q = tl.interleave((qu & 0xFF).to(tl.uint8), (qu >> 8).to(tl.uint8))  # (128, 32) packed bytes
    else:
        if ROUNDING == "sr":
            # Software stochastic rounding (mirrors gold `_f32_to_packed_fp4_sr`): clamp to the fp4
            # range, dither the 22 mantissa bits fp32->e2m1 drops with a uniform 22-bit value, then
            # truncate them off. The hardware cvt below then RNE-encodes the result -- a no-op for
            # normals (which now sit exactly on the e2m1 grid), and identical to gold's software encode
            # on the subnormal ties the truncation creates (verified bit-exact once the scale uses
            # div.rn above). The gold clamps BEFORE dithering, so we do too. Bit-matches via the same
            # Philox stream.
            x_blocks = tl.minimum(tl.maximum(x_blocks, -6.0), 6.0)
            seed, base = tl.split(tl.load(seed_ptr + tl.arange(0, 2)))  # full key: key[0] seed, key[1] base
            # Key the dither on each element's GLOBAL row-major flat index f = offs_m*N + col, matching
            # the gold's `prng.bits(key, data_scaled.shape)` fill order (data_scaled is a contiguous
            # reshape of x). N%64==0 so row_base = offs_m*N + pid_n*64 is a multiple of 4, hence f>>2 ==
            # (row_base>>2) + (col>>2) and f&3 == col&3. prng.bits honors key[1] as a base counter
            # offset, so the counter is key[1] + f>>2. randint4x on the per-4-element group counter,
            # straight (r0,r1,r2,r3) lane order (same interleave idiom as _sr_bf16_global_kernel), gives
            # rand[row, 4g+lane] == philox(seed, key[1]+f>>2)[f&3] -- bit-for-bit the gold's per-element draw.
            grp_counter = base + (((offs_m * N + pid_n * 64) >> 2) + tl.arange(0, 16)[None, :]).to(tl.int64)  # (128, 16)
            r0, r1, r2, r3 = tl.randint4x(seed, grp_counter)  # 4 streams, each (128, 16)
            rand = tl.interleave(tl.interleave(r0, r2), tl.interleave(r1, r3))  # (128,64) -> (r0,r1,r2,r3)
            dither = (rand & 0x3FFFFF).to(tl.int32).reshape(128, 4, 16)  # uniform low-22-bit dither
            xi = (x_blocks.to(tl.int32, bitcast=True) + dither) & -4194304  # add, truncate low 22 bits
            x_blocks = xi.to(tl.float32, bitcast=True)  # dithered+truncated data for the shared HW pack
        # hardware fp4 pack (RNE + saturate): (128,4,16) -> (128,32,2) pairs -> (128,32) packed bytes.
        q = _convert_fp32_to_fp4_packed(x_blocks.reshape(128, 32, 2).split())
    q_offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    tl.store(q_ptr + offs_m * (N // 2) + q_offs_n, q)


def nvfp4_triton(x, outer_scale, key=None, swizzle=False, rounding="rtne", **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0 and N % 64 == 0, "MSLK-style nvfp4 kernel needs M%128==0 and N%64==0"
    q = torch.empty(M, N // 2, dtype=torch.uint8, device=x.device)
    # Both SR modes key the dither on the (device-resident) full Philox key [key[0], key[1]] (see the
    # SR branches in _nvfp4_kernel); the RNE path never loads seed_ptr, so reuse outer_scale as a
    # harmless placeholder pointer there.
    seed = key.reshape(-1).view(torch.int64) if rounding in ("sr", "nvidia_sr") else outer_scale
    grid = (N // 64, M // 128)  # ONE grid for both paths: one 128x64 atom per program
    if swizzle:
        ncb = (N // 16) // 4  # == N // 64
        s = torch.empty(M // 128, ncb, 32, 16, dtype=torch.float8_e4m3fn, device=x.device)
        _nvfp4_kernel[grid](x, outer_scale, q, s, seed, x.stride(0), x.stride(1), 0, 0, M, N, ncb,
                            SWIZZLE=True, ROUNDING=rounding)  # ss* unused on swizzle path
    else:
        s = torch.empty(M, N // 16, dtype=torch.float8_e4m3fn, device=x.device)
        _nvfp4_kernel[grid](x, outer_scale, q, s, seed, x.stride(0), x.stride(1), s.stride(0),
                            s.stride(1), M, N, 0, SWIZZLE=False, ROUNDING=rounding)  # NCB unused
    return q.view(torch.float4_e2m1fn_x2), s


NVFP4 = QuantCastTritonRecipe.from_gold(
    Nvfp4GsGold, triton_fn=functools.partial(nvfp4_triton, swizzle=False)
)
NVFP4_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Nvfp4GsSwizzleGold, triton_fn=functools.partial(nvfp4_triton, swizzle=True)
)
# NVFP4_SWIZZLE with stochastic rounding: same kernel/grid, selected by ROUNDING (see _nvfp4_kernel).
# The gold's aux order is (x, outer_scale, key), so `key` arrives positionally. Bit-matches the gold.
NVFP4_SR_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Nvfp4GsSRSwizzleGold,
    triton_fn=functools.partial(nvfp4_triton, swizzle=True, rounding="sr"),
)
# Same, but hardware SR via the Blackwell cvt.rs.satfinite.e2m1x4.f32 intrinsic (ROUNDING="nvidia_sr").
NVFP4_NVIDIA_SR_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Nvfp4GsNVIDIASRSwizzleGold,
    triton_fn=functools.partial(nvfp4_triton, swizzle=True, rounding="nvidia_sr"),
)


# ---------------------------------------------------------------------------
# nvfp4 with a 128x128-blocked outer scale (Mb, Nb): same as above but the outer scale is looked
# up per (row, 16-group) from its 128x128 block. Mirrors nvfp4_blocked_outer_f.
# Grid: (cdiv(M, BM), N // 16).
# ---------------------------------------------------------------------------
@triton.jit
def _nvfp4_blocked_outer_kernel(
    x_ptr, outer_ptr, q_ptr, s_ptr, M, N, sxm, sxn, qsm, qsn, NB, NCB, BM: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    offs_n = pid_g * 16 + tl.arange(0, 16)
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=m_mask[:, None]).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)  # (BM,)
    # outer scale for this group: block (row//128, (g*16)//128) == (row//128, g//8)
    mb = offs_m // 128
    nb = pid_g // 8
    outer = tl.load(outer_ptr + mb * NB + nb, mask=m_mask)  # (BM,), 1/S per 128x128 block
    # Match the gold's fp32 ops bit-for-bit so e4m3/fp4 RNE ties resolve identically. `outer` is 1/S,
    # so the gold MULTIPLIES by it. Two subtleties:
    #  * `/ 6.0` in the gold is a tensor-by-PYTHON-SCALAR divide, which torch lowers to a
    #    reciprocal-MULTIPLY (`* (1/6)`); `* outer` is a correctly-rounded tensor*tensor multiply, so
    #    mirror both with multiplies.
    #  * the remaining `outer / inner` is tensor-by-tensor in the gold (correctly-rounded div.rn), so
    #    use tl.div_rn (the default Triton `/` lowers to the approximate div.full.f32, ~1 ULP off).
    inner_val = tl.minimum(tl.maximum(amax * (1.0 / 6.0) * outer, 0.015625), 448.0)
    inner_e4 = inner_val.to(tl.float8e4nv)
    recip = tl.div_rn(outer, inner_e4.to(tl.float32))  # (BM,)
    data = tl.minimum(tl.maximum(x * recip[:, None], -6.0), 6.0)
    code = _f32_to_f4_code_tl(data)
    lo, hi = tl.split(tl.reshape(code, (BM, 8, 2)))
    packed = (lo | (hi << 4)).to(tl.uint8)
    q_off = offs_m[:, None] * qsm + (pid_g * 8 + tl.arange(0, 8))[None, :] * qsn
    tl.store(q_ptr + q_off, packed, mask=m_mask[:, None])
    br = offs_m // 128
    r128 = offs_m % 128
    a = r128 // 32
    b = r128 % 32
    bc = pid_g // 4
    c4 = pid_g % 4
    flat = ((br * NCB + bc) * 32 + b) * 16 + (a * 4 + c4)
    tl.store(s_ptr + flat, inner_e4.to(tl.uint8, bitcast=True), mask=m_mask)


def nvfp4_blocked_outer_triton(x, outer_blocked, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    # %128 is load-bearing here (not just a swizzle shortcut, unlike the other swizzle kernels): the
    # 128x128 outer-block scale itself requires it -- the outer-scale input is a per-128x128-block
    # reduction (nvfp4_blocked_outer_scale reshapes (Mb,128,Nb,128)) and this kernel indexes it via
    # offs_m//128 / pid_g//8, so a non-128-multiple M/N would read out-of-bounds outer scales. It also
    # makes the swizzled inner-scale grid exact (no padded rows/cols), so torch.empty below is safe.
    assert M % 128 == 0 and N % 128 == 0, "nvfp4_blocked_outer kernel needs M%128==0 and N%128==0"
    q = torch.empty(M, N // 2, dtype=torch.uint8, device=x.device)
    n_scale_cols = N // 16
    nrb = (M + 127) // 128
    ncb = (n_scale_cols + 3) // 4
    s = torch.empty(nrb, ncb, 32, 16, dtype=torch.uint8, device=x.device)
    outer_blocked = outer_blocked.contiguous()
    grid = (triton.cdiv(M, 64), N // 16)
    _nvfp4_blocked_outer_kernel[grid](
        x, outer_blocked, q, s, M, N, x.stride(0), x.stride(1), q.stride(0), q.stride(1),
        outer_blocked.stride(0), ncb, BM=64,
    )
    return q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e4m3fn)


NVFP4_BLOCKED_OUTER = QuantCastTritonRecipe.from_gold(
    Nvfp4BlockedOuterGold, triton_fn=nvfp4_blocked_outer_triton
)


# ---------------------------------------------------------------------------
# 16x16 randomized Hadamard transform (bf16 in, bf16 out, no scale). Mirrors hadamard_rht_f:
# reshape the last dim into groups of 16 and right-multiply each group by the 16x16 RHT matrix
# (`out = x.reshape(..., 16) @ rht`). The RHT matrix is an explicit input (built once on the host);
# the kernel just reads it. We flatten x to (n_groups, 16) -- every 16 contiguous elements along the
# last dim form one group -- and give each program a BLOCK_G x 16 tile, so the whole thing is a batch
# of (BLOCK_G, 16) @ (16, 16) matmuls via tl.dot with fp32 accumulation (matching torch's bf16 matmul,
# which accumulates in fp32 on tensor cores), cast back to bf16 on store. Memory-bound: 4 bytes moved
# per element (bf16 in + bf16 out), the 16x16 matrix is a negligible one-time read.
# ---------------------------------------------------------------------------
@triton.jit
def _rht_kernel(x_ptr, rht_ptr, y_ptr, n_groups, BLOCK_G: tl.constexpr):
    pid = tl.program_id(0)
    g = pid * BLOCK_G + tl.arange(0, BLOCK_G)  # (BLOCK_G,) group index along the flattened last dim
    gmask = g < n_groups
    cols = tl.arange(0, 16)
    xoff = g[:, None] * 16 + cols[None, :]  # (BLOCK_G, 16): 16 contiguous elements per group
    x = tl.load(x_ptr + xoff, mask=gmask[:, None], other=0.0)  # bf16
    r = tl.load(rht_ptr + cols[:, None] * 16 + cols[None, :])  # (16, 16) RHT matrix, row-major bf16
    out = tl.dot(x, r, out_dtype=tl.float32).to(tl.bfloat16)  # (BLOCK_G, 16), fp32 accum then bf16
    tl.store(y_ptr + xoff, out, mask=gmask[:, None])


def rht_triton(x, rht, **kwargs):
    """16x16 randomized Hadamard transform along the last dim (mirrors `hadamard_rht_f`). `rht` is the
    precomputed 16x16 RHT matrix (an explicit input). Returns a 1-tuple `(out,)` -- no scale."""
    assert x.dtype == torch.bfloat16, f"RHT expects bf16 input, got {x.dtype}"
    assert x.is_contiguous()
    assert x.shape[-1] % 16 == 0, f"last dim {x.shape[-1]} not divisible by 16"
    out = torch.empty_like(x)
    n_groups = x.numel() // 16
    BLOCK_G = 512  # rows per program; big tiles amortize the tiny K=16 dot (swept best on B200)
    def grid(meta):
        return (triton.cdiv(n_groups, meta["BLOCK_G"]),)
    _rht_kernel[grid](x, rht, out, n_groups, BLOCK_G=BLOCK_G)
    return (out,)


BF16_RHT = QuantCastTritonRecipe.from_gold(HadamardRht, triton_fn=rht_triton)


# ---------------------------------------------------------------------------
# Tiling-INVARIANT stochastic-rounding fp32 -> bf16 (mirroring sr_bf16_global_f). SR add-then-truncate:
# dither the 16 mantissa bits fp32->bf16 drops with a uniform 16-bit value, then mask them off. The
# gold recipe keys the dither on each element's GLOBAL index so the draws don't shift with tiling;
# under flex_tile_map it's told the tile's origin/stride (global_row/global_col/num_col) to reconstruct
# that index from a sub-tile. A standalone Triton kernel needs none of that: it receives the whole
# tensor and owns its own blocking, so an element's global index is just its flat position `f` in `x`.
# We key Philox on `counter = f >> 2` -- so the result is invariant to the internal block size (change
# BLOCK and every element still draws the same dither), which is the meaningful sense of
# "tile-invariant" here, and it bit-matches the gold (same single-seed Philox stream).
#
# `counter = f >> 2` lets one Philox counter serve 4 consecutive elements via randint4x's 4 streams
# (Philox runs once per 4 elements). The 4 streams are interleaved back into the contiguous element
# span for a coalesced load/store; the induced per-lane stream permutation is still a deterministic
# function of `f`, so invariance and unbiasedness hold. No materialized key/uniform tensors (vs the
# gold's 4.29 GB key round-trip) -- the RNG is fused in-register.
# ---------------------------------------------------------------------------
@triton.jit
def _sr_bf16_global_kernel(x_ptr, y_ptr, seed_ptr, n_elements, BLOCK: tl.constexpr):
    # Consume the WHOLE Philox key (the int64 [key[0], key[1]] at seed_ptr): `seed` is the full 64-bit
    # key[0] (Triton splits it into the two 32-bit Philox key words k0/k1) and `base` is key[1], the
    # base counter offset (in 4-word Philox blocks) that prng.bits adds to every block index.
    # tl.randint4x lays a 64-bit offset into the low two counter words (c0,c1), matching torch's
    # philox4x32-10 counter, so an advanced key (fold_in / split, which set both words to full 64-bit
    # values) reproduces the gold bit-for-bit.
    seed, base = tl.split(tl.load(seed_ptr + tl.arange(0, 2)))  # int64 key[0] (seed), key[1] (base)
    pid = tl.program_id(0)
    grp = base + (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)  # (BLOCK,) counter = key[1] + f>>2
    r0, r1, r2, r3 = tl.randint4x(seed, grp)  # 4 streams; the group's 4 elements each take one
    # interleave the 4 streams back to the contiguous 4*BLOCK element span -> coalesced ld/st. Element
    # at flat index f gets counter f>>2 (independent of BLOCK) and stream f&3 -- a pure function of f,
    # so the dither is invariant to the launch tiling. Lane order is the STRAIGHT (r0,r1,r2,r3) so
    # element 4c+lane == randint4x(seed,c)[lane], matching prng.bits(key, n) flat (the gold path);
    # the nesting is a free choice (identical PTX -- 4 words are register-local, contiguous stores).
    rand = tl.interleave(tl.interleave(r0, r2), tl.interleave(r1, r3))  # (4*BLOCK,) -> (r0,r1,r2,r3)
    offs = pid * (4 * BLOCK) + tl.arange(0, 4 * BLOCK)  # contiguous global flat indices
    mask = offs < n_elements
    xi = tl.load(x_ptr + offs, mask=mask).to(tl.int32, bitcast=True)
    rand16 = (rand & 0xFFFF).to(tl.int32)  # uniform 16-bit dither; randint4x is uint32
    xi = (xi + rand16) & -65536  # add dither, then truncate the low 16 mantissa bits
    y = xi.to(tl.float32, bitcast=True).to(tl.bfloat16)  # exact: low 16 bits are zero
    tl.store(y_ptr + offs, y, mask=mask)


def sr_bf16_global_triton(x, key, **kwargs):
    """Tiling-invariant fp32 -> bf16 stochastic rounding (mirrors `sr_bf16_global_f`).
    Keys Philox on each element's global flat index, so the output is invariant to
    the internal block size -- no `global_row`/`global_col`/`num_col` needed (those are flex_tile_map
    artifacts; Triton owns its own tiling). No materialized key/uniform tensors. Returns `(out,)`."""
    assert x.dtype == torch.float32, f"SR bf16 expects fp32 input, got {x.dtype}"
    assert x.is_contiguous()
    out = torch.empty_like(x, dtype=torch.bfloat16)
    n = x.numel()
    # Pass the full Philox key on-device: key[0] is the 64-bit seed, key[1] the base counter offset.
    # (prng.key(int) yields [seed<2**32, 0]; fold_in/split set both words to full 64-bit values.)
    seed = key.reshape(-1).view(torch.int64)  # [key[0], key[1]]
    BLOCK = 1024

    def grid(meta):
        return (triton.cdiv(n, 4 * meta["BLOCK"]),)

    _sr_bf16_global_kernel[grid](x, out, seed, n, BLOCK=BLOCK)
    return (out,)


SR_F32_TO_BF16_GLOBAL = QuantCastTritonRecipe.from_gold(
    SrF32ToBf16Global, triton_fn=sr_bf16_global_triton
)


# Order mirrors quant_cast_gold.ALL_RECIPES (skipping the gold entries with no Triton impl:
# mxfp8_bias).
ALL_RECIPES = [
    # elementwise
    ("fp8_tensorwise_precalc_scale", FP8_TENSORWISE_PRECALC_SCALE),
    ("fp8_rowwise_precalc_scale", FP8_ROWWISE_PRECALC_SCALE),
    ("fp8_colwise_precalc_scale", FP8_COLWISE_PRECALC_SCALE),
    # 8-bit 1D, dim-k reduction
    ("mxfp8", MXFP8),
    ("mxfp8_swizzle", MXFP8_SWIZZLE),
    ("fp8_deepseek_1x128", FP8_DEEPSEEK_1X128),
    # 8-bit 1D, dim-m reduction
    ("mxfp8_dim_m", MXFP8_DIM_M),
    ("mxfp8_dim_m_swizzle", MXFP8_DIM_M_SWIZZLE),
    ("fp8_deepseek_1x128_dim_m", FP8_DEEPSEEK_1X128_DIM_M),
    # 8-bit 1D, dim-km reduction
    ("mxfp8_dim_km", MXFP8_DIM_KM),
    ("mxfp8_dim_km_swizzle", MXFP8_DIM_KM_SWIZZLE),
    ("fp8_deepseek_1x128_dim_km", FP8_DEEPSEEK_1X128_DIM_KM),
    # 8-bit 2D
    ("mxfp8_32x32", MXFP8_32X32),
    ("mxfp8_32x32_swizzle", MXFP8_32X32_SWIZZLE),
    ("mxfp8_32x32_dim_m_swizzle", MXFP8_32X32_DIM_M_SWIZZLE),
    ("mxfp8_32x32_dim_km_swizzle", MXFP8_32X32_DIM_KM_SWIZZLE),
    (
        "mxfp8_32x32_qdata_dim_k_scale_dim_km_swizzle",
        MXFP8_32X32_QDATA_DIM_K_SCALE_DIM_KM_SWIZZLE,
    ),
    ("fp8_deepseek_128x128", FP8_DEEPSEEK_128X128),
    # 8-bit rowwise/colwise
    ("fp8_rowwise", FP8_ROWWISE),
    ("fp8_colwise", FP8_COLWISE),
    # 4 bit 1D
    ("nvfp4", NVFP4),
    ("nvfp4_swizzle", NVFP4_SWIZZLE),
    ("nvfp4_sr_swizzle", NVFP4_SR_SWIZZLE),
    ("nvfp4_nvidia_sr_swizzle", NVFP4_NVIDIA_SR_SWIZZLE),
    ("nvfp4_blocked_outer", NVFP4_BLOCKED_OUTER),
    # RHT
    ("bf16_rht", BF16_RHT),
    # stochastic rounding
    ("fp32_to_bf16_sr_global_offsets", SR_F32_TO_BF16_GLOBAL),
]
