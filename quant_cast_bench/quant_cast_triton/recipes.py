"""Triton implementations of the quant_cast_gold recipes.

Each recipe is a `QuantCastTritonRecipe` -- it inherits the gold reference
(`pt_ref_fn`/`correctness_fn`/`example_input_fn`/`perf_description`) from a
`QuantCastSingleKernelGold` and adds `triton_fn`, a Triton-backed implementation of the same
cast. Mirrors flexquant_v3's `RecipeV2` (inherit-from-gold + `from_gold`). test.py grades each
`triton_fn` against its gold `pt_ref_fn`.
"""

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
    Mxfp832x32FloorGold,
    Mxfp8FloorDimKmGold,
    Mxfp8FloorDimMGold,
    Mxfp8FloorGold,
    Mxfp8FloorSwizzleGold,
    Nvfp4BlockedOuterGold,
    Nvfp4GsSwizzleGold,
    QuantCastSingleKernelGold,
    RowwiseFp8Gold,
    RowwisePrecalcGold,
)


@dataclass(frozen=True)
class QuantCastTritonRecipe(QuantCastSingleKernelGold):
    """A gold recipe plus a Triton implementation of its `pt_ref_fn`. Mirrors flexquant_v3's
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
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=n_mask[None, :]).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)  # (BN,) per column
    scale = amax / 448.0
    y = (x * (1.0 / scale)[None, :]).to(tl.float8e4nv)  # (128, BN)
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
    grid = (M // 128, triton.cdiv(N, 64))
    _fp8_deepseek_1x128_dim_m_kernel[grid](
        x, y, s, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        s.stride(0), s.stride(1), BN=64,
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
# e8m0 device helpers (mxfp8). Exact ports of _amax_to_e8m0_floor / _e8m0_to_fp32 (recipes.py)
# so the scale matches the reference bit-for-bit. e8m0 is stored as its uint8 biased-exponent
# byte (the wrapper .view()s it as float8_e8m0fnu).
# ---------------------------------------------------------------------------
@triton.jit
def _amax_to_e8m0_floor_tl(amax):
    # amax: fp32. Returns the e8m0 biased exponent as int32 (caller stores it as uint8).
    i = amax.to(tl.int32, bitcast=True)
    extracted_pow2 = ((i >> 23) & 0xFF) - 127
    unbiased = extracted_pow2 - 8  # - f8e4m3_max_pow2
    unbiased = tl.minimum(tl.maximum(unbiased, -127), 128)
    biased = unbiased + 127
    return tl.where(amax != amax, 255, biased)  # NaN -> 255


@triton.jit
def _amax_to_e8m0_floor_cvt(amax):
    # Blackwell (SM100+) hardware e8m0 FLOOR: `cvt.rz.satfinite.ue8m0x2.f32` rounds toward zero,
    # which for a non-negative amax is exactly floor of the exponent. Prescale by 2**-8 to fold in
    # the f8e4m3 max-pow2 offset, so cvt.rz.ue8m0(amax * 2**-8) = 2**(floor(log2 amax) - 8) as the
    # biased e8m0 exponent -- matching _amax_to_e8m0_floor_tl without the register-heavy bit math.
    # The x2 op packs two e8m0 into a .b16; we feed 0.0 as the high lane and keep the low byte.
    a = (amax * 0.00390625).to(tl.float32)  # 2**-8
    packed = tl.inline_asm_elementwise(
        asm="cvt.rz.satfinite.ue8m0x2.f32 $0, 0f00000000, $1;",
        constraints="=h,f",
        args=[a],
        dtype=tl.int16,
        is_pure=True,
        pack=1,
    )
    return packed.to(tl.int32) & 0xFF


@triton.jit
def _e8m0_to_fp32_tl(biased):
    # biased: int32 e8m0 exponent -> fp32 pow2 factor, clamped to the smallest normal.
    fp = (biased << 23).to(tl.float32, bitcast=True)
    return tl.maximum(fp, 2.0**-126)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR 1x32: one e8m0 scale per (row, 32-col-block). Mirrors mxfp8_floor_f.
# Grid: (cdiv(M, BM), N // 32).
# ---------------------------------------------------------------------------
@triton.jit
def _mxfp8_floor_kernel(x_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, ssm, ssn, BM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M
    mask = m_mask[:, None] & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=mask).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)  # (BM,) -- mxfp8 does NOT clamp amax
    biased = _amax_to_e8m0_floor_tl(amax)
    sfp = _e8m0_to_fp32_tl(biased)
    y = (x / sfp[:, None]).to(tl.float8e4nv)
    tl.store(y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn, y, mask=mask)
    tl.store(s_ptr + offs_m * ssm + pid_b * ssn, biased.to(tl.uint8), mask=m_mask)


def mxfp8_floor_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s_u8 = torch.empty(M, N // 32, dtype=torch.uint8, device=x.device)
    grid = (triton.cdiv(M, 64), N // 32)
    _mxfp8_floor_kernel[grid](
        x, y, s_u8, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        s_u8.stride(0), s_u8.stride(1), BM=64,
    )
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_FLOOR = QuantCastTritonRecipe.from_gold(Mxfp8FloorGold, triton_fn=mxfp8_floor_triton)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR 32x32: one e8m0 scale per 32x32 block. Mirrors mxfp8_32x32_floor_f.
# Perf: one 32x32 block per program is tiny/low-intensity. Batch CB col-blocks per program
# (32 rows x CB*32 cols), reshaping to (32, CB, 32) and reducing the row + within-block dims;
# autotune CB and num_warps. Grid: (M // 32, cdiv(N, CB*32)).
# ---------------------------------------------------------------------------
_MXFP8_32X32_CONFIGS = [
    triton.Config({"CB": cb}, num_warps=w) for cb in (2, 4, 8, 16) for w in (2, 4, 8)
]


@triton.autotune(configs=_MXFP8_32X32_CONFIGS, key=["M", "N"])
@triton.jit
def _mxfp8_32x32_kernel(x_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, ssm, ssn, CB: tl.constexpr):
    pid_rb = tl.program_id(0)  # 32-row block
    pid_cb = tl.program_id(1)  # group of CB 32-col blocks
    offs_m = pid_rb * 32 + tl.arange(0, 32)
    offs_n = pid_cb * (CB * 32) + tl.arange(0, CB * 32)
    n_mask = offs_n < N
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=n_mask[None, :], other=0.0
    ).to(tl.float32)  # (32, CB*32)
    xr = tl.reshape(x, (32, CB, 32))
    amax = tl.max(tl.max(tl.abs(xr), axis=2), axis=0)  # (CB,): within-block cols, then 32 rows
    biased = _amax_to_e8m0_floor_tl(amax)  # (CB,)
    sfp = _e8m0_to_fp32_tl(biased)
    y = tl.reshape((xr / sfp[None, :, None]).to(tl.float8e4nv), (32, CB * 32))
    tl.store(y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn, y, mask=n_mask[None, :])
    s_cols = pid_cb * CB + tl.arange(0, CB)
    tl.store(s_ptr + pid_rb * ssm + s_cols * ssn, biased.to(tl.uint8), mask=s_cols < (N // 32))


def mxfp8_32x32_floor_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s_u8 = torch.empty(M // 32, N // 32, dtype=torch.uint8, device=x.device)
    grid = lambda meta: (M // 32, triton.cdiv(N, meta["CB"] * 32))  # noqa: E731
    _mxfp8_32x32_kernel[grid](
        x, y, s_u8, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        s_u8.stride(0), s_u8.stride(1),
    )
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_32X32_FLOOR = QuantCastTritonRecipe.from_gold(
    Mxfp832x32FloorGold, triton_fn=mxfp8_32x32_floor_triton
)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR dim-M: 32-row blocks down M, one e8m0 scale per (32-row-block, col); transposed
# outputs (N, M) / (N, M//32). Mirrors mxfp8_floor_dim_m_f.
# Perf: process RB 32-row blocks x BN cols per program; reshape (RB*32, BN) -> (RB, 32, BN) and
# reduce the within-block 32. This kernel is memory-bound and OCCUPANCY-limited: the fp32 tile is
# register-heavy, so a large (RB*32, BN) tile spills registers and collapses occupancy (ncu: RB=4
# BN=128 -> 210 reg/thread, 12% warps active, 30% DRAM). Bandwidth here comes from device-wide
# TMA/load parallelism = occupancy (see the tma_occupancy_not_pipelining note), NOT from wider
# coalesced stores -- shrinking the tile *worsens* store coalescing yet nearly doubles BW (RB=1
# BN=64 W=1 -> 69 reg/thread, 40% warps active, 57% DRAM). So we autotune RB (not fix it) and
# include few-warp configs. Requires M % 128 == 0. Grid: (M // (RB*32), cdiv(N, BN)).
# ---------------------------------------------------------------------------
_DIM_M_CONFIGS = [
    triton.Config({"BN": bn, "RB": rb}, num_warps=w)
    for rb in (1, 2, 4)
    for bn in (32, 64, 128, 256)
    for w in (1, 2, 4)
]


@triton.autotune(configs=_DIM_M_CONFIGS, key=["M", "N"])
@triton.jit
def _mxfp8_floor_dim_m_kernel(
    x_ptr, y_ptr, s_ptr, M, N, sxm, sxn, sym, syn, ssm, ssn, BN: tl.constexpr, RB: tl.constexpr
):
    pid_rb = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_rb * (RB * 32) + tl.arange(0, RB * 32)  # 128 rows
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn, mask=n_mask[None, :]
    ).to(tl.float32)  # (128, BN)
    xr = tl.reshape(x, (RB, 32, BN))
    amax = tl.max(tl.abs(xr), axis=1)  # (RB, BN): per (row-block, col)
    biased = _amax_to_e8m0_floor_cvt(amax)  # (RB, BN); hardware cvt.rz e8m0 floor
    sfp = _e8m0_to_fp32_tl(biased)
    y = tl.reshape((xr / sfp[:, None, :]).to(tl.float8e4nv), (RB * 32, BN))  # (128, BN)
    # transposed qdata store into (N, M): out[n, m] = y[m_in_tile, n]; 128-wide contiguous per row.
    out_off = offs_n[:, None] * sym + offs_m[None, :] * syn
    tl.store(y_ptr + out_off, tl.trans(y), mask=n_mask[:, None])
    # transposed scale store into (N, M//32): out_scale[n, pid_rb*RB + rb] = biased[rb, n]
    s_cols = pid_rb * RB + tl.arange(0, RB)
    tl.store(
        s_ptr + offs_n[:, None] * ssm + s_cols[None, :] * ssn, tl.trans(biased.to(tl.uint8)),
        mask=n_mask[:, None],
    )


def mxfp8_floor_dim_m_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0, "mxfp8_floor_dim_m fast kernel needs M%128==0"
    y = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    s_u8 = torch.empty((N, M // 32), dtype=torch.uint8, device=x.device)
    grid = lambda meta: (M // (meta["RB"] * 32), triton.cdiv(N, meta["BN"]))  # noqa: E731
    _mxfp8_floor_dim_m_kernel[grid](
        x, y, s_u8, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        s_u8.stride(0), s_u8.stride(1),
    )
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_FLOOR_DIM_M = QuantCastTritonRecipe.from_gold(
    Mxfp8FloorDimMGold, triton_fn=mxfp8_floor_dim_m_triton
)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR in BOTH directions, ONE pass. Each program owns a (RB*32) x BN tile of x (read once)
# and reduces it both ways: dim-K = 1x32 blocks along columns (reshape (BM, BN//32, 32), reduce the
# 32), dim-M = 32x1 blocks along rows (reshape (RB, 32, BN), reduce the 32). Emits 4 outputs: qdata_k
# (M,N)/scale_k (M,N//32) like mxfp8_floor, and qdata_m (N,M)/scale_m (N,M//32) like mxfp8_floor_dim_m
# (transposed store). Uses the bit-math e8m0 floor (bit-exact vs gold). Requires M%128==0 and N%128==0.
# Perf: like mxfp8_floor_dim_m, the transposed dim-M store is the binding cost -- taller tiles (larger
# RB) widen its contiguous runs, wider BN raises work/occupancy; autotune RB/BN/num_warps to trade
# off (the fixed 32x32 version only reached ~31%). Grid: (M // (RB*32), N // BN).
# ---------------------------------------------------------------------------
_DIM_KM_CONFIGS = [
    triton.Config({"BN": bn, "RB": rb}, num_warps=w)
    for rb in (1, 2, 4)
    for bn in (32, 64, 128)
    for w in (1, 2, 4)
]


@triton.autotune(configs=_DIM_KM_CONFIGS, key=["M", "N"])
@triton.jit
def _mxfp8_floor_dim_km_kernel(
    x_ptr, yk_ptr, sk_ptr, ym_ptr, sm_ptr, M, N,
    sxm, sxn, sykm, sykn, sskm, sskn, symn, symm, ssmn, ssmm,
    BN: tl.constexpr, RB: tl.constexpr,
):
    BM: tl.constexpr = RB * 32   # rows in the tile
    CB: tl.constexpr = BN // 32  # 32-col blocks in the tile
    pid_m = tl.program_id(0)     # row-block group (BM rows)
    pid_n = tl.program_id(1)     # col group (BN cols)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_n[None, :] * sxn).to(tl.float32)  # (BM, BN)
    # dim-K: 1x32 blocks along columns -> (BM, CB, 32), reduce the 32. mxfp8 does NOT clamp amax.
    xk = tl.reshape(x, (BM, CB, 32))
    bk = _amax_to_e8m0_floor_tl(tl.max(tl.abs(xk), axis=2))  # (BM, CB) per (row, col-block)
    yk = tl.reshape((xk / _e8m0_to_fp32_tl(bk)[:, :, None]).to(tl.float8e4nv), (BM, BN))
    tl.store(yk_ptr + offs_m[:, None] * sykm + offs_n[None, :] * sykn, yk)
    sk_cols = pid_n * CB + tl.arange(0, CB)
    tl.store(sk_ptr + offs_m[:, None] * sskm + sk_cols[None, :] * sskn, bk.to(tl.uint8))
    # dim-M: 32x1 blocks along rows -> (RB, 32, BN), reduce the 32; transposed store.
    xm = tl.reshape(x, (RB, 32, BN))
    bm = _amax_to_e8m0_floor_tl(tl.max(tl.abs(xm), axis=1))  # (RB, BN) per (row-block, col)
    ym = tl.reshape((xm / _e8m0_to_fp32_tl(bm)[:, None, :]).to(tl.float8e4nv), (BM, BN))
    # out[n, m] = ym[row, col] with n=offs_n[col], m=offs_m[row] -> store tl.trans(ym) into (N, M).
    tl.store(ym_ptr + offs_n[:, None] * symn + offs_m[None, :] * symm, tl.trans(ym))
    sm_cols = pid_m * RB + tl.arange(0, RB)
    tl.store(sm_ptr + offs_n[:, None] * ssmn + sm_cols[None, :] * ssmm, tl.trans(bm.to(tl.uint8)))


def mxfp8_floor_dim_km_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0, "mxfp8_floor_dim_km kernel needs M%128==0 and N%128==0"
    yk = torch.empty_like(x, dtype=torch.float8_e4m3fn)                    # (M, N)
    sk = torch.empty(M, N // 32, dtype=torch.uint8, device=x.device)
    ym = torch.empty(N, M, dtype=torch.float8_e4m3fn, device=x.device)    # (N, M) transposed
    sm = torch.empty(N, M // 32, dtype=torch.uint8, device=x.device)
    grid = lambda meta: (M // (meta["RB"] * 32), N // meta["BN"])  # noqa: E731
    _mxfp8_floor_dim_km_kernel[grid](
        x, yk, sk, ym, sm, M, N,
        x.stride(0), x.stride(1), yk.stride(0), yk.stride(1), sk.stride(0), sk.stride(1),
        ym.stride(0), ym.stride(1), sm.stride(0), sm.stride(1),
    )
    return yk, sk.view(torch.float8_e8m0fnu), ym, sm.view(torch.float8_e8m0fnu)


MXFP8_FLOOR_DIM_KM = QuantCastTritonRecipe.from_gold(
    Mxfp8FloorDimKmGold, triton_fn=mxfp8_floor_dim_km_triton
)


# ---------------------------------------------------------------------------
# mxfp8 FLOOR 1x32 with the e8m0 scale written directly into the NVIDIA-swizzled 4D block grid
# (nrb, ncb, 32, 16). Same quant as mxfp8_floor; the scale for pre-swizzle position (row, col)
# lands at flat offset ((br*ncb+bc)*32 + b)*16 + (a*4+c4), where br=row//128, r128=row%128,
# a=r128//32, b=r128%32, bc=col//4, c4=col%4 (derived from _to_blocked_4d). Mirrors
# mxfp8_floor_swizzle_f.
#
# Perf: mirror Inductor's codegen -- flatten all (row, 32-group) pairs into one 1-D persistent
# reduction over `n_groups = M * (N//32)`. Each 32-group is exactly a 32-contiguous chunk of the
# row-major input (group g -> flat elements [g*32, g*32+32)), so consecutive groups are
# contiguous and the loads/stores coalesce. Grid: (cdiv(n_groups, GBLOCK),).
# ---------------------------------------------------------------------------
_SWIZZLE_CONFIGS = [
    triton.Config({"GBLOCK": g}, num_warps=w)
    for g in (32, 64, 128, 256, 512, 1024)
    for w in (2, 4, 8)
]


@triton.autotune(configs=_SWIZZLE_CONFIGS, key=["n_groups"])
@triton.jit
def _mxfp8_floor_swizzle_kernel(x_ptr, y_ptr, s_ptr, n_groups, NGC, NCB, GBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    g = pid * GBLOCK + tl.arange(0, GBLOCK)  # flat 32-group indices
    g_mask = g < n_groups
    off = g[:, None] * 32 + tl.arange(0, 32)[None, :]  # (GBLOCK, 32) flat, contiguous per group
    x = tl.load(x_ptr + off, mask=g_mask[:, None]).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)  # (GBLOCK,)
    biased = _amax_to_e8m0_floor_tl(amax)
    sfp = _e8m0_to_fp32_tl(biased)
    y = (x / sfp[:, None]).to(tl.float8e4nv)
    tl.store(y_ptr + off, y, mask=g_mask[:, None])
    # swizzled scale store: pre-swizzle position row = g // NGC, col = g % NGC
    row = g // NGC
    col = g % NGC
    br = row // 128
    r128 = row % 128
    a = r128 // 32
    b = r128 % 32
    bc = col // 4
    c4 = col % 4
    flat = ((br * NCB + bc) * 32 + b) * 16 + (a * 4 + c4)
    tl.store(s_ptr + flat, biased.to(tl.uint8), mask=g_mask)


def mxfp8_floor_swizzle_triton(x, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ngc = N // 32  # 32-groups per row
    n_groups = M * ngc
    nrb = (M + 127) // 128
    ncb = (ngc + 3) // 4
    # zero-filled so any padded (row/col beyond the real grid) positions match gold's zeros.
    s_u8 = torch.zeros(nrb, ncb, 32, 16, dtype=torch.uint8, device=x.device)
    grid = lambda meta: (triton.cdiv(n_groups, meta["GBLOCK"]),)  # noqa: E731
    _mxfp8_floor_swizzle_kernel[grid](x, y, s_u8, n_groups, ngc, ncb)
    return y, s_u8.view(torch.float8_e8m0fnu)


MXFP8_FLOOR_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Mxfp8FloorSwizzleGold, triton_fn=mxfp8_floor_swizzle_triton
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
# qdata, inner scale written to the swizzled 4D grid. Mirrors nvfp4_gs_swizzle_f, restructured
# after MSLK's triton_quantize_nvfp4 kernel: each program handles one 128x4 swizzle atom = 128
# rows x 64 cols (= 4 inner groups), so the scale store is a coherent per-atom write and the fp4
# encode uses the hardware `cvt.rn.satfinite.e2m1x2.f32`. Requires M % 128 == 0 and N % 64 == 0.
# Numerics: the inner e4m3 scale / reciprocal / data-scaling are identical to the gold reference
# (bit-exact); only the fp4 encoding may differ from gold's f32_to_f4_unpacked on rare RNE ties
# (both round-to-nearest-even + saturate to +-6). Grid: (N // 64, M // 128).
# ---------------------------------------------------------------------------
@triton.jit
def _nvfp4_swizzle_kernel(x_ptr, outer_ptr, q_ptr, s_ptr, sxm, sxn, M, N, NCB):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * 128 + tl.arange(0, 128)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]
    x = tl.load(x_ptr + offs_m * sxm + offs_n * sxn).to(tl.float32)  # (128, 64)
    x_blocks = x.reshape(128, 4, 16)
    amax = tl.max(tl.abs(x_blocks), axis=2)  # (128, 4)
    outer = tl.load(outer_ptr)  # per-tensor scalar
    inner_val = tl.minimum(tl.maximum((amax / 6.0) / outer, 0.015625), 448.0)
    inner_e4 = inner_val.to(tl.float8e4nv)  # (128, 4)
    recip = (1.0 / outer) / inner_e4.to(tl.float32)  # (128, 4)
    x_blocks = x_blocks * recip[:, :, None]  # (128, 4, 16); cvt saturates to +-6
    # coherent swizzled scale store: atom (pid_m, pid_n) at flat offset (pid_m*NCB + pid_n)*512.
    layout_off = (pid_m * NCB + pid_n) * (128 * 4)
    scale_offs = layout_off + _nvfp4_scale_swizzle_offsets(tl.arange(0, 128)[:, None])
    tl.store(s_ptr + scale_offs, inner_e4)
    # hardware fp4 pack: (128,4,16) -> (128,32,2) pairs -> (128,32) packed bytes.
    q = _convert_fp32_to_fp4_packed(x_blocks.reshape(128, 32, 2).split())
    q_offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    tl.store(q_ptr + offs_m * (N // 2) + q_offs_n, q)


def nvfp4_swizzle_triton(x, outer_scale, **kwargs):
    assert x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    assert M % 128 == 0 and N % 64 == 0, "MSLK-style nvfp4 kernel needs M%128==0 and N%64==0"
    q = torch.empty(M, N // 2, dtype=torch.uint8, device=x.device)
    nrb = M // 128
    ncb = (N // 16) // 4  # == N // 64
    s = torch.empty(nrb, ncb, 32, 16, dtype=torch.float8_e4m3fn, device=x.device)
    grid = (N // 64, M // 128)
    _nvfp4_swizzle_kernel[grid](x, outer_scale, q, s, x.stride(0), x.stride(1), M, N, ncb)
    return q.view(torch.float4_e2m1fn_x2), s


NVFP4_SWIZZLE = QuantCastTritonRecipe.from_gold(
    Nvfp4GsSwizzleGold, triton_fn=nvfp4_swizzle_triton
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
    outer = tl.load(outer_ptr + mb * NB + nb, mask=m_mask)  # (BM,)
    inner_val = tl.minimum(tl.maximum((amax / 6.0) / outer, 0.015625), 448.0)
    inner_e4 = inner_val.to(tl.float8e4nv)
    recip = (1.0 / outer) / inner_e4.to(tl.float32)  # (BM,)
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
    q = torch.empty(M, N // 2, dtype=torch.uint8, device=x.device)
    n_scale_cols = N // 16
    nrb = (M + 127) // 128
    ncb = (n_scale_cols + 3) // 4
    s = torch.zeros(nrb, ncb, 32, 16, dtype=torch.uint8, device=x.device)
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


# Order mirrors quant_cast_gold.ALL_RECIPES (skipping the gold entries with no Triton impl:
# bf16_rht, fp32_to_bf16_sr, fp32_to_bf16_sr_global_offsets, mxfp8_bias).
ALL_RECIPES = [
    # elementwise
    ("fp8_tensorwise_precalc_scale", FP8_TENSORWISE_PRECALC_SCALE),
    ("fp8_rowwise_precalc_scale", FP8_ROWWISE_PRECALC_SCALE),
    ("fp8_colwise_precalc_scale", FP8_COLWISE_PRECALC_SCALE),
    # 1x32, 8-bit
    ("mxfp8_floor", MXFP8_FLOOR),
    ("mxfp8_floor_swizzle", MXFP8_FLOOR_SWIZZLE),
    ("mxfp8_floor_dim_m", MXFP8_FLOOR_DIM_M),
    ("mxfp8_floor_dim_km", MXFP8_FLOOR_DIM_KM),
    ("mxfp8_32x32_floor", MXFP8_32X32_FLOOR),
    # 1x128, 8-bit
    ("fp8_deepseek_1x128", FP8_DEEPSEEK_1X128),
    ("fp8_deepseek_1x128_dim_m", FP8_DEEPSEEK_1X128_DIM_M),
    ("fp8_deepseek_1x128_dim_km", FP8_DEEPSEEK_1X128_DIM_KM),
    # 128x128, 8-bit
    ("fp8_deepseek_128x128", FP8_DEEPSEEK_128X128),
    # rowwise/colwise, 8-bit
    ("fp8_rowwise", FP8_ROWWISE),
    ("fp8_colwise", FP8_COLWISE),
    # 1x16, 4 bit
    ("nvfp4_swizzle", NVFP4_SWIZZLE),
    ("nvfp4_blocked_outer", NVFP4_BLOCKED_OUTER),
]
