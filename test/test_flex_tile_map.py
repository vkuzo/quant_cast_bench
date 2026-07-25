"""Battle-test flex_tile_map against a plain-PyTorch reference, recipe by recipe.

Comparison discipline mirrors flexquant v1/v2 test.py: bit-exact `torch.equal` on both
qdata (compared as fp32) and scale. Recipes live in recipes.py.
"""

import importlib.metadata
import os
import sys

import pytest
import torch
import torch._inductor.exc
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code, run_fw_bw_and_get_code
from torch.testing import FileCheck

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdata_utils import qdata_equal
# Importing the package auto-installs the mm -> flex_gemm post-grad fusion pass (see
# flex_gemm_to_tile_map_fusion._auto_install).
from quant_cast_bench.flex_tile_map.api import (
    AuxKind,
    FlexTileMapBackend,
    OutputKind,
    flex_tile_map,
)
from quant_cast_bench.flex_tile_map.recipes import (
    DEEPSEEK_1X128,
    DEEPSEEK_1X128_DIM_M,
    MXFP8_FLOOR,
    MXFP8_FLOOR_SWIZZLE,
    RECIPES_V2,
    SR_BF16,
    SR_BF16_GLOBAL,
)
from quant_cast_bench.quant_cast_gold.recipes import (
    debug_relu_f,
    deepseek_1x128_dim_m_f,
    mxfp8_32x32_floor_f,
    mxfp8_floor_dim_m_f,
    nvfp4_gs_f,
    nvfp4_gs_scale,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

# The fusion tests exercise flex_gemm's QUACK backend, which emits a fused CuteDSL GEMM+epilogue
# kernel. QUACK needs nvidia-cutlass-dsl >= 4.5.2: older releases have incompatible cutlass.cute
# APIs (OperandMajorMode moved out of the top-level namespace, make_trivial_tiled_mma changed
# signature), so gate on both the import AND the version.
_MIN_CUTEDSL = (4, 5, 2)
try:
    import cutlass  # noqa: F401

    _cutedsl_version = tuple(
        int(x) for x in importlib.metadata.version("nvidia-cutlass-dsl").split(".")[:3]
    )
except (ImportError, importlib.metadata.PackageNotFoundError):
    _cutedsl_version = None

HAS_CUTEDSL = _cutedsl_version is not None and _cutedsl_version >= _MIN_CUTEDSL

SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (10, 0)


# raw-fn correctness (each recipe's pt_ref_fn output clears its own correctness_fn, no
# flex_tile_map) is a gold-package concern and lives in quant_cast_gold/test.py::test_ref_correctness.
# The tests below exercise the flex_tile_map path (INDUCTOR, and INDUCTOR == MANUAL_TILE).


@pytest.mark.parametrize(
    "name, recipe",
    RECIPES_V2,
    ids=[name for name, _ in RECIPES_V2],
)
def test_flex_tile_map_ref_correctness(name, recipe):
    # tests that running correctness_fn on the outputs of flex_tile_map passes

    torch.manual_seed(0)
    # example_input_fn returns the full positional inputs (x, *aux); flex_tile_map takes x as the
    # tiled input and the rest as captured aux_inputs (their tiling given by recipe.aux_kinds).
    inputs = recipe.example_input_fn(512, 512)
    x, aux = inputs[0], inputs[1:]

    outputs = flex_tile_map(
        x,
        recipe.pt_ref_fn,
        aux_inputs=aux,
        aux_kinds=recipe.aux_kinds,
        output_kinds=recipe.output_kinds,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )
    recipe.correctness_fn(inputs, outputs)  # raises AssertionError on failure

@pytest.mark.parametrize(
    "name, recipe",
    RECIPES_V2,
    ids=[name for name, _ in RECIPES_V2],
)
def test_flex_tile_map_backends_keep_numerics(name, recipe):
    # every RecipeV2 is tile-invariant, so the MANUAL_TILE backend must produce bit-identical
    # outputs to INDUCTOR. Compares every output tensor (qdata + any scale/aux outputs)
    # exactly via qdata_equal (packed fp4 and e8m0 scales via their uint8 view; everything else --
    # fp8_e4m3, fp32, 4D swizzle grids -- as a bit-exact fp32 compare).
    #
    # the SR recipes are skipped here; both keep their INDUCTOR-vs-MANUAL_TILE behavior in
    # dedicated tests. sr_bf16 is the NON-tile-invariant counterexample (dither keyed on
    # tile-local order, so MANUAL_TILE != INDUCTOR by design -- test_sr_bf16_tiling_changes_rounding).
    # sr_bf16_global IS tile-invariant (keyed on global position); that equality is asserted by
    # test_sr_bf16_global_tiling_invariant, so it's skipped here too rather than duplicated.
    if name in ("fp32_to_bf16_sr", "fp32_to_bf16_sr_global_offsets"):
        pytest.skip(f"{name}: INDUCTOR-vs-MANUAL_TILE behavior is covered by a dedicated SR test")

    torch.manual_seed(0)
    inputs = recipe.example_input_fn(512, 512)
    x, aux = inputs[0], inputs[1:]

    kw = dict(
        aux_inputs=aux,
        aux_kinds=recipe.aux_kinds,
        output_kinds=recipe.output_kinds,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )
    ref = flex_tile_map(x, recipe.pt_ref_fn, _backend=FlexTileMapBackend.INDUCTOR, **kw)
    tile = flex_tile_map(x, recipe.pt_ref_fn, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert len(ref) == len(tile), f"{name}: output count {len(tile)} != {len(ref)}"
    for i, (r, t) in enumerate(zip(ref, tile)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )
        assert qdata_equal(t, r), f"{name} output {i}: MANUAL_TILE differs from INDUCTOR"


# dim-M deepseek: `f` transposes the tile + reduces last dim, and OutputKind.SWAP_TILE_INDEX
# grid-transposes the placement. Together they reproduce deepseek_1x128_f(x.t()) -- the dim-M
# layout that used to be expressed by the removed global_input_transform=SWAP_0_AND_1_AXES.
_DIM_M_SWAP = (OutputKind.SWAP_TILE_INDEX, OutputKind.SWAP_TILE_INDEX)


# dim-M whole-tensor correctness and INDUCTOR == MANUAL_TILE (square) are covered by the
# generic RECIPES_V2 suite (DEEPSEEK_1X128_DIM_M carries output_kinds=SWAP_TILE_INDEX). The
# non-square case below is kept: it uniquely exercises the grid-transpose with P != Q.
def test_triton_template_relu_eager():
    # uncompiled: the TRITON_TEMPLATE backend calls the HOP, whose eager body runs `f` directly.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    (out,) = flex_tile_map(x, debug_relu_f, _backend=FlexTileMapBackend.TRITON_TEMPLATE)
    torch.testing.assert_close(out, torch.relu(x))


def test_triton_template_pointwise_compiled_raises():
    # A pointwise `f` under TRITON_TEMPLATE has no template lowering (the template is
    # reduction-only), and a HOP `@register_lowering` that raises does NOT gracefully fall back to
    # the eager body -- it hard-errors (InductorError: LoweringException). This replaces a prior
    # test that asserted a graceful fallback; that only ever "passed" on a stale on-disk Inductor
    # cache (fresh, it fails identically). Pointwise casts belong on INDUCTOR / regular Inductor.
    torch.manual_seed(0)
    torch._dynamo.reset()
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    compiled = torch.compile(flex_tile_map)
    with torch._inductor.config.patch(force_disable_caches=True):
        with pytest.raises(torch._inductor.exc.InductorError):
            compiled(x, debug_relu_f, _backend=FlexTileMapBackend.TRITON_TEMPLATE)


def test_triton_template_deepseek_dim_m_compiled():
    # dim-M deepseek exercises the emitter's TRANSPOSED reduction path: the traced `f` splits
    # dim0 into 128-row groups, reduces the MIDDLE axis (amax over rows), then `.t()`s both
    # outputs. FxTritonEmitter lowers the row-group reshape + a tl.trans, and the dim-M template
    # stores the transposed tiles into the (N, M) / (N, M//128) output layouts.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qr, sr = deepseek_1x128_dim_m_f(x)  # eager reference (whole tensor, transposed outputs)

    compiled = torch.compile(flex_tile_map)
    q, s = compiled(x, deepseek_1x128_dim_m_f, _backend=FlexTileMapBackend.TRITON_TEMPLATE)

    assert q.shape == (256, 256) and q.dtype == torch.float8_e4m3fn
    assert s.shape == (256, 2) and s.dtype == torch.float32
    # tile-invariant recipe, so the template result is bit-exact vs the reference.
    assert qdata_equal(q, qr)
    assert torch.equal(s, sr)


def test_triton_template_deepseek_dim_m_non_square_compiled():
    # non-square input exercises the transposed store with P != Q: a 384x512 input reduces down
    # rows and produces (512, 384) qdata / (512, 3) scale.
    torch.manual_seed(0)
    x = torch.randn(384, 512, dtype=torch.bfloat16, device="cuda")

    qr, sr = deepseek_1x128_dim_m_f(x)

    compiled = torch.compile(flex_tile_map)
    q, s = compiled(x, deepseek_1x128_dim_m_f, _backend=FlexTileMapBackend.TRITON_TEMPLATE)

    assert q.shape == (512, 384) and s.shape == (512, 384 // 128)
    assert qdata_equal(q, qr)
    assert torch.equal(s, sr)


def test_triton_template_mxfp8_floor_dim_m_compiled():
    # mxfp8-floor dim-M: same transposed group-reduction shape as deepseek, but a 32-row group and
    # an e8m0 (uint8) power-of-two scale. Exercises the emitter's e8m0 exponent extraction --
    # view.dtype bitcast, bitwise shift/and, isnan, where, full -- and the group-32 template
    # (template_mxfp8_floor_dim_m.py.jinja), selected via the group-keyed template dispatch.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qr, sr = mxfp8_floor_dim_m_f(x)  # eager reference (whole tensor, transposed outputs)

    compiled = torch.compile(flex_tile_map)
    q, s = compiled(x, mxfp8_floor_dim_m_f, _backend=FlexTileMapBackend.TRITON_TEMPLATE)

    assert q.shape == (256, 256) and q.dtype == torch.float8_e4m3fn
    assert s.shape == (256, 256 // 32) and s.dtype == torch.float8_e8m0fnu
    # tile-invariant recipe, so the template result is bit-exact vs the reference (both compared
    # as bytes: qdata is fp8, scale is e8m0).
    assert qdata_equal(q, qr)
    assert qdata_equal(s, sr)


@pytest.mark.skipif(not SM100, reason="nvfp4 hardware fp4 pack (cvt.e2m1x2) requires SM100")
def test_triton_template_nvfp4_compiled():
    # nvfp4 exercises the emitter's dim-K path (reduce ALONG columns in 16-groups, NO transpose) +
    # three new capabilities: (1) a REPLICATE aux operand (the per-tensor fp32 outer scale, threaded
    # through the HOP and loaded once), (2) the fp4 even/odd deinterleave via aten.slice -> tl.split,
    # (3) the SM100 hardware fp4 pack (cvt.rn.satfinite.e2m1x2.f32) via tl.inline_asm_elementwise.
    # Outputs: fp4-packed qdata (M, N//2) + e4m3 inner scale (M, N//16), plain row-major (no swizzle).
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    outer_scale = nvfp4_gs_scale(x)

    qr, sr = nvfp4_gs_f(x, outer_scale)  # eager reference (whole tensor)

    compiled = torch.compile(flex_tile_map)
    q, s = compiled(
        x,
        nvfp4_gs_f,
        aux_inputs=(outer_scale,),
        aux_kinds=(AuxKind.REPLICATE,),
        _backend=FlexTileMapBackend.TRITON_TEMPLATE,
    )

    assert q.shape == (256, 256 // 2) and q.dtype == torch.float4_e2m1fn_x2
    assert s.shape == (256, 256 // 16) and s.dtype == torch.float8_e4m3fn
    # tile-invariant recipe, so the template result is bit-exact vs the reference (qdata compared as
    # packed-fp4 bytes, scale as e4m3 bytes).
    assert qdata_equal(q, qr)
    assert qdata_equal(s, sr)


def test_triton_template_mxfp8_32x32_floor_compiled():
    # mxfp8-floor 32x32 exercises the emitter's block_2d path: the traced `f` splits BOTH dims into
    # 32x32 blocks (a rank-4 reshape + permute swapping the two middle axes), flattens each block to
    # 1024 elements, reduces the whole-block amax to an e8m0 scale, then un-blocks the fp8 qdata back
    # to the input shape -- NO transpose. Outputs: fp8 qdata (M, N) + e8m0 scale (M//32, N//32).
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qr, sr = mxfp8_32x32_floor_f(x)  # eager reference (whole tensor)

    compiled = torch.compile(flex_tile_map)
    q, s = compiled(x, mxfp8_32x32_floor_f, _backend=FlexTileMapBackend.TRITON_TEMPLATE)

    assert q.shape == (256, 256) and q.dtype == torch.float8_e4m3fn
    assert s.shape == (256 // 32, 256 // 32) and s.dtype == torch.float8_e8m0fnu
    # tile-invariant recipe, so the template result is bit-exact vs the reference (qdata compared as
    # fp8 bytes, scale as e8m0 bytes).
    assert qdata_equal(q, qr)
    assert qdata_equal(s, sr)


def test_deepseek_dim_m_non_square():
    # non-square input exercises the grid-transpose (P != Q): a 384x512 input produces a
    # (512, 384) qdata / (512, 3) scale swapped-grid output; INDUCTOR == MANUAL_TILE bit-exact.
    torch.manual_seed(0)
    (x,) = DEEPSEEK_1X128_DIM_M.example_input_fn(384, 512)

    kernel = DEEPSEEK_1X128_DIM_M.pt_ref_fn
    kw = dict(output_kinds=_DIM_M_SWAP, valid_tile_size_fn=DEEPSEEK_1X128_DIM_M.valid_tile_size_fn)
    qr, sr = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.INDUCTOR, **kw)
    qt, st = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)
    assert qr.shape == (512, 384)  # grid-transposed
    assert sr.shape == (512, 384 // 128)
    assert qdata_equal(qt, qr)
    assert torch.equal(st, sr)


# input padding (`pad_input_to_multiple_of`): a ragged input (e.g. LLM decode/prefill token
# dim) is zero-padded up to a multiple so the tile-invariant recipe sees an aligned shape.
# Outputs are returned at the PADDED shape (the swizzle scale grid is 128-row-atom-structured
# and can't be sliced back to an arbitrary original M). Pad multiples are chosen to satisfy
# each recipe's block/atom so the padded shape passes the existing constraint asserts.
def _ceil_to(v, m):
    return ((v + m - 1) // m) * m


def test_valid_tile_size_fn_unsatisfiable_raises_then_pad_fixes():
    # deepseek's predicate (actual[1] % 128 == 0) can't be satisfied on a ragged 512x300 (the
    # 44-wide edge fails, and spanning 300 fails too) -> the tile-size search raises. Padding the
    # columns up to a multiple of 128 makes it satisfiable.
    torch.manual_seed(0)
    (x,) = DEEPSEEK_1X128.example_input_fn(512, 300)

    with pytest.raises(ValueError):
        flex_tile_map(
            x,
            DEEPSEEK_1X128.pt_ref_fn,
            valid_tile_size_fn=DEEPSEEK_1X128.valid_tile_size_fn,
            _backend=FlexTileMapBackend.MANUAL_TILE,
        )

    # pad N 300 -> 384 (multiple of 128); now every tile's column extent is 128-aligned.
    qdata, scale = flex_tile_map(
        x,
        DEEPSEEK_1X128.pt_ref_fn,
        valid_tile_size_fn=DEEPSEEK_1X128.valid_tile_size_fn,
        pad_input_to_multiple_of=(1, 128),
        _backend=FlexTileMapBackend.MANUAL_TILE,
    )
    assert qdata.shape == (512, 384)  # returned at the padded shape


def test_pad_ref_shapes_swizzle():
    # ragged 200x300 padded to (128,128)-multiple -> (256, 384); swizzle grid nrb=2, ncb=3.
    torch.manual_seed(0)
    (x,) = MXFP8_FLOOR_SWIZZLE.example_input_fn(200, 300)
    qdata, scale = flex_tile_map(
        x,
        MXFP8_FLOOR_SWIZZLE.pt_ref_fn,
        pad_input_to_multiple_of=(128, 128),
        valid_tile_size_fn=MXFP8_FLOOR_SWIZZLE.valid_tile_size_fn,
    )
    assert qdata.shape == (256, 384)
    assert scale.shape == (2, 3, 32, 16)


@pytest.mark.parametrize(
    "recipe, pad_to",
    [
        (MXFP8_FLOOR, (1, 32)),
        (MXFP8_FLOOR_SWIZZLE, (128, 128)),
        (DEEPSEEK_1X128, (1, 128)),
    ],
    ids=["mxfp8_floor", "mxfp8_floor_swizzle", "fp8_deepseek_1x128"],
)
def test_pad_backends_match(recipe, pad_to):
    # padded ragged input: MANUAL_TILE must match INDUCTOR bit-exact (padding happens before
    # tiling in both paths, so the two backends see the identical padded tensor).
    torch.manual_seed(0)
    (x,) = recipe.example_input_fn(200, 300)
    kernel = recipe.pt_ref_fn
    kw = dict(
        pad_input_to_multiple_of=pad_to,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )
    qdata_ref, scale_ref = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.INDUCTOR, **kw)
    qdata_tile, scale_tile = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)
    assert qdata_equal(qdata_tile, qdata_ref)
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


def test_pad_matches_manual_pad():
    # padding inside the API == padding the input outside it, then running the recipe.
    torch.manual_seed(0)
    (x,) = MXFP8_FLOOR.example_input_fn(200, 300)
    kernel = MXFP8_FLOOR.pt_ref_fn
    qdata, scale = flex_tile_map(
        x,
        kernel,
        pad_input_to_multiple_of=(1, 32),
        valid_tile_size_fn=MXFP8_FLOOR.valid_tile_size_fn,
    )
    # manual pad: 200 stays (mult of 1), 300 -> 320 (mult of 32); high-edge zero pad.
    x_padded = F.pad(x, (0, _ceil_to(300, 32) - 300, 0, 0))
    qdata_ref, scale_ref = kernel(x_padded)
    assert qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


def test_sr_bf16_tiling_changes_rounding():
    # documents the accepted non-invariance: INDUCTOR vs MANUAL_TILE differ bit-for-bit
    # (tile-local offsets repeat), yet both stay unbiased (mean ~= input).
    torch.manual_seed(0)
    inputs = SR_BF16.example_input_fn(512, 512)  # (x, key); x is the fp32 constant
    x, aux = inputs[0], inputs[1:]
    v = x.flatten()[0].item()

    kw = dict(aux_inputs=aux, aux_kinds=SR_BF16.aux_kinds)
    (out_ref,) = flex_tile_map(x, SR_BF16.pt_ref_fn, _backend=FlexTileMapBackend.INDUCTOR, **kw)
    (out_tile,) = flex_tile_map(x, SR_BF16.pt_ref_fn, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert not torch.equal(out_ref, out_tile)
    assert abs(out_ref.float().mean().item() - v) < 2e-3
    assert abs(out_tile.float().mean().item() - v) < 2e-3


def test_sr_bf16_global_tiling_invariant():
    # the tiling-invariant SR: keyed on GLOBAL element position, so INDUCTOR == MANUAL_TILE
    # bit-for-bit (contrast test_sr_bf16_tiling_changes_rounding, which uses the tile-local key).
    torch.manual_seed(0)
    inputs = SR_BF16_GLOBAL.example_input_fn(512, 512)  # (x, key); x is the fp32 constant
    x, aux = inputs[0], inputs[1:]
    v = x.flatten()[0].item()

    kw = dict(aux_inputs=aux, aux_kinds=SR_BF16_GLOBAL.aux_kinds)
    (out_ref,) = flex_tile_map(x, SR_BF16_GLOBAL.pt_ref_fn, _backend=FlexTileMapBackend.INDUCTOR, **kw)
    (out_tile,) = flex_tile_map(x, SR_BF16_GLOBAL.pt_ref_fn, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert torch.equal(out_ref, out_tile)  # global-position keying is tiling-invariant
    assert abs(out_ref.float().mean().item() - v) < 2e-3  # still unbiased


# ---------------------------------------------------------------------------
# mm + flex_tile_map -> flex_gemm fusion (fwd + bwd), mirroring flex_tile_map_v2/test.py.
# The user writes `c = mm(a, b); d = flex_tile_map(c, f)` and, under torch.compile, an Inductor
# post-grad pass re-fuses the pair into a single flex_gemm. Uses the INDUCTOR backend (the
# fusible BaseHOP path); the hand-rolled Triton-template backend is deliberately not fused.
# ---------------------------------------------------------------------------


def _sqnr(ref, actual):
    """Signal-to-quantization-noise ratio in dB (standard low-precision numerics metric)."""
    ref = ref.double()
    actual = actual.double()
    noise = (ref - actual).pow(2).mean()
    if noise == 0:
        return float("inf")
    return (10 * torch.log10(ref.pow(2).mean() / noise)).item()


def _functional_mlp_act_fn(acc):
    return acc.relu()


class _FunctionalMLPActFn(torch.autograd.Function):
    """The activation of a functional (non-gated) MLP: d = relu(c), expressed via flex_tile_map,
    with the surrounding matmuls (the w1 up-projection and w2 down-projection) kept OUTSIDE.

    Both forward and backward wrap their epilogue in flex_tile_map, so both graphs expose a
    fusible mm -> flex_tile_map pair. The backward is the true VJP of relu: grad_c = grad_out *
    (c > 0), capturing the saved pre-activation c into the epilogue (a lifted freevar).
    """

    @staticmethod
    def forward(ctx, c):
        ctx.save_for_backward(c)
        return flex_tile_map(c, _functional_mlp_act_fn)

    @staticmethod
    def backward(ctx, grad_out):
        (c,) = ctx.saved_tensors
        return flex_tile_map(grad_out, lambda go: go * (c > 0))


@pytest.mark.skipif(
    not (SM100 and HAS_CUTEDSL),
    reason="flex_gemm QUACK fusion requires SM100 + nvidia-cutlass-dsl >= 4.5.2",
)
def test_functional_mlp_only_first_gemm_fuses_forward():
    torch._dynamo.reset()

    # A functional (non-gated) MLP forward: out = relu(x @ w1) @ w2.
    def fn(x, w1, w2):
        c = torch.mm(x, w1)               # up-proj gemm (fused with relu -> one QUACK cutedsl kernel)
        d = _FunctionalMLPActFn.apply(c)  # relu activation (fused into the up-proj gemm)
        return torch.mm(d, w2)            # down-proj gemm (NOT fused -> stays a plain extern mm)

    # forward only: inputs do not require grad
    x = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)

    actual, (code,) = run_and_get_code(
        torch.compile(fn, backend="inductor", fullgraph=True), x, w1, w2
    )
    ref = torch.mm(x, w1).relu() @ w2
    assert _sqnr(ref, actual) > 30.0

    # the up-proj mm+relu fused into a single QUACK cutedsl kernel, so exactly ONE plain extern mm
    # (the down-proj gemm) is left -- had the up-proj gemm NOT fused, there would be two.
    FileCheck().check("cutedsl_").check_count(
        "extern_kernels.mm(", 1, exactly=True
    ).run(code)


@pytest.mark.skipif(
    not (SM100 and HAS_CUTEDSL),
    reason="flex_gemm QUACK fusion requires SM100 + nvidia-cutlass-dsl >= 4.5.2",
)
def test_functional_mlp_fusion_fires_in_forward_and_backward():
    torch._dynamo.reset()

    # A functional (non-gated) MLP: out = relu(x @ w1) @ w2, trained (fwd + bwd).
    def fn(x, w1, w2):
        c = torch.mm(x, w1)
        d = _FunctionalMLPActFn.apply(c)
        return torch.mm(d, w2)

    def make_inputs():
        x = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w1 = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w2 = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        return x, w1, w2

    cx, cw1, cw2 = make_inputs()
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    # run_fw_bw_and_get_code runs fn() then .sum().backward(), returning the wrapper code for the
    # forward AND backward graphs (and populating .grad along the way).
    _, codes = run_fw_bw_and_get_code(lambda: compiled(cx, cw1, cw2))

    ex = cx.detach().clone().requires_grad_(True)
    ew1 = cw1.detach().clone().requires_grad_(True)
    ew2 = cw2.detach().clone().requires_grad_(True)
    fn(ex, ew1, ew2).sum().backward()

    # compiled and eager run in bf16 with different reduction orders, so compare with SQNR.
    for cg, eg, name in [(cx.grad, ex.grad, "x"), (cw1.grad, ew1.grad, "w1"), (cw2.grad, ew2.grad, "w2")]:
        assert _sqnr(eg, cg) > 30.0, f"grad {name} SQNR too low"

    # pin the true VJP of relu: grad_d = grad_e @ w2.T (grad_e = ones from .sum()),
    # grad_c = grad_d * (c > 0), grad_x = grad_c @ w1.T.
    c = ex @ ew1
    grad_d = torch.ones(256, 32, device="cuda", dtype=torch.bfloat16) @ ew2.t()
    grad_c = grad_d * (c > 0)
    grad_x_ref = grad_c @ ew1.t()
    assert _sqnr(grad_x_ref, cx.grad) > 30.0, "grad x vs pinned VJP"

    # the fusion fired in BOTH the forward and backward graphs -> a fused QUACK cutedsl kernel in
    # each of the two returned code strings.
    assert len(codes) == 2, f"expected fwd+bwd code, got {len(codes)}"
    for gcode in codes:
        FileCheck().check("cutedsl_").run(gcode)

    # The forward computes x@w1 exactly ONCE. Its raw product `c` is saved for the backward's
    # (c > 0) mask, but the fused kernel emits `c` as a SECOND output (dual-output epilogue) rather
    # than recomputing it as a separate matmul -- so the forward has exactly ONE plain extern mm
    # (the un-fused down-proj gemm `d @ w2`). Before dual-output fusion this was TWO (the saved-c
    # recompute plus the down-proj mm). The forward graph is the one WITHOUT AOTAutograd
    # `tangents_*` inputs.
    (fwd_code,) = [c for c in codes if "tangents" not in c]
    FileCheck().check("cutedsl_").check_count(
        "extern_kernels.mm(", 1, exactly=True
    ).run(fwd_code)

    # Pin the backward as tightly as the forward. The backward has four matmuls -- grad_d =
    # grad_e @ w2.T, grad_x = grad_c @ w1.T, grad_w1 = x.T @ grad_c, grad_w2 = d.T @ grad_e -- and
    # exactly ONE fuses: the down-proj-derivative `grad_d = grad_e @ w2.T` that feeds the
    # relu-derivative epilogue (`* (c > 0)`), leaving three plain extern mms. (The backward graph is
    # the one taking AOTAutograd `tangents_*` inputs.)
    (bwd_code,) = [c for c in codes if "tangents" in c]
    FileCheck().check("cutedsl_").run(bwd_code)
    FileCheck().check_count("extern_kernels.mm(", 3, exactly=True).run(bwd_code)


@pytest.mark.skipif(
    not (SM100 and HAS_CUTEDSL),
    reason="flex_gemm QUACK fusion requires SM100 + nvidia-cutlass-dsl >= 4.5.2",
)
def test_functional_mlp_fusion_fires_without_autograd_function():
    # Same as test_functional_mlp_fusion_fires_in_forward_and_backward, but WITHOUT wrapping the
    # activation in a torch.autograd.Function: flex_tile_map is applied inline in the forward and
    # autograd is left to differentiate through it. flex_tile_map_inductor_hop is a BaseHOP, so under
    # torch.compile AOTAutograd derives the VJP for free (grad_c = grad_out * (c > 0), capturing c
    # as a lifted freevar) -- yielding a joint graph, and a fusion in BOTH the forward and backward,
    # identical to the hand-written autograd.Function version. This is the README's "after" form
    # (fuse mm + activation directly under @torch.compile).
    #
    # The inline form also works in EAGER: flex_tile_map_inductor wraps a single-tensor epilogue so its
    # BaseHOP subgraph returns a 1-tuple, which is what BaseHOP's backward (create_fw_bw_graph)
    # needs to count outputs correctly -- so the eager fn() below is a valid reference.
    torch._dynamo.reset()

    # out = relu(x @ w1) @ w2, with relu applied inline (no autograd.Function).
    def fn(x, w1, w2):
        c = torch.mm(x, w1)
        d = flex_tile_map(c, _functional_mlp_act_fn)  # no autograd.Function wrapper
        return torch.mm(d, w2)

    def make_inputs():
        x = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w1 = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w2 = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        return x, w1, w2

    cx, cw1, cw2 = make_inputs()
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    _, codes = run_fw_bw_and_get_code(lambda: compiled(cx, cw1, cw2))

    # Reference: run the SAME fn() eagerly (inline flex_tile_map, no autograd.Function). Eager
    # backward through the inline flex_tile_map works, so this exercises both paths against each
    # other.
    ex = cx.detach().clone().requires_grad_(True)
    ew1 = cw1.detach().clone().requires_grad_(True)
    ew2 = cw2.detach().clone().requires_grad_(True)
    fn(ex, ew1, ew2).sum().backward()

    for cg, eg, name in [(cx.grad, ex.grad, "x"), (cw1.grad, ew1.grad, "w1"), (cw2.grad, ew2.grad, "w2")]:
        assert _sqnr(eg, cg) > 30.0, f"grad {name} SQNR too low"

    # fusion fired in both graphs, and the forward's dual-output epilogue keeps x@w1 to a single
    # matmul (one un-fused down-proj gemm + one fused kernel) -- matching the autograd.Function test.
    assert len(codes) == 2, f"expected fwd+bwd code, got {len(codes)}"
    for gcode in codes:
        FileCheck().check("cutedsl_").run(gcode)
    (fwd_code,) = [c for c in codes if "tangents" not in c]
    FileCheck().check("cutedsl_").check_count(
        "extern_kernels.mm(", 1, exactly=True
    ).run(fwd_code)
    (bwd_code,) = [c for c in codes if "tangents" in c]
    FileCheck().check("cutedsl_").run(bwd_code)
    FileCheck().check_count("extern_kernels.mm(", 3, exactly=True).run(bwd_code)


@pytest.mark.skipif(
    not (SM100 and HAS_CUTEDSL),
    reason="flex_gemm QUACK fusion requires SM100 + nvidia-cutlass-dsl >= 4.5.2",
)
def test_compile_manual_tile_raises():
    # MANUAL_TILE is an eager-only debug backend; under torch.compile it must raise.
    torch._dynamo.reset()
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(flex_tile_map, fullgraph=True)
    with pytest.raises((NotImplementedError, torch._dynamo.exc.Unsupported, RuntimeError)):
        compiled(x, debug_relu_f, _backend=FlexTileMapBackend.MANUAL_TILE)


@pytest.mark.skipif(
    not (SM100 and HAS_CUTEDSL),
    reason="flex_gemm QUACK fusion requires SM100 + nvidia-cutlass-dsl >= 4.5.2",
)
def test_reference_compile_no_mm_inlines():
    # INDUCTOR + compile with NO preceding mm: the ref HOP survives stage-1 fusion and is spliced
    # back in (stage 2) as plain pointwise ops, so regular Inductor lowers it. Compiling at all
    # proves the inline happened -- a surviving HOP has no lowering and hard-errors. Guards the
    # benchmark's `--mode compile` path.
    torch._dynamo.reset()
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)

    def fn(x):
        return flex_tile_map(x, lambda a: a.sin(), _backend=FlexTileMapBackend.INDUCTOR)

    out, (code,) = run_and_get_code(
        torch.compile(fn, backend="inductor", fullgraph=True), x
    )
    torch.testing.assert_close(out, x.sin(), rtol=2e-2, atol=2e-2)
    # no mm -> nothing to fuse: it lowers to plain pointwise Inductor, not a QUACK cutedsl gemm.
    FileCheck().check_not("cutedsl_").run(code)
