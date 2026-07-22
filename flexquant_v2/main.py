"""flexquant_v2 prototype.

Starts from Ed's high-level ``tile_map`` primitive and adds quant recipes one at a
time, in Helion, to battle-test the abstraction. See FINDINGS.md for the deltas from
Ed's toy signature that were forced by making it actually run.

Ed's original toy primitive:

    def tile_map(f, x):
      m, n = x.size()
      out, aux = alloc_empty_out_aux(f, x)
      for tile_m, tile_n in hl.tile([m, n]):
        acc = x[tile_m, tile_n]
        out[tile_m, tile_n], aux[tile_m, tile_n] = f(acc)
      return out, aux

The delta: for a real quant recipe (deepseek 1x128) the ``aux`` (scale) output is
*coarser* than ``x`` -- shape (M, N//128), not (M, N). So aux lives in block space,
not the element grid, and Ed's ``aux[tile_m, tile_n] = ...`` does not type-check.
"""

import torch
import helion
import helion.language as hl

# From flexquant v1 recipes.py:15-16
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12


# ---------------------------------------------------------------------------
# The recipe: deepseek fp8 1x128, expressed purely as an `f` for tile_map.
#
# Mirrors v1's _deepseek_fp8_amax_to_scale_fn + _deepseek_fp8_cast_to_dtype_fn
# (recipes.py:45-57). No framework/target branching -- the recipe is just this
# closure. `f` returns (out_local, aux_local); the group of 128 is exposed via
# reshape so the framework can *infer* the block size from f's output shapes
# (option b).
# ---------------------------------------------------------------------------
def deepseek_1x128_f(block):
    """Per-block recipe: given one quant block ``(BM, BLOCK_N)``, reduce dim=-1.

    IMPORTANT (see FINDINGS Hole A): ``f`` must be authored in per-block form and
    must NOT reshape. An earlier version reshaped to expose the group
    (``acc.reshape(m, -1, 128)``) so the block size was *inferable from f*
    (option b), but Helion cannot trace a reshape of a symbolic tile in-kernel
    (``64 not in VR[128,128]``). Reducing dim=-1 on the block lowers fine. The
    consequence: block size is no longer encoded in ``f`` -- tile_map must be told
    it (option c). Helion's constraints push (b) -> (c).
    """
    amax = block.abs().amax(dim=-1, keepdim=True).clamp(min=EPS).to(torch.float32)
    scale = amax / FP8_MAX  # (BM, 1), forward scale
    qdata = (block.to(torch.float32) * (1.0 / scale)).to(torch.float8_e4m3fn)
    return qdata, scale  # (BM, BLOCK_N), (BM, 1)


# Block size the recipe reduces over along N. In the option-(b) design this is
# inferred by probing `f`; kept here as the ground-truth for the pinned-tile
# first cut and for the probe's sanity check.
BLOCK_N = 128


def _probe_specs(f, x, block_shape):
    """Probe f on ONE block to recover per-output (m_div, n_div, dtype).

    Because f is per-block (no grouping reshape -- see Hole A), the block size is
    NOT inferable from f; the caller passes ``block_shape=(bm, bn)`` (option c).
    We run f on a single (bm, bn) block and read each output's shape relative to
    the block to get its divisor:
      - qdata: (bm, bn) -> divisor (1, 1)         [element grid]
      - scale: (bm, 1)  -> divisor (1, bn)        [block grid: reduced along n]
    """
    bm, bn = block_shape
    probe = torch.randn(bm, bn, dtype=x.dtype, device=x.device)
    outs = f(probe)
    specs = []
    for t in outs:
        dm = bm // t.shape[0]
        dn = bn // t.shape[1]
        specs.append((dm, dn, t.dtype))
    return tuple(specs)


# Tile width along N, in units of quant blocks. >1 exercises the tile>block
# slice-write path (Hole B): each tile spans TILE_BLOCKS blocks, sub-reduced
# in-tile, and aux is written as a *column slice*, not a single column.
TILE_BLOCKS = 2


@helion.kernel(
    config=helion.Config(block_sizes=[32, TILE_BLOCKS * BLOCK_N], num_warps=4)
)
def _tile_map_kernel(x: torch.Tensor, f, aux_n_div: hl.constexpr):
    """Generic kernel (Hole A): `f` is a kernel ARGUMENT, called per block.

    Passing `f` as a parameter (not capturing it) is how Helion supports callable
    epilogues -- see examples/matmul.py, matmul_split_k.py. `f` may itself close
    over tensors in the caller's scope; Helion inlines its body during tracing.
    The earlier module-global hack was unnecessary (a kernel may not have free
    vars, but a callable *argument* is fine). See FINDINGS Hole A.

    Nested tiling (Hole B): outer autotuned tile, inner tile pinned to BLOCK_N so
    each `f` call sees exactly one quant block (no symbolic in-tile reshape). Each
    returned output is written by the option-D rule: element grid if n-divisor==1,
    else block grid at ``blk.begin // aux_n_div``.
    """
    M, N = x.size()
    out = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    aux = torch.empty((M, N // aux_n_div), dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([M, N]):
        for blk in hl.tile(tile_n.begin, tile_n.end, block_size=BLOCK_N):
            acc = x[tile_m, blk]  # (BM, BLOCK_N)
            out_local, aux_local = f(acc)  # generic: call f, don't inline
            # option-D writes: out on element grid, aux on block grid.
            out[tile_m, blk] = out_local
            aux[tile_m, blk.begin // aux_n_div] = aux_local.squeeze(-1)
    return out, aux


def tile_map(f, x, *, block_shape):
    """Ed's primitive: option (c) explicit block + option (D) output writes,
    calling an arbitrary per-block `f` passed as a kernel argument (Hole A).

    `block_shape=(bm, bn)` is passed explicitly -- Helion's no-reshape constraint
    forced `f` into per-block form, so the block is no longer inferable from `f`.
    We probe `f` on one block to recover output divisors/dtypes, then run the
    kernel: iterate blocks, call `f(block)`, write each output at
    `coords // divisor` (option D). `f` goes in as a plain argument -- no globals.
    """
    assert x.ndim == 2
    bm, bn = block_shape
    assert bm == 1 and bn == BLOCK_N, (
        f"kernel compiled for (1, {BLOCK_N}) blocks, got {block_shape}"
    )
    specs = _probe_specs(f, x, block_shape)
    aux_n_div = specs[1][1]  # n-divisor of the aux (scale) output
    return _tile_map_kernel(x, f, aux_n_div)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    qdata, scale = tile_map(deepseek_1x128_f, x, block_shape=(1, BLOCK_N))
    print(f"x:     {tuple(x.shape)} {x.dtype}")
    print(f"qdata: {tuple(qdata.shape)} {qdata.dtype}")
    print(f"scale: {tuple(scale.shape)} {scale.dtype}")
    print(f"specs probed from f: {_probe_specs(deepseek_1x128_f, x, (1, BLOCK_N))}")
