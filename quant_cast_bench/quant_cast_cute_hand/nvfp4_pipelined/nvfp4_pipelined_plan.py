"""Launch policy for persistent TMA/UMMA NVFP4 RHT quantization kernels."""


def select_nvfp4_pipelined_tiles_per_cta(
    M: int,
    K: int,
    *,
    stochastic: bool,
    do_dim_k: bool,
) -> int:
    """Choose the number of adjacent 128x128 tiles processed by each CTA."""
    tiles_m = M // 128
    tiles_k = K // 128
    # Amortize fixed RHT/TMEM setup while retaining at least 128 CTAs. Dim-M-only SR benefits from
    # a longer persistent run because it has no row-warp consumer and more epilogue work per tile.
    max_tiles_per_cta = 32 if stochastic and not do_dim_k else 16
    min_ctas = 128
    return next(
        candidate
        for candidate in (32, 16, 8, 4, 2, 1)
        if candidate <= max_tiles_per_cta
        and tiles_k % candidate == 0
        and (candidate == 1 or tiles_m * tiles_k // candidate >= min_ctas)
    )
