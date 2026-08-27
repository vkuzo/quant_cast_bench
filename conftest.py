import os

# Persist Triton autotune timings to disk across pytest processes. Triton disk-caches compiled kernel
# binaries by default, but NOT the autotuner's per-shape benchmark sweep -- that winner is only
# memoized in-process, so every fresh `pytest` run re-benchmarks each @triton.autotune kernel for every
# distinct (M, N), costing seconds per shape. TRITON_CACHE_AUTOTUNING=1 writes the timings to the
# Triton cache dir and reloads them on later runs, skipping the sweep (measured ~2.7s -> ~0.5s per
# shape here). Set before any triton import (conftest loads before test modules) and via setdefault so
# an explicit override from the environment still wins.
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
