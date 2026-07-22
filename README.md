# quant_cast_bench

Bench all the casts — reference recipes and benchmarks for single-kernel quantization casts
(fp8 / mxfp8 / nvfp4), across PyTorch (`torch.compile`), Triton, and CuTeDSL backends.

## Installation

Requires a CUDA GPU with a recent PyTorch + Triton (the perf numbers and the CuTeDSL backend assume
a Blackwell B200; `nvidia-cutlass-dsl` is needed only for the `cute` backend). These heavyweight,
environment-specific packages are **not** installed by this project — bring your own working
`torch` / `triton` / `nvidia-cutlass-dsl`.

Editable install (makes `quant_cast_bench` importable anywhere and pulls the light deps
`fire` / `tabulate` / `jinja2`):

```bash
git clone git@github.com:vkuzo/quant_cast_bench.git
cd quant_cast_bench
pip install -e .
```

Or run straight from a checkout without installing — the entry points (`benchmarks/`, `test/`) add
the repo root to `sys.path`, so `import quant_cast_bench...` resolves as long as you run from the
repo root.

## Usage

```bash
# run the tests (references, Triton kernels, CuTeDSL kernels, flexquant)
pytest test/ -q

# run the memory-bandwidth benchmark sweep
python benchmarks/benchmark.py --mode compile   # torch.compile the gold reference fns
python benchmarks/benchmark.py --mode triton    # hand-written Triton kernels
python benchmarks/benchmark.py --mode cute       # CuTeDSL kernels (Blackwell)
```

## Layout

- `quant_cast_bench/` — the importable package: `quant_cast_gold` (plain-PyTorch reference recipes),
  `quant_cast_triton` / `quant_cast_cute` (backend kernels), and the `flexquant` / `flexquant_v3`
  tiling frameworks.
- `benchmarks/` — the benchmark sweep (see `benchmarks/README.md`).
- `test/` — the test suite.

See `quant_cast_bench/quant_cast_gold/README.md` for how the reference recipes are parametrized.
