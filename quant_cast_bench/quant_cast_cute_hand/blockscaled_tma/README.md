# Block-scaled TMA review notes

The implementation supports MXFP8, MXFP4, and NVFP4 quantization across
dim-K, dim-M, and dim-KM orientations. Before upstreaming, address the issues
below in priority order.

## Correctness blockers

### Bound NVFP4 dim-KM scale stores

NVFP4 scale storage pads each quantized dimension in 64-element units, while
the dim-KM CTA grid pads the dimensions to 128 elements. On shapes for which
the number of 64-element blocks is odd, CTAs covering the second half of the
final 128-element block can issue packed scale stores beyond the allocated
dim-K and dim-M scale tensors.

For example, with `M = K = 32`, each scale tensor contains one 512-byte block.
A guarded-allocation test observed writes to all 512 bytes immediately after
both scale allocations. Ordinary numerical tests can miss this because the
out-of-bounds writes may land in allocator padding or be overwritten by a
later qdata transfer.

The boundary paths must suppress a packed scale store when its starting scale
column is outside the padded 64-element scale extent. This applies to both
directions of NVFP4 dim-KM.

Relevant locations:

- `blockscale_tma_plan.py`: dim-KM pads the CTA grid to 128 elements.
- `blockscaled_tma_kernels.py`: the boundary dim-M and dim-K packed scale
  stores do not guard the destination scale column.
- `blockscaled_tma_impl.py`: scale allocations use 64-element NVFP4 blocks.

### Make runtime ceil division overflow-safe

The host wrapper permits each logical dimension to be as large as
`INT32_MAX`, but the shared runtime helper implements ceil division as
`(num + den - 1) // den`. When `num` is a `cutlass.Int32`, the addition can
overflow before the division.

For example, `K = 2,147,483,616` is a legal multiple of 32 below `INT32_MAX`,
but `K + 127` overflows. Runtime grid and scale-layout calculations use this
operation even though the corresponding host calculations are safe Python
integer arithmetic.

Use an overflow-safe form such as `num // den + (num % den != 0)`, or widen
before adding. Audit all runtime grid, padded-extent, and scale-layout
calculations that consume `M` or `K`.

## Safety and maintainability

### Replace host-facing assertions with explicit exceptions

NVFP4 shape, outer-scale, mode, and unexpected-keyword validation still uses
Python `assert`. These checks disappear under `python -O`, potentially letting
invalid shapes reach kernel layout and address calculations. Replace public
boundary assertions with `ValueError` or `TypeError`. Compile-time assertions
inside the CuTe kernel remain appropriate.

### Use one source of truth for launch-grid calculation

`select_blockscaled_tma_plan` computes `grid_m` and `grid_k`, and the host uses
those values for CUDA grid-limit validation. `_BlockscaledTma.__call__` then
recomputes the grid independently for the actual launch. A future change can
therefore make the validated grid differ from the launched grid.

Centralize the grid formulas or pass the validated runtime grid values into
the launcher.

### Make launch policy architecture-aware

The launch planner is explicitly tuned for B200, while the public eligibility
check accepts every CUDA capability at or above 10.0. Isolate selection by
architecture so other Blackwell products and future architectures do not
silently inherit B200-specific tile and clustering choices. Architecture may
also need to participate in compilation-cache selection when generated code
is architecture-specific.

### Remove import-order dependence from cache fingerprinting

`blockscaled_tma_kernels.py` mutates QuACK's process-global
`EXTRA_SOURCE_DIRS`. QuACK memoizes its source fingerprint at the first cache
lookup, so importing this package after another cached kernel has already run
can omit these sources from the fingerprint and permit stale persistent-cache
entries. The current directory-wide fingerprint also recompiles these kernels
after unrelated handwritten-kernel changes.

Register source dependencies centrally before any cache lookup, or use a
cache mechanism that accepts per-kernel source dependencies.

### Separate implementation from benchmark recipes

`blockscaled_tma_impl.py` imports Gold reference classes and constructs
`QuantCastCuteRecipe` objects. This couples the reusable implementation to the
benchmark/reference layer. Move recipe registration to the parent
`quant_cast_cute_hand/recipes.py` module so this package contains only launch
planning, validation, compilation, and kernel implementation.

### Clean up package boundaries and naming

- Move `ScaleAlgo` out of the launch-plan module because it describes kernel
  semantics and is also imported by shared utilities.
- Standardize `mode` and `quant_orientation` in the NVFP4 and MX entry points.
- Rename `blockscale_tma_plan.py` to match the `blockscaled_tma` package name.
- Rename generic helpers such as `_nvfp4_load_philox_key` and
  `_e8m0_scale_store_as_uint`; they are now used outside their original NVFP4
  or E8M0-specific contexts.
- Add complete input and return annotations to the host wrappers and fake
  tensor helpers.

## Validation performed during review

- 43 targeted CUDA tests covering NVFP4 padding and empty tensors, grid-limit
  rejection, and compilation-cache reuse passed.
- All 10 launch-plan tests passed.
- Python bytecode compilation passed for the package.
- A guarded-allocation experiment exposed the NVFP4 dim-KM out-of-bounds scale
  writes described above.
- CUDA Compute Sanitizer could not be used in this environment because it
  failed during initialization with `cuGetProcAddress_v2` invalid-argument
  errors.
