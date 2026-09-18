# Block-scaled TMA review notes

The implementation supports MXFP8, MXFP4, and NVFP4 quantization across
dim-K, dim-M, and dim-KM orientations. Before upstreaming, address the issues
below in priority order.

## Safety and maintainability

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

## Validation performed during review

- 43 targeted CUDA tests covering NVFP4 padding and empty tensors, grid-limit
  rejection, and compilation-cache reuse passed.
- All 10 launch-plan tests passed.
- Python bytecode compilation passed for the package.
- CUDA Compute Sanitizer could not be used in this environment because it
  failed during initialization with `cuGetProcAddress_v2` invalid-argument
  errors.
