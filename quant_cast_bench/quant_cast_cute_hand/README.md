# Handwritten CuTe DSL quantization kernels

This directory contains optimized handwritten CuTe DSL quantization kernels. The
`mxfp8_v2.py` implementation is an optimized prototype rather than an
upstream-ready PyTorch operator. The notes below capture the remaining work and
review findings for upstreaming it into PyTorch core.

## `mxfp8_v2.py` upstream-readiness review

The kernel supports dim-K, dim-M, and dim-KM quantization, RTNE and stochastic
rounding, BF16/FP16/FP32 inputs, padded scale outputs, and mixed-width indexing.
Its synchronization sequence appears coherent for currently supported shapes:

1. Initialize the input TMA barrier.
2. Issue and await the input TMA transfer.
3. Complete the dim-M input reads and output writes, when enabled.
4. Synchronize before dim-K overwrites the aliased input shared-memory buffer.
5. Fence and synchronize the qdata shared-memory writes.
6. Issue the enabled qdata TMA stores.
7. Overlap the dim-K global scale stores with the qdata TMA store.

The following remaining issues should be addressed before upstreaming.

### Must fix

#### Define the stochastic-rounding RNG contract

The current API accepts an explicit two-element Philox key instead of using
PyTorch's generator and Philox-offset machinery. An upstream operator must
define:

- generator advancement and counter consumption per call;
- deterministic and reproducibility behavior;
- CUDA graph capture behavior;
- distributed-execution behavior;
- overflow/wraparound behavior for the counter.

### PyTorch operator integration

Production kernel code should be separated from the benchmark harness. The
current module imports gold recipes, `QuantCastCuteRecipe`, and a private Philox
helper, and it registers benchmark recipe objects at the bottom of the file.
Split it into:

1. Stable low-level kernel helpers and the CuTe kernel.
2. A pure, testable launch-policy selector.
3. A dispatcher-facing allocation, validation, and launch wrapper.
4. Gold references and benchmark recipe registration outside production code.

An upstream operator also needs:

- an operator schema;
- CUDA-only dispatch and an explicit supported-hardware check;
- FakeTensor/meta support;
- defined `torch.compile` behavior;
- an explicit autograd policy;
- current-stream and CUDA-graph integration tests.

A non-default-stream dependency test passed during review, but it should become
a permanent test rather than an implicit assumption.

### API and data-contract cleanup

- A contiguous view is not necessarily 16-byte aligned. The TVM-FFI callable is
  compiled from a fake input tensor with `assumed_align=16`, but the public
  wrapper does not validate that contract for offset contiguous views. Explicitly
  document and validate the requirement, or provide a fallback/copy path.
- Validate that the selected CUDA device supports all required TMA,
  scale-conversion, and shared-memory features before compiling. Unsupported
  GPUs currently fail inside CuTe rather than at the public boundary.
- Decide whether zero-sized tensors should return correctly shaped empty outputs.
  PyTorch operators generally support an empty fast path when meaningful.
- The return arity changes with `quant_orientation`: dim-K and dim-M return two
  tensors, while dim-KM returns four. Decide whether separate schemas or a fixed
  structured result would provide a cleaner dispatcher and tracing contract.
- Add public return annotations and docstrings that define qdata orientation,
  scale shape/swizzle, scale padding, dtype support, alignment requirements, and
  stochastic semantics.
- `functools.partial` recipe wrappers allow a caller to override their supposedly
  fixed `quant_orientation`. Fixed-orientation public wrappers should not expose
  that override.
- The launch heuristics and comments are tuned specifically for B200. Isolate the
  launch policy so other Blackwell products and future architectures can select
  their own policy without changing the semantic kernel code.

### Kernel documentation and maintenance

The input TMA and output TMA copies use an unusual participation contract: one
selected warp invokes `cute.copy`, while only one elected lane performs the
input barrier arrival. Document why this generates exactly one transfer, which
operations are warp-collective versus lane-elected, and which CuTe/CUDA contract
guarantees it.

The kernel docstring and dim-K comments say that qdata occupies the first half of
the aliased input buffer. That is true for 16-bit inputs, but FP32 qdata occupies
only the first quarter. Describe it as the initial `tile_m_size * tile_k_size`
bytes instead.

Other cleanup expected for upstream code:

- add precise return types to the implementation and recipe-facing wrappers;
- use consistent PyTorch/CuTe naming conventions;
- replace generic `"unsupported"` errors with actionable diagnostics;
- explain the physical E8M0 scale layout and its intended GEMM consumers;
- document why each shape divisibility restriction is required;
- preserve the existing local-variable layout names where they help connect code
  to CuTe layouts, but avoid benchmark-specific naming in the public API.

### Required test coverage

Before upstreaming, add coverage for:

- full tiles and boundary tiles for every supported dtype and orientation;
- repeated use and compilation-cache isolation across heterogeneous GPUs;
- non-default streams and CUDA graph capture;
- aligned and offset/misaligned contiguous views;
- concurrent first invocation and compilation;
- shapes at the CUDA grid limits and grid-X overflow (grid-Y overflow is covered);
- mixed-width address and stochastic-counter calculations in every orientation
  and rounding mode;
- empty dimensions and all minimum legal shapes;
- NaN, infinities, signed zero, subnormals, saturation boundaries, and all-zero
  groups;

The existing large mixed-width test allocates a multi-gigabyte tensor and covers
only stochastic dim-K indexing. For routine upstream CI, prefer a smaller
synthetic test of the address/counter calculation and reserve the full allocation
test for large-GPU or periodic testing.
