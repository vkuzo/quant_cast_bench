# Vendored third-party skills

The `kernel-triton-writing/` and `kernel-cute-writing/` skill directories in this
folder are **vendored verbatim** from NVIDIA's TensorRT-LLM repository:

- Upstream: https://github.com/NVIDIA/TensorRT-LLM (`.claude/skills/`)
- Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- License: Apache License 2.0 (full text in `LICENSE.apache-2.0`, alongside this file)

These files are unmodified copies. Each `SKILL.md` declares `license: Apache-2.0`
in its frontmatter and the bundled `scripts/*.py` carry SPDX Apache-2.0 headers.

The rest of the `quant_cast_bench` repository is licensed under the MIT License
(see the repository-root `LICENSE`). Apache-2.0 is permissive and compatible with
MIT; the vendored files remain under Apache-2.0 as required by that license.

If you modify a vendored skill, note the change here (Apache-2.0 §4 requires
stating significant modifications to redistributed files).
