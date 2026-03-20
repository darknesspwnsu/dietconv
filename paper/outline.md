# DietConv Paper Outline

## Working title

- DietConv Revisited: Memory-Efficient Strip-Buffered GEMM Convolution with a Tiled v2 Schedule

## Core claim

- DietConv is a viable memory-efficient alternative to full `im2col` lowering.
- DietConv-v2 preserves the original strip-buffer idea while improving practical behavior with width tiling, row reuse, and autotuning.
- The strongest result is memory efficiency; runtime competitiveness is conditional on shape and thread count.

## Intended contribution framing

- Original contribution: describe the 2016 DietConv schedule clearly and reproducibly.
- New contribution: introduce and evaluate DietConv-v2 as a practical refinement.
- Artifact contribution: provide an end-to-end benchmark suite spanning NumPy, C++, and compiled PyTorch.

## Figures

- `paper/figures/cpp_size_scaling.pdf`
- `paper/figures/dietconv_overview.pdf`
- `paper/figures/torch_size_scaling.pdf`
- `paper/figures/torch_memory_scaling.pdf`
- `paper/figures/v2_tile_ablation_cpp.pdf`

## Tables

- `paper/tables/cpp_runtime_summary.tex`
- `paper/tables/torch_runtime_summary.tex`
- `paper/tables/torch_memory_summary.tex`

## Section structure

### 1. Introduction

- Memory duplication from `im2col` is a longstanding cost of GEMM-based convolution.
- DietConv revisits the same decomposition with a strip buffer instead of a full lowered matrix.
- Main message: memory wins are structural; performance can be competitive when implemented carefully.

### 2. Background and motivation

- Direct convolution vs explicit lowering.
- Why `im2col` is fast but memory-hungry.
- Why the original poster idea is interesting to revisit now.

### 3. DietConv v1

- Output-row strip buffer.
- Weight reordering by kernel column.
- One GEMM per kernel column.
- Explicit workspace analysis.

### 4. DietConv v2

- Output-width tiling.
- Row-window reuse across adjacent output rows.
- Direct temp-to-GEMM path for stride-1.
- Autotuned tile width.
- Expected tradeoffs and failure modes.

### 5. Implementation

- NumPy reference.
- C++ kernel with Accelerate GEMM.
- Compiled PyTorch extension.
- Weight prepacking, optimization gating, and RSS measurement methodology.

### 6. Experimental setup

- CPU-only macOS arm64 environment.
- Benchmark families: size sweep, thread sweep, memory sweep, ablation sweep.
- Metrics: runtime, explicit workspace, isolated-process RSS delta, numerical error.

### 7. Results

- C++ results: v2 often beats v1 and drastically beats `im2col` on workspace.
- PyTorch runtime results: v2 is competitive with unfold and sometimes native.
- PyTorch memory results: v1/v2 often beat native RSS and strongly beat unfold.
- Ablation: tile width materially changes outcome; autotuning is justified.

### 8. Limitations

- CPU-only.
- Single-batch inference focus.
- Some regimes still regress.
- RSS is practical but not allocator-exact.

### 9. Conclusion

- DietConv is a publishable concept when presented as a memory-efficient lowering family.
- v2 strengthens the practical case without changing the core identity of the algorithm.

## Open items before submission-quality draft

- Expand the ablation story if more tuning knobs are exposed.
- Add repeated runs and confidence intervals for all reported tables.
- Consider broader hardware coverage if submission target expects it.
