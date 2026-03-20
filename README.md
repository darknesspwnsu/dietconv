# DietConv Showcase

This repository recreates the central idea from the 2016 Carnegie Mellon poster: keep the GEMM-friendly structure of fast convolution, but avoid materializing the full `im2col` workspace used by Caffe-style convolution.

## Poster gist

The poster describes a middle ground between:

- textbook convolution, which has minimal extra memory but weak matrix-multiply reuse
- Caffe's `im2col + GEMM`, which is fast because it turns convolution into one large matrix multiply, but duplicates every overlapping patch

DietConv keeps the GEMM reduction, but only materializes a narrow temporary strip for one output row at a time. For each output row:

1. Copy an `F`-row strip of the input image into a temporary buffer.
2. Reorder filters by kernel-column so each kernel slice can be multiplied against that strip.
3. Accumulate `F` smaller GEMMs instead of one giant GEMM over the full `im2col` matrix.

That changes the explicit workspace from:

- `im2col`: `C * Fh * Fw * Hout * Wout`
- DietConv strip buffer: `C * Fh * Win`

For an AlexNet-style first layer (`3 x 227 x 227`, `96` filters, `11 x 11`, stride `4`), this is:

- `im2col`: `1,098,075` floats, about `4.19 MiB`
- DietConv: `7,491` floats, about `0.03 MiB`
- workspace reduction: about `146.6x`

That lines up with the poster's claim that the biggest win is memory and data duplication, not a fundamentally different arithmetic count.

## What's in the repo

- `dietconv/algorithms.py`: direct, `im2col`, and DietConv-style convolution kernels in NumPy
- `dietconv/torch_ops.py`: PyTorch `unfold` and DietConv v2 operators for CPU benchmarks
- `scripts/run_benchmarks.py`: generates CSV and JSON benchmark summaries
- `scripts/plot_benchmarks.py`: turns benchmark CSV output into a chart
- `scripts/showcase_cnn.py`: runs a small 3-layer CNN forward pass with both backends
- `cpp/dietconv_benchmark.cpp`: lower-level C++ benchmark driver using Accelerate `sgemm`
- `scripts/run_cpp_benchmarks.py`: builds and runs size-scaling and thread-scaling C++ sweeps
- `scripts/plot_cpp_benchmarks.py`: generates plots for the C++ sweeps
- `scripts/run_torch_benchmarks.py`: runs CPU PyTorch benchmarks against native `conv2d`, explicit `unfold`, and DietConv v2
- `scripts/plot_torch_benchmarks.py`: generates plots for the PyTorch sweeps
- `tests/test_algorithms.py`: correctness checks
- `tests/test_torch_ops.py`: PyTorch correctness checks

## Quick start

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/run_benchmarks.py --repeat 5 --warmup 1
python3 scripts/plot_benchmarks.py
python3 scripts/showcase_cnn.py
python3 scripts/run_cpp_benchmarks.py --repeat 3 --warmup 1
python3 scripts/plot_cpp_benchmarks.py
python3 scripts/run_torch_benchmarks.py --repeat 2 --warmup 1
python3 scripts/plot_torch_benchmarks.py
```

Results are written to `results/`.

## Current results

These files were generated in this checkout with `python3 scripts/run_benchmarks.py --repeat 3 --warmup 1` and `python3 scripts/showcase_cnn.py`.

| Problem | im2col ms | DietConv ms | im2col MiB | DietConv MiB | Workspace reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| alexnet-conv1 | 80.76 | 15.59 | 4.19 | 0.03 | 146.6x |
| stem-64 | 13.55 | 1.74 | 0.57 | 0.01 | 102.4x |
| feature-32 | 23.40 | 44.59 | 3.45 | 0.02 | 162.2x |
| feature-64 | 2.41 | 10.39 | 1.72 | 0.02 | 78.4x |

For the lightweight 3-layer CNN showcase:

- `im2col` peak workspace: `1.125 MiB`
- `DietConv` peak workspace: `0.012 MiB`
- workspace reduction: `90.35x`
- final output checksum difference: `0.0`

## DietConv v2

DietConv v1 is the direct strip-buffer interpretation of the poster: copy a full `Fh x Win` input strip for each output row, then run one GEMM per kernel column.

DietConv v2 adds one more idea:

- tile the output width and only pack the input span needed for that tile
- reuse the packed tile window across adjacent output rows in the lower-level C++ implementation
- feed GEMM directly from the tiled window on stride-`1` cases instead of repacking a second strip matrix
- autotune the output tile width during C++ and PyTorch benchmark sweeps, then choose the smallest workspace within `5%` of the fastest timing
- keep a smaller per-thread lowering buffer so memory grows more slowly with thread count

Tradeoffs:

- v2 usually lowers workspace further than v1
- v2 is now clearly faster than v1 on most of the stride-`1` size sweep in the C++ benchmark, but it can still lose on very wide cases or under some thread counts
- v2 can still be slower when tiling causes too much repeated packing across the width dimension or when the chosen tile width is not ideal for a specific thread count

## C++ benchmark results

The NumPy version is useful for explaining the algorithm, but the newer C++ path is the more meaningful performance view because it removes most Python overhead and uses Accelerate `sgemm` for the matrix multiplies.

Key observations from the generated C++ sweeps:

- Size scaling with `C=32`, `K=64`, `3x3`, stride `1`, padding `1`: v2 now beats v1 on `32`, `48`, `64`, `96`, and `128`, and only loses at `160`.
- At `32x32`, v1 is `0.369 ms / 0.024 MiB` and v2 is `0.181 ms / 0.012 MiB`.
- At `128x128`, `im2col` is `39.54 ms / 18.00 MiB`, v1 is `2.58 ms / 0.094 MiB`, and v2 is `2.41 ms / 0.048 MiB`.
- At `160x160`, v1 is still faster than v2 (`3.19 ms` vs `3.59 ms`), but v2 keeps workspace at about `0.048 MiB` while v1 grows to about `0.118 MiB`.
- On `alexnet-conv1`, v2 is modestly better than v1 at `1`, `2`, and `8` threads, and substantially better in workspace at `1`, `4`, and `8` threads.
- On the `128x128` thread sweep, v2 is better than v1 at `1` and `4` threads while using about half the workspace in both cases.

Generated C++ artifacts:

- `results/cpp_size_scaling.csv`
- `results/cpp_thread_scaling.csv`
- `results/cpp_size_scaling.png`
- `results/cpp_thread_runtime.png`
- `results/cpp_thread_workspace.png`

## PyTorch op benchmark results

The PyTorch path is intentionally a framework integration story, not the fastest implementation in the repo. The op in `dietconv/torch_ops.py` is a CPU tensor implementation of DietConv v2 that can be benchmarked against:

- native `torch.nn.functional.conv2d`
- explicit `torch.nn.functional.unfold` lowering plus matrix multiply
- DietConv v2 on torch tensors

Key observations from the generated PyTorch sweeps:

- Native `conv2d` remains the fastest option by a wide margin. That is expected because the DietConv op is written in Python over torch tensors rather than as a compiled extension.
- Compared with explicit `unfold`, the DietConv v2 torch op uses drastically less lowering memory on every case.
- On the size sweep, DietConv v2 is faster than `torch-unfold` at `48x48`, `64x64`, and `96x96`, while using much less workspace.
- For `96x96`, `torch-unfold` is `5.82 ms / 10.125 MiB` and DietConv v2 is `4.54 ms / 0.036 MiB`.
- On `alexnet-conv1`, the Python torch op is not competitive yet, which is a good argument for a future compiled PyTorch extension rather than more Python-level tuning.

Generated PyTorch artifacts:

- `results/torch_size_scaling.csv`
- `results/torch_thread_scaling.csv`
- `results/torch_size_scaling.png`
- `results/torch_thread_runtime.png`

## Notes on fidelity

This is a benchmark-and-explanation repository, not a production kernel:

- the implementation is CPU-only and uses NumPy BLAS
- the lower-level benchmark path is C++ plus Apple's Accelerate BLAS on macOS
- the PyTorch path is a CPU-only framework integration demo, not a compiled extension
- the direct kernel is for correctness, not performance
- the DietConv kernel mirrors the strip-buffer structure from the poster
- DietConv v2 is a practical enhancement, not something claimed in the original poster
- the memory advantage is structural and shows up consistently; the runtime advantage depends on the problem shape and how well NumPy's BLAS path handles one large GEMM versus several smaller ones
- for stride greater than `1`, the NumPy implementation still makes a compact contiguous slice per GEMM so `@` can consume it efficiently

## Next steps

- Turn the torch DietConv op into a compiled PyTorch extension so the framework-level benchmark story is not dominated by Python loop overhead.
- Improve v2 autotuning so tile width adapts to thread count and problem shape more reliably; the `128x128` and `160x160` results show there is still room to choose better shapes.
- Add thread-scaling experiments that separate arithmetic time from packing time, so it is clearer when v2 wins because of cache reuse versus because of reduced memory traffic.
- Expand the benchmark suite with more CNN-relevant layer shapes from AlexNet, VGG, ResNet, and MobileNet to show where strip-buffer convolution helps most and where large monolithic GEMMs still dominate.
- Separate theoretical workspace from measured process memory so the repository can report both the structural duplication reduction and the real end-to-end peak RSS seen during runs.
- Add a PyTorch inference showcase that swaps a few convolution layers between `unfold`-style lowering and DietConv-style strip buffering while preserving numerics, so the repo demonstrates the idea inside a recognizable model.
- Document the exact correspondence between the poster pseudocode and the implementation, including filter reordering, temporary-buffer layout, and how stride affects the copied strip.
