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
- `scripts/run_benchmarks.py`: generates CSV and JSON benchmark summaries
- `scripts/plot_benchmarks.py`: turns benchmark CSV output into a chart
- `scripts/showcase_cnn.py`: runs a small 3-layer CNN forward pass with both backends
- `tests/test_algorithms.py`: correctness checks

## Quick start

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/run_benchmarks.py --repeat 5 --warmup 1
python3 scripts/plot_benchmarks.py
python3 scripts/showcase_cnn.py
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

## Notes on fidelity

This is a benchmark-and-explanation repository, not a production kernel:

- the implementation is CPU-only and uses NumPy BLAS
- the direct kernel is for correctness, not performance
- the DietConv kernel mirrors the strip-buffer structure from the poster
- the memory advantage is structural and shows up consistently; the runtime advantage depends on the problem shape and how well NumPy's BLAS path handles one large GEMM versus several smaller ones
- for stride greater than `1`, the NumPy implementation still makes a compact contiguous slice per GEMM so `@` can consume it efficiently

## Next steps

- Move the DietConv kernel into a lower-level implementation, either as a C++ microbenchmark or a PyTorch custom op, so runtime comparisons reflect the algorithm instead of mostly reflecting NumPy overhead.
- Add thread-scaling experiments to mirror the poster more closely, including outer-loop parallelism and a comparison against a strong BLAS-backed `im2col` baseline across `1`, `2`, `4`, and `8` threads.
- Expand the benchmark suite with more CNN-relevant layer shapes from AlexNet, VGG, ResNet, and MobileNet to show where strip-buffer convolution helps most and where large monolithic GEMMs still dominate.
- Separate theoretical workspace from measured process memory so the repository can report both the structural duplication reduction and the real end-to-end peak RSS seen during runs.
- Add a PyTorch inference showcase that swaps a few convolution layers between `im2col`-style lowering and DietConv-style strip buffering while preserving numerics, so the repo demonstrates the idea inside a recognizable model.
- Document the exact correspondence between the poster pseudocode and the implementation, including filter reordering, temporary-buffer layout, and how stride affects the copied strip.
