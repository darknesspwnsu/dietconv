from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dietconv.algorithms import (
    Conv2DProblem,
    conv2d_dietconv,
    conv2d_im2col,
    workspace_bytes_dietconv,
    workspace_bytes_im2col,
)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=x.dtype)


def run_network(backend: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    layers = [
        Conv2DProblem("conv1", 3, 64, 64, 16, 7, 7, stride=2, padding=3),
        Conv2DProblem("conv2", 16, 32, 32, 32, 3, 3, stride=1, padding=1),
        Conv2DProblem("conv3", 32, 32, 32, 32, 3, 3, stride=1, padding=1),
    ]
    kernels = [rng.standard_normal(layer.weight_shape, dtype=np.float32) for layer in layers]
    activations = rng.standard_normal(layers[0].input_shape, dtype=np.float32)

    total_ms = 0.0
    peak_workspace = 0
    for layer, weight in zip(layers, kernels):
        fn = conv2d_im2col if backend == "im2col" else conv2d_dietconv
        workspace = (
            workspace_bytes_im2col(activations, weight, stride=layer.stride, padding=layer.padding)
            if backend == "im2col"
            else workspace_bytes_dietconv(activations, weight, padding=layer.padding)
        )
        peak_workspace = max(peak_workspace, workspace)
        start = time.perf_counter()
        activations = fn(activations, weight, stride=layer.stride, padding=layer.padding)
        total_ms += (time.perf_counter() - start) * 1000.0
        activations = relu(activations)

    return {
        "backend": backend,
        "total_ms": total_ms,
        "peak_workspace_mib": peak_workspace / (1024.0 * 1024.0),
        "final_output_shape": list(activations.shape),
        "final_output_checksum": float(np.sum(activations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight CNN showcase.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cnn_showcase.json"),
        help="Path to the JSON summary.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    args = parser.parse_args()

    im2col_result = run_network("im2col", seed=args.seed)
    dietconv_result = run_network("dietconv", seed=args.seed)
    summary = {
        "im2col": im2col_result,
        "dietconv": dietconv_result,
        "workspace_ratio_im2col_over_dietconv": (
            im2col_result["peak_workspace_mib"] / dietconv_result["peak_workspace_mib"]
        ),
        "checksum_abs_diff": abs(
            im2col_result["final_output_checksum"] - dietconv_result["final_output_checksum"]
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "update_readme_benchmarks.py")], check=True, cwd=ROOT)
    print(f"Wrote showcase summary to {args.output}")


if __name__ == "__main__":
    main()
