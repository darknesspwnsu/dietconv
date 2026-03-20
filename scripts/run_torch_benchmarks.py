from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dietconv.torch_ops import (
    dietconv2d_v1_compiled,
    dietconv2d_v1_compiled_prepacked,
    dietconv2d_v2_compiled,
    dietconv2d_v2_compiled_prepacked,
    load_dietconv_extension,
    prepack_dietconv_weight,
    workspace_bytes_dietconv2d_v2,
    workspace_bytes_dietconv2d_v1,
    workspace_bytes_unfold,
    unfold_conv2d,
)


SIZE_SWEEP = [32, 48, 64, 96]
THREAD_SWEEP = [1, 2, 4, 8]
THREAD_PROBLEMS = [
    {
        "problem_name": "alexnet-conv1",
        "c": 3,
        "h": 227,
        "w": 227,
        "k": 96,
        "fh": 11,
        "fw": 11,
        "stride": 4,
        "pad": 0,
    },
    {
        "problem_name": "scale-96",
        "c": 32,
        "h": 96,
        "w": 96,
        "k": 64,
        "fh": 3,
        "fw": 3,
        "stride": 1,
        "pad": 1,
    },
]
NUMERIC_TOLERANCE = 1e-4


def make_inputs(problem: dict, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(
        (1, problem["c"], problem["h"], problem["w"]),
        generator=generator,
        dtype=torch.float32,
    )
    weight = torch.randn(
        (problem["k"], problem["c"], problem["fh"], problem["fw"]),
        generator=generator,
        dtype=torch.float32,
    )
    return x, weight


def time_call(fn, repeat: int, warmup: int) -> tuple[torch.Tensor, float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        timings = []
        result = None
        for _ in range(repeat):
            start = time.perf_counter()
            result = fn()
            timings.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    mean = sum(timings) / len(timings)
    variance = sum((timing - mean) ** 2 for timing in timings) / len(timings)
    return result, mean, variance ** 0.5


def candidate_tile_widths(out_w: int, stride: int) -> list[int]:
    base = [16, 24, 32, 48, 64, 96, out_w]
    if stride > 1:
        base = [8, 16, 24, 32, out_w]
    return sorted({min(width, out_w) for width in base})


def autotune_tile_width(x: torch.Tensor, packed_weight: torch.Tensor, weight: torch.Tensor, problem: dict) -> int:
    out_w = (problem["w"] + 2 * problem["pad"] - problem["fw"]) // problem["stride"] + 1
    trials = []
    for width in candidate_tile_widths(out_w, problem["stride"]):
        _, mean, _ = time_call(
            lambda: dietconv2d_v2_compiled_prepacked(
                x,
                packed_weight,
                stride=problem["stride"],
                padding=problem["pad"],
                tile_out_width=width,
            ),
            repeat=1,
            warmup=1,
        )
        workspace_bytes = workspace_bytes_dietconv2d_v2(
            x,
            weight,
            stride=problem["stride"],
            padding=problem["pad"],
            tile_out_width=width,
        )
        trials.append((width, mean, workspace_bytes))
    fastest = min(mean for _, mean, _ in trials)
    viable = [trial for trial in trials if trial[1] <= fastest * 1.05]
    return min(viable, key=lambda trial: (trial[2], trial[0]))[0]


def run_problem(problem: dict, backend: str, threads: int, seed: int, repeat: int, warmup: int) -> dict:
    torch.set_num_threads(threads)
    x, weight = make_inputs(problem, seed)
    reference = F.conv2d(x, weight, stride=problem["stride"], padding=problem["pad"])
    packed_weight = None

    tile_out_width = 0
    if backend == "torch-native":
        fn = lambda: F.conv2d(x, weight, stride=problem["stride"], padding=problem["pad"])
        workspace_bytes = 0
    elif backend == "torch-unfold":
        fn = lambda: unfold_conv2d(x, weight, stride=problem["stride"], padding=problem["pad"])
        workspace_bytes = workspace_bytes_unfold(x, weight, stride=problem["stride"], padding=problem["pad"])
    elif backend == "dietconv-v1-compiled":
        packed_weight = prepack_dietconv_weight(weight)
        fn = lambda: dietconv2d_v1_compiled_prepacked(x, packed_weight, stride=problem["stride"], padding=problem["pad"])
        workspace_bytes = workspace_bytes_dietconv2d_v1(x, weight, padding=problem["pad"])
    elif backend == "dietconv-v2-compiled":
        packed_weight = prepack_dietconv_weight(weight)
        tile_out_width = autotune_tile_width(x, packed_weight, weight, problem)
        fn = lambda: dietconv2d_v2_compiled_prepacked(
            x,
            packed_weight,
            stride=problem["stride"],
            padding=problem["pad"],
            tile_out_width=tile_out_width,
        )
        workspace_bytes = workspace_bytes_dietconv2d_v2(
            x,
            weight,
            stride=problem["stride"],
            padding=problem["pad"],
            tile_out_width=tile_out_width,
        )
    else:
        raise ValueError(f"Unknown backend {backend}")

    output, mean_ms, std_ms = time_call(fn, repeat=repeat, warmup=warmup)
    max_abs_diff = float((output - reference).abs().max().item())
    if max_abs_diff > NUMERIC_TOLERANCE:
        raise AssertionError(
            f"{backend} exceeded numeric tolerance: {max_abs_diff} > {NUMERIC_TOLERANCE}"
        )
    return {
        "backend": backend,
        "threads": threads,
        "tile_out_width": tile_out_width,
        "c": problem["c"],
        "h": problem["h"],
        "w": problem["w"],
        "k": problem["k"],
        "fh": problem["fh"],
        "fw": problem["fw"],
        "stride": problem["stride"],
        "pad": problem["pad"],
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "workspace_bytes": workspace_bytes,
        "workspace_mib": workspace_bytes / (1024.0 * 1024.0),
        "max_abs_diff_vs_native": max_abs_diff,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyTorch DietConv benchmark sweeps.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    load_dietconv_extension()

    size_rows = []
    for index, size in enumerate(SIZE_SWEEP):
        problem = {
            "c": 32,
            "h": size,
            "w": size,
            "k": 64,
            "fh": 3,
            "fw": 3,
            "stride": 1,
            "pad": 1,
        }
        for backend in ["torch-native", "torch-unfold", "dietconv-v1-compiled", "dietconv-v2-compiled"]:
            row = run_problem(problem, backend, threads=1, seed=args.seed + index, repeat=args.repeat, warmup=args.warmup)
            row["problem_name"] = f"scale-{size}"
            row["sweep"] = "size"
            size_rows.append(row)

    thread_rows = []
    for problem_index, problem in enumerate(THREAD_PROBLEMS):
        for threads in THREAD_SWEEP:
            for backend in ["torch-native", "torch-unfold", "dietconv-v1-compiled", "dietconv-v2-compiled"]:
                row = run_problem(
                    problem,
                    backend,
                    threads=threads,
                    seed=args.seed + 100 + problem_index,
                    repeat=args.repeat,
                    warmup=args.warmup,
                )
                row["problem_name"] = problem["problem_name"]
                row["sweep"] = "threads"
                thread_rows.append(row)

    write_csv(args.results_dir / "torch_size_scaling.csv", size_rows)
    write_csv(args.results_dir / "torch_thread_scaling.csv", thread_rows)
    with (args.results_dir / "torch_benchmark_summary.json").open("w") as handle:
        json.dump(
            {
                "size_scaling_cases": len(size_rows),
                "thread_scaling_cases": len(thread_rows),
                "torch_version": torch.__version__,
                "compiled_extension": True,
                "numeric_tolerance": NUMERIC_TOLERANCE,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    subprocess.run([sys.executable, str(ROOT / "scripts" / "update_readme_benchmarks.py")], check=True, cwd=ROOT)
    print(f"Wrote torch benchmark results to {args.results_dir}")


if __name__ == "__main__":
    main()
