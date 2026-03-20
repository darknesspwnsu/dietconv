from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
CPP_BINARY = BUILD_DIR / "dietconv_benchmark"
CPP_SOURCE = ROOT / "cpp" / "dietconv_benchmark.cpp"


SIZE_SWEEP = [32, 48, 64, 96, 128, 160]
THREAD_SWEEP = [1, 2, 4, 8]
BACKENDS = ["im2col", "dietconv-v1", "dietconv-v2"]
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
        "problem_name": "scale-128",
        "c": 32,
        "h": 128,
        "w": 128,
        "k": 64,
        "fh": 3,
        "fw": 3,
        "stride": 1,
        "pad": 1,
    },
]


def build_binary() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        "clang++",
        "-O3",
        "-std=c++17",
        "-Wno-deprecated-declarations",
        str(CPP_SOURCE),
        "-framework",
        "Accelerate",
        "-o",
        str(CPP_BINARY),
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def run_case(parameters: dict[str, str | int]) -> dict:
    env = os.environ.copy()
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    command = [str(CPP_BINARY)]
    for key, value in parameters.items():
        command.extend([f"--{key}", str(value)])
    completed = subprocess.run(
        command,
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No benchmark rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_tile_widths(out_w: int, stride: int) -> list[int]:
    candidates = [16, 24, 32, 48, 64, 96, 128, out_w]
    if stride > 1:
        candidates = [8, 16, 24, 32, 48, out_w]
    filtered = sorted({min(width, out_w) for width in candidates if width > 0})
    return filtered


def autotune_v2_case(base_parameters: dict[str, str | int]) -> int:
    out_w = ((int(base_parameters["w"]) + 2 * int(base_parameters["pad"]) - int(base_parameters["fw"])) // int(base_parameters["stride"])) + 1
    candidates = candidate_tile_widths(out_w, int(base_parameters["stride"]))
    trials = []
    for width in candidates:
        row = run_case(
            {
                **base_parameters,
                "backend": "dietconv-v2",
                "tile-out-width": width,
                "repeat": 1,
                "warmup": 1,
            }
        )
        trials.append(row)
    fastest_ms = min(float(row["mean_ms"]) for row in trials)
    viable = [
        row for row in trials
        if float(row["mean_ms"]) <= fastest_ms * 1.05
    ]
    chosen = min(viable, key=lambda row: (float(row["workspace_bytes"]), int(row["tile_out_width"])))
    return int(chosen["tile_out_width"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run C++ DietConv benchmarks.")
    parser.add_argument("--repeat", type=int, default=3, help="Measured iterations per case.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per case.")
    parser.add_argument("--seed", type=int, default=17, help="Base RNG seed.")
    parser.add_argument(
        "--tile-out-width",
        type=int,
        default=0,
        help="Output tile width for the DietConv v2 benchmark.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for CSV and JSON artifacts.",
    )
    args = parser.parse_args()

    build_binary()

    size_rows = []
    for index, size in enumerate(SIZE_SWEEP):
        for backend in BACKENDS:
            parameters = {
                "backend": backend,
                "c": 32,
                "h": size,
                "w": size,
                "k": 64,
                "fh": 3,
                "fw": 3,
                "stride": 1,
                "pad": 1,
                "threads": 1,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "seed": args.seed + index,
                "tile-out-width": args.tile_out_width,
            }
            if backend == "dietconv-v2" and args.tile_out_width == 0:
                parameters["tile-out-width"] = autotune_v2_case(parameters)
            row = run_case(parameters)
            row["sweep"] = "size"
            row["problem_name"] = f"scale-{size}"
            size_rows.append(row)

    thread_rows = []
    for problem_index, problem in enumerate(THREAD_PROBLEMS):
        for threads in THREAD_SWEEP:
            for backend in BACKENDS:
                parameters = {
                    "backend": backend,
                    "threads": threads,
                    "repeat": args.repeat,
                    "warmup": args.warmup,
                    "seed": args.seed + 100 + problem_index,
                    "tile-out-width": args.tile_out_width,
                    **{key: value for key, value in problem.items() if key != "problem_name"},
                }
                if backend == "dietconv-v2" and args.tile_out_width == 0:
                    parameters["tile-out-width"] = autotune_v2_case(parameters)
                row = run_case(parameters)
                row["sweep"] = "threads"
                row["problem_name"] = problem["problem_name"]
                thread_rows.append(row)

    write_csv(args.results_dir / "cpp_size_scaling.csv", size_rows)
    write_csv(args.results_dir / "cpp_thread_scaling.csv", thread_rows)

    summary = {
        "size_scaling_cases": len(size_rows),
        "thread_scaling_cases": len(thread_rows),
        "tile_out_width": args.tile_out_width,
    }
    with (args.results_dir / "cpp_benchmark_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "update_readme_benchmarks.py")], check=True, cwd=ROOT)

    print(f"Wrote C++ benchmark results to {args.results_dir}")


if __name__ == "__main__":
    main()
