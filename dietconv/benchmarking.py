from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np

from .algorithms import (
    Conv2DProblem,
    conv2d_dietconv,
    conv2d_im2col,
    workspace_bytes_dietconv,
    workspace_bytes_im2col,
)


DEFAULT_PROBLEMS = [
    Conv2DProblem(
        name="alexnet-conv1",
        input_channels=3,
        input_height=227,
        input_width=227,
        output_channels=96,
        kernel_height=11,
        kernel_width=11,
        stride=4,
        padding=0,
    ),
    Conv2DProblem(
        name="stem-64",
        input_channels=3,
        input_height=64,
        input_width=64,
        output_channels=64,
        kernel_height=7,
        kernel_width=7,
        stride=2,
        padding=3,
    ),
    Conv2DProblem(
        name="feature-32",
        input_channels=32,
        input_height=56,
        input_width=56,
        output_channels=64,
        kernel_height=3,
        kernel_width=3,
        stride=1,
        padding=1,
    ),
    Conv2DProblem(
        name="feature-64",
        input_channels=64,
        input_height=28,
        input_width=28,
        output_channels=128,
        kernel_height=3,
        kernel_width=3,
        stride=1,
        padding=1,
    ),
]


@dataclass
class BenchmarkRow:
    problem_name: str
    backend: str
    mean_ms: float
    std_ms: float
    workspace_bytes: int
    workspace_mib: float
    output_channels: int
    output_height: int
    output_width: int
    max_abs_diff_vs_im2col: float


def _rng_inputs(problem: Conv2DProblem, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(problem.input_shape, dtype=np.float32)
    weight = rng.standard_normal(problem.weight_shape, dtype=np.float32)
    return x, weight


def _time_call(fn: Callable[[], np.ndarray], repeat: int, warmup: int) -> tuple[np.ndarray, List[float]]:
    for _ in range(warmup):
        fn()
    timings = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    return result, timings


def benchmark_problem(
    problem: Conv2DProblem,
    repeat: int = 5,
    warmup: int = 1,
    seed: int = 0,
) -> List[BenchmarkRow]:
    x, weight = _rng_inputs(problem, seed=seed)
    out_channels, out_height, out_width = problem.output_shape

    baseline, _ = _time_call(
        lambda: conv2d_im2col(
            x,
            weight,
            stride=problem.stride,
            padding=problem.padding,
        ),
        repeat=1,
        warmup=0,
    )

    rows: List[BenchmarkRow] = []
    backends: Dict[str, Callable[[], np.ndarray]] = {
        "im2col": lambda: conv2d_im2col(
            x,
            weight,
            stride=problem.stride,
            padding=problem.padding,
        ),
        "dietconv": lambda: conv2d_dietconv(
            x,
            weight,
            stride=problem.stride,
            padding=problem.padding,
        ),
    }
    workspace_map = {
        "im2col": workspace_bytes_im2col(x, weight, stride=problem.stride, padding=problem.padding),
        "dietconv": workspace_bytes_dietconv(x, weight, padding=problem.padding),
    }

    for backend_name, fn in backends.items():
        output, timings = _time_call(fn, repeat=repeat, warmup=warmup)
        rows.append(
            BenchmarkRow(
                problem_name=problem.name,
                backend=backend_name,
                mean_ms=float(np.mean(timings)),
                std_ms=float(np.std(timings)),
                workspace_bytes=workspace_map[backend_name],
                workspace_mib=workspace_map[backend_name] / (1024.0 * 1024.0),
                output_channels=out_channels,
                output_height=out_height,
                output_width=out_width,
                max_abs_diff_vs_im2col=float(np.max(np.abs(output - baseline))),
            )
        )

    return rows


def benchmark_suite(
    problems: Iterable[Conv2DProblem] = DEFAULT_PROBLEMS,
    repeat: int = 5,
    warmup: int = 1,
    seed: int = 0,
) -> List[BenchmarkRow]:
    rows: List[BenchmarkRow] = []
    for index, problem in enumerate(problems):
        rows.extend(
            benchmark_problem(
                problem,
                repeat=repeat,
                warmup=warmup,
                seed=seed + index,
            )
        )
    return rows


def save_rows_csv(rows: Iterable[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("No benchmark rows were provided.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_summary_json(rows: Iterable[BenchmarkRow], path: Path) -> None:
    rows = list(rows)
    grouped: Dict[str, Dict[str, float]] = {}
    for row in rows:
        grouped.setdefault(row.problem_name, {})
        key_prefix = row.backend
        grouped[row.problem_name][f"{key_prefix}_mean_ms"] = row.mean_ms
        grouped[row.problem_name][f"{key_prefix}_workspace_mib"] = row.workspace_mib
        grouped[row.problem_name][f"{key_prefix}_max_abs_diff"] = row.max_abs_diff_vs_im2col
    for name, metrics in grouped.items():
        if "im2col_workspace_mib" in metrics and "dietconv_workspace_mib" in metrics:
            metrics["workspace_ratio_im2col_over_dietconv"] = (
                metrics["im2col_workspace_mib"] / metrics["dietconv_workspace_mib"]
            )
        if "im2col_mean_ms" in metrics and "dietconv_mean_ms" in metrics:
            metrics["speed_ratio_im2col_over_dietconv"] = (
                metrics["im2col_mean_ms"] / metrics["dietconv_mean_ms"]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(grouped, handle, indent=2, sort_keys=True)
