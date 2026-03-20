from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "torch-native": "#7a4e00",
    "torch-unfold": "#d96c06",
    "dietconv-v1-compiled": "#0d6b63",
    "dietconv-v2-compiled": "#2d4f8b",
}
LABELS = {
    "torch-native": "torch-native",
    "torch-unfold": "torch-unfold",
    "dietconv-v1-compiled": "dietconv-v1-compiled",
    "dietconv-v2-compiled": "dietconv-v2-compiled",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def plot_size(rows: list[dict[str, str]], output_dir: Path) -> None:
    by_backend: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(row)
    for backend_rows in by_backend.values():
        backend_rows.sort(key=lambda row: int(row["h"]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for backend, backend_rows in by_backend.items():
        xs = [int(row["h"]) for row in backend_rows]
        ys_ms = [float(row["mean_ms"]) for row in backend_rows]
        ys_ws = [max(float(row["workspace_mib"]), 1e-6) for row in backend_rows]
        axes[0].plot(xs, ys_ms, marker="o", linewidth=2, color=COLORS[backend], label=LABELS[backend])
        axes[1].plot(xs, ys_ws, marker="o", linewidth=2, color=COLORS[backend], label=LABELS[backend])

    axes[0].set_title("PyTorch runtime vs input size (1 thread)")
    axes[0].set_xlabel("Input height/width")
    axes[0].set_ylabel("Mean runtime (ms)")
    axes[0].legend()

    axes[1].set_title("PyTorch explicit lowering workspace vs input size (1 thread)")
    axes[1].set_xlabel("Input height/width")
    axes[1].set_ylabel("Workspace (MiB)")
    axes[1].set_yscale("log")
    axes[1].legend()

    fig.savefig(output_dir / "torch_size_scaling.png", dpi=160)


def plot_threads(rows: list[dict[str, str]], output_dir: Path) -> None:
    problems = sorted({row["problem_name"] for row in rows})
    fig, axes = plt.subplots(1, len(problems), figsize=(6 * len(problems), 4), constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for axis, problem in zip(axes, problems):
        problem_rows = [row for row in rows if row["problem_name"] == problem]
        by_backend: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in problem_rows:
            by_backend[row["backend"]].append(row)
        for backend_rows in by_backend.values():
            backend_rows.sort(key=lambda row: int(row["threads"]))
        for backend, backend_rows in by_backend.items():
            xs = [int(row["threads"]) for row in backend_rows]
            ys = [float(row["mean_ms"]) for row in backend_rows]
            axis.plot(xs, ys, marker="o", linewidth=2, color=COLORS[backend], label=LABELS[backend])
        axis.set_title(f"PyTorch runtime vs threads: {problem}")
        axis.set_xlabel("Threads")
        axis.set_ylabel("Mean runtime (ms)")
        axis.set_xticks([1, 2, 4, 8])
        axis.legend()
    fig.savefig(output_dir / "torch_thread_runtime.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PyTorch DietConv benchmark outputs.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    size_rows = read_rows(args.results_dir / "torch_size_scaling.csv")
    thread_rows = read_rows(args.results_dir / "torch_thread_scaling.csv")
    plot_size(size_rows, args.results_dir)
    plot_threads(thread_rows, args.results_dir)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "update_readme_benchmarks.py")], check=True, cwd=ROOT)
    print(f"Wrote torch benchmark plots to {args.results_dir}")


if __name__ == "__main__":
    main()
