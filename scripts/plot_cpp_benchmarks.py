from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


COLORS = {
    "im2col": "#d96c06",
    "dietconv-v1": "#0d6b63",
    "dietconv-v2": "#2d4f8b",
}


def plot_size_scaling(rows: list[dict[str, str]], output_dir: Path) -> None:
    by_backend: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(row)
    for backend_rows in by_backend.values():
        backend_rows.sort(key=lambda row: int(row["h"]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for backend, backend_rows in by_backend.items():
        xs = [int(row["h"]) for row in backend_rows]
        ys_ms = [float(row["mean_ms"]) for row in backend_rows]
        ys_ws = [float(row["workspace_mib"]) for row in backend_rows]
        label = backend.replace("dietconv", "DietConv")
        axes[0].plot(xs, ys_ms, marker="o", linewidth=2, color=COLORS[backend], label=label)
        axes[1].plot(xs, ys_ws, marker="o", linewidth=2, color=COLORS[backend], label=label)

    axes[0].set_title("C++ runtime vs input size (1 thread)")
    axes[0].set_xlabel("Input height/width")
    axes[0].set_ylabel("Mean runtime (ms)")
    axes[0].legend()

    axes[1].set_title("C++ explicit workspace vs input size (1 thread)")
    axes[1].set_xlabel("Input height/width")
    axes[1].set_ylabel("Workspace (MiB)")
    axes[1].set_yscale("log")
    axes[1].legend()

    fig.savefig(output_dir / "cpp_size_scaling.png", dpi=160)


def plot_thread_scaling(rows: list[dict[str, str]], output_dir: Path) -> None:
    problems = sorted({row["problem_name"] for row in rows})
    fig_runtime, runtime_axes = plt.subplots(1, len(problems), figsize=(6 * len(problems), 4), constrained_layout=True)
    fig_workspace, workspace_axes = plt.subplots(1, len(problems), figsize=(6 * len(problems), 4), constrained_layout=True)

    if len(problems) == 1:
        runtime_axes = [runtime_axes]
        workspace_axes = [workspace_axes]

    for axis, workspace_axis, problem in zip(runtime_axes, workspace_axes, problems):
        problem_rows = [row for row in rows if row["problem_name"] == problem]
        by_backend: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in problem_rows:
            by_backend[row["backend"]].append(row)
        for backend_rows in by_backend.values():
            backend_rows.sort(key=lambda row: int(row["threads"]))

        for backend, backend_rows in by_backend.items():
            xs = [int(row["threads"]) for row in backend_rows]
            ys_ms = [float(row["mean_ms"]) for row in backend_rows]
            ys_ws = [float(row["workspace_mib"]) for row in backend_rows]
            label = backend.replace("dietconv", "DietConv")
            axis.plot(xs, ys_ms, marker="o", linewidth=2, color=COLORS[backend], label=label)
            workspace_axis.plot(xs, ys_ws, marker="o", linewidth=2, color=COLORS[backend], label=label)

        axis.set_title(f"Runtime vs threads: {problem}")
        axis.set_xlabel("Threads")
        axis.set_ylabel("Mean runtime (ms)")
        axis.set_xticks([1, 2, 4, 8])
        axis.legend()

        workspace_axis.set_title(f"Workspace vs threads: {problem}")
        workspace_axis.set_xlabel("Threads")
        workspace_axis.set_ylabel("Workspace (MiB)")
        workspace_axis.set_xticks([1, 2, 4, 8])
        workspace_axis.set_yscale("log")
        workspace_axis.legend()

    fig_runtime.savefig(output_dir / "cpp_thread_runtime.png", dpi=160)
    fig_workspace.savefig(output_dir / "cpp_thread_workspace.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the C++ DietConv benchmark outputs.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory containing cpp_size_scaling.csv and cpp_thread_scaling.csv",
    )
    args = parser.parse_args()

    size_rows = read_rows(args.results_dir / "cpp_size_scaling.csv")
    thread_rows = read_rows(args.results_dir / "cpp_thread_scaling.csv")
    plot_size_scaling(size_rows, args.results_dir)
    plot_thread_scaling(thread_rows, args.results_dir)
    print(f"Wrote C++ benchmark plots to {args.results_dir}")


if __name__ == "__main__":
    main()
