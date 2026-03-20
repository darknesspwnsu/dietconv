from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DietConv benchmark results.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/benchmark_results.csv"),
        help="CSV file emitted by scripts/run_benchmarks.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_plot.png"),
        help="Path to the generated chart image.",
    )
    args = parser.parse_args()

    rows = read_rows(args.results_csv)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[row["problem_name"]][row["backend"]] = row

    problems = list(by_problem.keys())
    im2col_ms = [float(by_problem[name]["im2col"]["mean_ms"]) for name in problems]
    dietconv_ms = [float(by_problem[name]["dietconv"]["mean_ms"]) for name in problems]
    im2col_ws = [float(by_problem[name]["im2col"]["workspace_mib"]) for name in problems]
    dietconv_ws = [float(by_problem[name]["dietconv"]["workspace_mib"]) for name in problems]

    positions = list(range(len(problems)))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].bar([p - width / 2 for p in positions], im2col_ms, width=width, label="im2col", color="#d96c06")
    axes[0].bar([p + width / 2 for p in positions], dietconv_ms, width=width, label="DietConv", color="#0d6b63")
    axes[0].set_ylabel("Mean runtime (ms)")
    axes[0].set_title("Runtime by problem")
    axes[0].set_xticks(positions, problems, rotation=15)
    axes[0].legend()

    axes[1].bar([p - width / 2 for p in positions], im2col_ws, width=width, label="im2col", color="#d96c06")
    axes[1].bar([p + width / 2 for p in positions], dietconv_ws, width=width, label="DietConv", color="#0d6b63")
    axes[1].set_ylabel("Workspace (MiB)")
    axes[1].set_title("Explicit workspace by problem")
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, problems, rotation=15)
    axes[1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
