from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dietconv.benchmarking import benchmark_suite, save_rows_csv, save_summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DietConv benchmarks.")
    parser.add_argument("--repeat", type=int, default=5, help="Measured iterations per backend.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per backend.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write CSV and JSON summaries into.",
    )
    args = parser.parse_args()

    rows = benchmark_suite(repeat=args.repeat, warmup=args.warmup, seed=args.seed)
    save_rows_csv(rows, args.results_dir / "benchmark_results.csv")
    save_summary_json(rows, args.results_dir / "benchmark_summary.json")

    print(f"Wrote {len(rows)} benchmark rows to {args.results_dir}")


if __name__ == "__main__":
    main()
