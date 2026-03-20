from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "torch_memory_probe_worker.py"

SIZE_SWEEP = [32, 48, 64, 96]
THREAD_SWEEP = [1, 2, 4, 8]
BACKENDS = ["torch-native", "torch-unfold", "dietconv-v1-compiled", "dietconv-v2-compiled"]
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


def read_rss_mib(pid: int) -> float | None:
    completed = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    stripped = completed.stdout.strip()
    if not stripped:
        return None
    rss_kib = int(stripped)
    return rss_kib / 1024.0


def run_case(problem: dict[str, int], backend: str, threads: int, seed: int, repeat: int, warmup: int) -> dict:
    command = [
        sys.executable,
        str(WORKER),
        "--backend",
        backend,
        "--c",
        str(problem["c"]),
        "--h",
        str(problem["h"]),
        "--w",
        str(problem["w"]),
        "--k",
        str(problem["k"]),
        "--fh",
        str(problem["fh"]),
        "--fw",
        str(problem["fw"]),
        "--stride",
        str(problem["stride"]),
        "--pad",
        str(problem["pad"]),
        "--threads",
        str(threads),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--seed",
        str(seed),
    ]
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdin is not None
    ready_line = proc.stdout.readline().strip()
    if not ready_line:
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        raise RuntimeError(f"memory worker failed before ready signal: {stderr}")
    ready = json.loads(ready_line)
    pid = int(ready["pid"])

    baseline_rss_mib = read_rss_mib(pid)
    peak_rss_mib = baseline_rss_mib if baseline_rss_mib is not None else 0.0

    proc.stdin.write("\n")
    proc.stdin.flush()

    while proc.poll() is None:
        sampled = read_rss_mib(pid)
        if sampled is not None:
            peak_rss_mib = max(peak_rss_mib, sampled)
        time.sleep(0.005)

    sampled = read_rss_mib(pid)
    if sampled is not None:
        peak_rss_mib = max(peak_rss_mib, sampled)

    result_line = proc.stdout.readline().strip()
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    if proc.returncode != 0:
        raise RuntimeError(f"memory worker failed: {stderr}")
    if not result_line:
        raise RuntimeError(f"memory worker returned no result. stderr={stderr}")
    row = json.loads(result_line)
    row["baseline_rss_mib"] = baseline_rss_mib
    row["peak_rss_mib"] = peak_rss_mib
    if baseline_rss_mib is None:
        row["rss_delta_mib"] = None
    else:
        row["rss_delta_mib"] = max(peak_rss_mib - baseline_rss_mib, 0.0)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated-process torch memory benchmarks.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

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
        for backend in BACKENDS:
            row = run_case(problem, backend, threads=1, seed=args.seed + index, repeat=args.repeat, warmup=args.warmup)
            row["problem_name"] = f"scale-{size}"
            row["sweep"] = "size"
            size_rows.append(row)

    thread_rows = []
    for problem_index, problem in enumerate(THREAD_PROBLEMS):
        for threads in THREAD_SWEEP:
            for backend in BACKENDS:
                row = run_case(
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

    write_csv(args.results_dir / "torch_memory_size_scaling.csv", size_rows)
    write_csv(args.results_dir / "torch_memory_thread_scaling.csv", thread_rows)
    with (args.results_dir / "torch_memory_summary.json").open("w") as handle:
        json.dump(
            {
                "size_scaling_cases": len(size_rows),
                "thread_scaling_cases": len(thread_rows),
                "platform": platform.platform(),
                "rss_sampling_interval_ms": 5,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    subprocess.run([sys.executable, str(ROOT / "scripts" / "update_readme_benchmarks.py")], check=True, cwd=ROOT)
    print(f"Wrote torch memory benchmark results to {args.results_dir}")


if __name__ == "__main__":
    main()
