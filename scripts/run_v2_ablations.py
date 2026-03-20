from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dietconv.torch_ops import (
    autotune_dietconv_v2_tile_width,
    dietconv2d_v2_compiled_prepacked,
    load_dietconv_extension,
    prepack_dietconv_weight,
    workspace_bytes_dietconv2d_v2,
)
from run_cpp_benchmarks import autotune_v2_case, build_binary, run_case as run_cpp_case


DEFAULT_CONFIG = ROOT / "paper" / "configs" / "v2_ablations.json"
NUMERIC_TOLERANCE = 1e-4


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_inputs(problem: dict[str, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((1, problem["c"], problem["h"], problem["w"]), generator=generator, dtype=torch.float32)
    weight = torch.randn((problem["k"], problem["c"], problem["fh"], problem["fw"]), generator=generator, dtype=torch.float32)
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


def run_torch_ablation_case(problem: dict[str, int], tile_out_width: int, seed: int, repeat: int, warmup: int) -> dict:
    torch.set_num_threads(problem["threads"])
    x, weight = make_inputs(problem, seed)
    packed = prepack_dietconv_weight(weight)
    reference = F.conv2d(x, weight, stride=problem["stride"], padding=problem["pad"])
    fn = lambda: dietconv2d_v2_compiled_prepacked(
        x,
        packed,
        stride=problem["stride"],
        padding=problem["pad"],
        tile_out_width=tile_out_width,
    )
    output, mean_ms, std_ms = time_call(fn, repeat=repeat, warmup=warmup)
    max_abs_diff = float((output - reference).abs().max().item())
    if max_abs_diff > NUMERIC_TOLERANCE:
        raise AssertionError(f"torch v2 ablation exceeded tolerance: {max_abs_diff}")
    return {
        "implementation": "torch",
        "problem_name": problem["problem_name"],
        "threads": problem["threads"],
        "tile_out_width": tile_out_width,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "workspace_mib": workspace_bytes_dietconv2d_v2(
            x,
            weight,
            stride=problem["stride"],
            padding=problem["pad"],
            tile_out_width=tile_out_width,
        )
        / (1024.0 * 1024.0),
        "max_abs_diff_vs_native": max_abs_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper ablations for DietConv v2 tile scheduling.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    build_binary()
    load_dietconv_extension()

    cpp_rows = []
    for problem_index, problem in enumerate(config["cpp"]["problems"]):
        autotuned = autotune_v2_case(
            {
                "backend": "dietconv-v2",
                "repeat": config["cpp"]["repeat"],
                "warmup": config["cpp"]["warmup"],
                "seed": config["cpp"]["seed"] + problem_index,
                "tile-out-width": 0,
                **{key: value for key, value in problem.items() if key != "problem_name"},
            }
        )
        for tile_out_width in sorted(set(config["cpp"]["tile_out_widths"] + [autotuned])):
            row = run_cpp_case(
                {
                    "backend": "dietconv-v2",
                    "repeat": config["cpp"]["repeat"],
                    "warmup": config["cpp"]["warmup"],
                    "seed": config["cpp"]["seed"] + problem_index,
                    "tile-out-width": tile_out_width,
                    **{key: value for key, value in problem.items() if key != "problem_name"},
                }
            )
            row["implementation"] = "cpp"
            row["problem_name"] = problem["problem_name"]
            row["mode"] = "autotuned" if tile_out_width == autotuned else "fixed"
            cpp_rows.append(row)

    torch_rows = []
    for problem_index, problem in enumerate(config["torch"]["problems"]):
        x, weight = make_inputs(problem, config["torch"]["seed"] + problem_index)
        packed = prepack_dietconv_weight(weight)
        autotuned = autotune_dietconv_v2_tile_width(
            x,
            packed,
            stride=problem["stride"],
            padding=problem["pad"],
            repeat=config["torch"]["repeat"],
            warmup=config["torch"]["warmup"],
            reuse_cache=False,
        )
        for tile_out_width in sorted(set(config["torch"]["tile_out_widths"] + [autotuned])):
            row = run_torch_ablation_case(
                problem,
                tile_out_width=tile_out_width,
                seed=config["torch"]["seed"] + problem_index,
                repeat=config["torch"]["repeat"],
                warmup=config["torch"]["warmup"],
            )
            row["mode"] = "autotuned" if tile_out_width == autotuned else "fixed"
            torch_rows.append(row)

    write_csv(args.results_dir / "paper_v2_ablation_cpp.csv", cpp_rows)
    write_csv(args.results_dir / "paper_v2_ablation_torch.csv", torch_rows)
    print(f"Wrote v2 ablations to {args.results_dir}")


if __name__ == "__main__":
    main()
