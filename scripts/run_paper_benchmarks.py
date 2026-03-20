from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "paper" / "configs" / "paper_benchmarks.json"


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen paper benchmark configuration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())

    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_benchmarks.py"),
            "--repeat",
            str(config["numpy"]["repeat"]),
            "--warmup",
            str(config["numpy"]["warmup"]),
            "--seed",
            str(config["numpy"]["seed"]),
        ]
    )
    run_command([sys.executable, str(ROOT / "scripts" / "plot_benchmarks.py")])
    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "showcase_cnn.py"),
            "--seed",
            str(config["cnn_showcase"]["seed"]),
        ]
    )
    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_cpp_benchmarks.py"),
            "--repeat",
            str(config["cpp"]["repeat"]),
            "--warmup",
            str(config["cpp"]["warmup"]),
            "--seed",
            str(config["cpp"]["seed"]),
        ]
    )
    run_command([sys.executable, str(ROOT / "scripts" / "plot_cpp_benchmarks.py")])
    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_torch_benchmarks.py"),
            "--repeat",
            str(config["torch"]["repeat"]),
            "--warmup",
            str(config["torch"]["warmup"]),
            "--seed",
            str(config["torch"]["seed"]),
        ]
    )
    run_command([sys.executable, str(ROOT / "scripts" / "plot_torch_benchmarks.py")])
    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_torch_memory_benchmarks.py"),
            "--repeat",
            str(config["torch_memory"]["repeat"]),
            "--warmup",
            str(config["torch_memory"]["warmup"]),
            "--seed",
            str(config["torch_memory"]["seed"]),
        ]
    )
    run_command([sys.executable, str(ROOT / "scripts" / "plot_torch_memory_benchmarks.py")])
    run_command([sys.executable, str(ROOT / "scripts" / "run_v2_ablations.py")])
    run_command([sys.executable, str(ROOT / "scripts" / "generate_paper_assets.py")])

    print(f"Completed paper benchmark pipeline using {args.config}")


if __name__ == "__main__":
    main()
