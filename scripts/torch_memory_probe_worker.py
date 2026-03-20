from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dietconv.torch_ops import (
    autotune_dietconv_v2_tile_width,
    dietconv2d_v1_compiled_prepacked,
    dietconv2d_v2_compiled_prepacked,
    load_dietconv_extension,
    prepack_dietconv_weight,
    unfold_conv2d,
    workspace_bytes_dietconv2d_v1,
    workspace_bytes_dietconv2d_v2,
    workspace_bytes_unfold,
)


NUMERIC_TOLERANCE = 1e-4


def make_inputs(problem: dict[str, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((1, problem["c"], problem["h"], problem["w"]), generator=generator, dtype=torch.float32)
    weight = torch.randn(
        (problem["k"], problem["c"], problem["fh"], problem["fw"]),
        generator=generator,
        dtype=torch.float32,
    )
    return x, weight


def build_runner(problem: dict[str, int], backend: str, seed: int) -> tuple[callable, dict[str, float | int]]:
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
        fn = lambda: dietconv2d_v1_compiled_prepacked(
            x,
            packed_weight,
            stride=problem["stride"],
            padding=problem["pad"],
        )
        workspace_bytes = workspace_bytes_dietconv2d_v1(x, weight, padding=problem["pad"])
    elif backend == "dietconv-v2-compiled":
        packed_weight = prepack_dietconv_weight(weight)
        tile_out_width = autotune_dietconv_v2_tile_width(
            x,
            packed_weight,
            stride=problem["stride"],
            padding=problem["pad"],
        )
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

    metadata: dict[str, float | int] = {
        "tile_out_width": tile_out_width,
        "workspace_bytes": workspace_bytes,
        "workspace_mib": workspace_bytes / (1024.0 * 1024.0),
        "reference_checksum": float(reference.sum().item()),
    }

    def wrapped() -> torch.Tensor:
        output = fn()
        max_abs_diff = float((output - reference).abs().max().item())
        if max_abs_diff > NUMERIC_TOLERANCE:
            raise AssertionError(f"{backend} exceeded numeric tolerance: {max_abs_diff} > {NUMERIC_TOLERANCE}")
        metadata["max_abs_diff_vs_native"] = max_abs_diff
        metadata["checksum"] = float(output.sum().item())
        return output

    return wrapped, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker process for torch RSS benchmark probes.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--c", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--fh", type=int, required=True)
    parser.add_argument("--fw", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--pad", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    load_dietconv_extension()

    problem = {
        "c": args.c,
        "h": args.h,
        "w": args.w,
        "k": args.k,
        "fh": args.fh,
        "fw": args.fw,
        "stride": args.stride,
        "pad": args.pad,
    }
    fn, metadata = build_runner(problem, args.backend, args.seed)

    ready = {"pid": os.getpid(), "status": "ready"}
    print(json.dumps(ready), flush=True)
    sys.stdin.readline()

    with torch.no_grad():
        for _ in range(args.warmup):
            fn()
        timings = []
        result = None
        for _ in range(args.repeat):
            start = time.perf_counter()
            result = fn()
            timings.append((time.perf_counter() - start) * 1000.0)

    assert result is not None
    mean_ms = sum(timings) / len(timings)
    variance = sum((timing - mean_ms) ** 2 for timing in timings) / len(timings)
    std_ms = variance ** 0.5
    metadata.update(
        {
            "backend": args.backend,
            "threads": args.threads,
            "c": args.c,
            "h": args.h,
            "w": args.w,
            "k": args.k,
            "fh": args.fh,
            "fw": args.fw,
            "stride": args.stride,
            "pad": args.pad,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
        }
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
