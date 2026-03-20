from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
START_MARKER = "<!-- BENCHMARK_DIGEST:START -->"
END_MARKER = "<!-- BENCHMARK_DIGEST:END -->"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_csv_optional(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    return read_csv(path)


def read_json_optional(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt_ms(value: float) -> str:
    return f"{value:.2f}"


def fmt_mib(value: float) -> str:
    return f"{value:.3f}"


def fmt_ratio(value: float) -> str:
    return f"{value:.1f}x"


def fmt_speedup(baseline_ms: float, candidate_ms: float) -> str:
    if candidate_ms <= 0:
        return "n/a"
    return fmt_ratio(baseline_ms / candidate_ms)


def winner_label(pairs: list[tuple[str, float]]) -> str:
    return min(pairs, key=lambda item: item[1])[0]


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_numpy_section() -> str:
    summary = read_json_optional(ROOT / "results" / "benchmark_summary.json")
    if summary is None:
        return "\n".join(
            [
                "### NumPy baseline",
                "",
                "_Run `python3 scripts/run_benchmarks.py` and `python3 scripts/showcase_cnn.py` to populate this section._",
            ]
        )
    rows = []
    for name, metrics in summary.items():
        rows.append(
            [
                name,
                fmt_ms(metrics["im2col_mean_ms"]),
                fmt_ms(metrics["dietconv_mean_ms"]),
                fmt_mib(metrics["im2col_workspace_mib"]),
                fmt_mib(metrics["dietconv_workspace_mib"]),
                fmt_ratio(metrics["workspace_ratio_im2col_over_dietconv"]),
            ]
        )
    cnn_summary = read_json_optional(ROOT / "results" / "cnn_showcase.json")
    lines = [
        "### NumPy baseline",
        "",
        "This is the simple reference showcase: easy to inspect, not the most meaningful performance layer.",
        "",
        make_table(
            ["Problem", "im2col ms", "DietConv ms", "im2col MiB", "DietConv MiB", "Workspace reduction"],
            rows,
        ),
        "",
    ]
    if cnn_summary is not None:
        lines.extend(
            [
                f"- 3-layer CNN showcase peak workspace: `im2col {fmt_mib(cnn_summary['im2col']['peak_workspace_mib'])} MiB` vs `DietConv {fmt_mib(cnn_summary['dietconv']['peak_workspace_mib'])} MiB`.",
                f"- CNN showcase workspace reduction: `{fmt_ratio(cnn_summary['workspace_ratio_im2col_over_dietconv'])}` with checksum diff `{cnn_summary['checksum_abs_diff']}`.",
            ]
        )
    plot_path = ROOT / "results" / "benchmark_plot.png"
    if plot_path.exists():
        lines.extend(["", "![NumPy benchmark plot](results/benchmark_plot.png)"])
    return "\n".join(lines)


def summarize_cpp_size(rows: list[dict[str, str]]) -> str:
    by_problem: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_name"], {})[row["backend"]] = row
    table_rows = []
    v2_wins = 0
    best_speedup = ("", 1.0)
    best_workspace_ratio = ("", 1.0)
    for problem in sorted(by_problem.keys(), key=lambda name: int(name.split("-")[-1])):
        data = by_problem[problem]
        im2col_ms = float(data["im2col"]["mean_ms"])
        v1_ms = float(data["dietconv-v1"]["mean_ms"])
        v2_ms = float(data["dietconv-v2"]["mean_ms"])
        if v2_ms < v1_ms:
            v2_wins += 1
            speedup = v1_ms / v2_ms
            if speedup > best_speedup[1]:
                best_speedup = (problem, speedup)
        workspace_ratio = float(data["im2col"]["workspace_mib"]) / float(data["dietconv-v2"]["workspace_mib"])
        if workspace_ratio > best_workspace_ratio[1]:
            best_workspace_ratio = (problem, workspace_ratio)
        winner = winner_label(
            [("im2col", im2col_ms), ("v1", v1_ms), ("v2", v2_ms)]
        )
        table_rows.append(
            [
                problem.replace("scale-", ""),
                winner,
                fmt_ms(im2col_ms),
                fmt_ms(v1_ms),
                fmt_ms(v2_ms),
                fmt_speedup(v1_ms, v2_ms),
                fmt_mib(float(data["dietconv-v1"]["workspace_mib"])),
                fmt_mib(float(data["dietconv-v2"]["workspace_mib"])),
                fmt_ratio(workspace_ratio),
            ]
        )
    lines = [
        "### C++ benchmark digest",
        "",
        f"- On the 1-thread size sweep, `v2` beats `v1` on `{v2_wins}` of `{len(by_problem)}` tested sizes.",
        f"- Best `v2` speedup over `v1`: `{best_speedup[0].replace('scale-', '')}x{best_speedup[0].replace('scale-', '')}` input at `{best_speedup[1]:.2f}x`.",
        f"- Largest `im2col` to `v2` workspace reduction in this sweep: `{best_workspace_ratio[0].replace('scale-', '')}x{best_workspace_ratio[0].replace('scale-', '')}` input at `{best_workspace_ratio[1]:.1f}x`.",
        "- The C++ path is the clearest view of the algorithm itself because it strips out most Python overhead.",
        "",
        make_table(
            ["Input size", "Fastest", "im2col ms", "v1 ms", "v2 ms", "v2 vs v1", "v1 MiB", "v2 MiB", "im2col/v2 ws"],
            table_rows,
        ),
    ]
    return "\n".join(lines)


def summarize_cpp_threads(rows: list[dict[str, str]]) -> str:
    problems = sorted({row["problem_name"] for row in rows})
    blocks = []
    for problem in problems:
        problem_rows = [row for row in rows if row["problem_name"] == problem]
        by_threads: dict[str, dict[str, dict[str, str]]] = {}
        for row in problem_rows:
            by_threads.setdefault(row["threads"], {})[row["backend"]] = row
        table_rows = []
        v2_thread_wins = 0
        for thread in sorted(by_threads.keys(), key=int):
            data = by_threads[thread]
            im2col_ms = float(data["im2col"]["mean_ms"])
            v1_ms = float(data["dietconv-v1"]["mean_ms"])
            v2_ms = float(data["dietconv-v2"]["mean_ms"])
            if v2_ms < v1_ms:
                v2_thread_wins += 1
            table_rows.append(
                [
                    thread,
                    winner_label([("im2col", im2col_ms), ("v1", v1_ms), ("v2", v2_ms)]),
                    fmt_ms(im2col_ms),
                    fmt_ms(v1_ms),
                    fmt_ms(v2_ms),
                    fmt_mib(float(data["dietconv-v1"]["workspace_mib"])),
                    fmt_mib(float(data["dietconv-v2"]["workspace_mib"])),
                ]
            )
        blocks.extend(
            [
                "",
                f"**Thread sweep: `{problem}`**",
                "",
                f"- `v2` beats `v1` on `{v2_thread_wins}` of `{len(table_rows)}` tested thread counts.",
                "",
                make_table(["Threads", "Fastest", "im2col ms", "v1 ms", "v2 ms", "v1 MiB", "v2 MiB"], table_rows),
            ]
        )
    if (ROOT / "results" / "cpp_size_scaling.png").exists():
        blocks.extend(["", "![C++ size scaling](results/cpp_size_scaling.png)"])
    if (ROOT / "results" / "cpp_thread_runtime.png").exists():
        blocks.extend(["", "![C++ thread runtime](results/cpp_thread_runtime.png)"])
    if (ROOT / "results" / "cpp_thread_workspace.png").exists():
        blocks.extend(["", "![C++ thread workspace](results/cpp_thread_workspace.png)"])
    return "\n".join(blocks)


def summarize_torch_size(rows: list[dict[str, str]]) -> str:
    by_problem: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_name"], {})[row["backend"]] = row
    table_rows = []
    v2_wins = 0
    compiled_beats_unfold = 0
    for problem in sorted(by_problem.keys(), key=lambda name: int(name.split("-")[-1])):
        data = by_problem[problem]
        native_ms = float(data["torch-native"]["mean_ms"])
        unfold_ms = float(data["torch-unfold"]["mean_ms"])
        v1_ms = float(data["dietconv-v1-compiled"]["mean_ms"])
        v2_ms = float(data["dietconv-v2-compiled"]["mean_ms"])
        if v2_ms < v1_ms:
            v2_wins += 1
        if v2_ms < unfold_ms:
            compiled_beats_unfold += 1
        winner = winner_label(
            [("native", native_ms), ("unfold", unfold_ms), ("v1", v1_ms), ("v2", v2_ms)]
        )
        table_rows.append(
            [
                problem.replace("scale-", ""),
                winner,
                fmt_ms(native_ms),
                fmt_ms(unfold_ms),
                fmt_ms(v1_ms),
                fmt_ms(v2_ms),
                fmt_speedup(v1_ms, v2_ms),
                fmt_mib(float(data["dietconv-v2-compiled"]["workspace_mib"])),
            ]
        )
    return "\n".join(
        [
            "### PyTorch benchmark digest",
            "",
            f"- On the 1-thread size sweep, compiled `v2` beats compiled `v1` on `{v2_wins}` of `{len(by_problem)}` tested sizes.",
            f"- Compiled `v2` beats explicit `torch-unfold` on `{compiled_beats_unfold}` of `{len(by_problem)}` tested sizes.",
            "- The torch digest is the practical framework story: native `conv2d`, explicit `unfold`, and compiled DietConv side by side.",
            "",
            make_table(
                ["Input size", "Fastest", "native ms", "unfold ms", "v1 ms", "v2 ms", "v2 vs v1", "v2 MiB"],
                table_rows,
            ),
        ]
    )


def summarize_torch_threads(rows: list[dict[str, str]]) -> str:
    problems = sorted({row["problem_name"] for row in rows})
    worst_diff = max(float(row["max_abs_diff_vs_native"]) for row in rows)
    blocks = [f"", f"- Numeric guardrail: current worst torch max-abs diff vs native `conv2d` is `{worst_diff:.6g}`."]
    for problem in problems:
        problem_rows = [row for row in rows if row["problem_name"] == problem]
        by_threads: dict[str, dict[str, dict[str, str]]] = {}
        for row in problem_rows:
            by_threads.setdefault(row["threads"], {})[row["backend"]] = row
        table_rows = []
        v2_thread_wins = 0
        for thread in sorted(by_threads.keys(), key=int):
            data = by_threads[thread]
            native_ms = float(data["torch-native"]["mean_ms"])
            unfold_ms = float(data["torch-unfold"]["mean_ms"])
            v1_ms = float(data["dietconv-v1-compiled"]["mean_ms"])
            v2_ms = float(data["dietconv-v2-compiled"]["mean_ms"])
            if v2_ms < v1_ms:
                v2_thread_wins += 1
            table_rows.append(
                [
                    thread,
                    winner_label([("native", native_ms), ("unfold", unfold_ms), ("v1", v1_ms), ("v2", v2_ms)]),
                    fmt_ms(native_ms),
                    fmt_ms(unfold_ms),
                    fmt_ms(v1_ms),
                    fmt_ms(v2_ms),
                ]
            )
        blocks.extend(
            [
                "",
                f"**Thread sweep: `{problem}`**",
                "",
                f"- Compiled `v2` beats compiled `v1` on `{v2_thread_wins}` of `{len(table_rows)}` tested thread counts.",
                "",
                make_table(["Threads", "Fastest", "native ms", "unfold ms", "v1 ms", "v2 ms"], table_rows),
            ]
        )
    if (ROOT / "results" / "torch_size_scaling.png").exists():
        blocks.extend(["", "![PyTorch size scaling](results/torch_size_scaling.png)"])
    if (ROOT / "results" / "torch_thread_runtime.png").exists():
        blocks.extend(["", "![PyTorch thread runtime](results/torch_thread_runtime.png)"])
    return "\n".join(blocks)


def build_digest() -> str:
    cpp_size_rows = read_csv_optional(ROOT / "results" / "cpp_size_scaling.csv")
    cpp_thread_rows = read_csv_optional(ROOT / "results" / "cpp_thread_scaling.csv")
    torch_size_rows = read_csv_optional(ROOT / "results" / "torch_size_scaling.csv")
    torch_thread_rows = read_csv_optional(ROOT / "results" / "torch_thread_scaling.csv")

    sections = [
        "## Benchmark digest",
        "",
        "_This section is autogenerated from files in `results/` by `scripts/update_readme_benchmarks.py`._",
        "",
        "How to read these results:",
        "",
        "- Lower runtime is better. Lower workspace is better.",
        "- `im2col` and `torch-unfold` are the duplication-heavy baselines.",
        "- `v1` is the poster-faithful strip-buffer kernel. `v2` is the tiled strip-buffer variant.",
        "",
        render_numpy_section(),
        "",
        summarize_cpp_size(cpp_size_rows) if cpp_size_rows else "### C++ benchmark digest\n\n_Run `python3 scripts/run_cpp_benchmarks.py` to populate this section._",
        summarize_cpp_threads(cpp_thread_rows) if cpp_thread_rows else "",
        "",
        summarize_torch_size(torch_size_rows) if torch_size_rows else "### PyTorch benchmark digest\n\n_Run `python3 scripts/run_torch_benchmarks.py` to populate this section._",
        summarize_torch_threads(torch_thread_rows) if torch_thread_rows else "",
    ]
    return "\n".join(sections).strip() + "\n"


def replace_digest_section(readme: str, digest: str) -> str:
    start = readme.find(START_MARKER)
    end = readme.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("README markers for benchmark digest were not found.")
    start += len(START_MARKER)
    new_digest = "\n\n" + digest.strip() + "\n\n"
    return readme[:start] + new_digest + readme[end:]


def main() -> None:
    readme = README_PATH.read_text()
    updated = replace_digest_section(readme, build_digest())
    README_PATH.write_text(updated)
    print(f"Updated {README_PATH}")


if __name__ == "__main__":
    main()
