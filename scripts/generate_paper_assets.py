from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"

COLORS = {
    "im2col": "#d96c06",
    "dietconv-v1": "#0d6b63",
    "dietconv-v2": "#2d4f8b",
    "torch-native": "#7a4e00",
    "torch-unfold": "#d96c06",
    "dietconv-v1-compiled": "#0d6b63",
    "dietconv-v2-compiled": "#2d4f8b",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def write_tex_table(path: Path, headers: list[str], rows: list[list[str]], caption: str, label: str) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines))


def publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_cpp_size() -> None:
    rows = read_rows(RESULTS / "cpp_size_scaling.csv")
    by_backend: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(row)
    for backend_rows in by_backend.values():
        backend_rows.sort(key=lambda row: int(row["h"]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    for backend, backend_rows in by_backend.items():
        xs = [int(row["h"]) for row in backend_rows]
        ys_ms = [float(row["mean_ms"]) for row in backend_rows]
        ys_ws = [float(row["workspace_mib"]) for row in backend_rows]
        axes[0].plot(xs, ys_ms, marker="o", linewidth=2, color=COLORS[backend], label=backend)
        axes[1].plot(xs, ys_ws, marker="o", linewidth=2, color=COLORS[backend], label=backend)
    axes[0].set_title("C++ Runtime")
    axes[0].set_xlabel("Input size")
    axes[0].set_ylabel("ms")
    axes[1].set_title("C++ Workspace")
    axes[1].set_xlabel("Input size")
    axes[1].set_ylabel("MiB")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper left")
    fig.savefig(FIGURES / "cpp_size_scaling.pdf")


def plot_torch_size() -> None:
    rows = read_rows(RESULTS / "torch_size_scaling.csv")
    by_backend: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(row)
    for backend_rows in by_backend.values():
        backend_rows.sort(key=lambda row: int(row["h"]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    for backend, backend_rows in by_backend.items():
        xs = [int(row["h"]) for row in backend_rows]
        ys_ms = [float(row["mean_ms"]) for row in backend_rows]
        ys_ws = [max(float(row["workspace_mib"]), 1e-6) for row in backend_rows]
        axes[0].plot(xs, ys_ms, marker="o", linewidth=2, color=COLORS[backend], label=backend)
        axes[1].plot(xs, ys_ws, marker="o", linewidth=2, color=COLORS[backend], label=backend)
    axes[0].set_title("PyTorch Runtime")
    axes[0].set_xlabel("Input size")
    axes[0].set_ylabel("ms")
    axes[1].set_title("PyTorch Explicit Workspace")
    axes[1].set_xlabel("Input size")
    axes[1].set_ylabel("MiB")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper left")
    fig.savefig(FIGURES / "torch_size_scaling.pdf")


def plot_torch_memory() -> None:
    rows = read_rows(RESULTS / "torch_memory_size_scaling.csv")
    by_backend: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(row)
    for backend_rows in by_backend.values():
        backend_rows.sort(key=lambda row: int(row["h"]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    for backend, backend_rows in by_backend.items():
        xs = [int(row["h"]) for row in backend_rows]
        peak = [float(row["peak_rss_mib"]) for row in backend_rows]
        delta = [max(float(row["rss_delta_mib"]), 1e-6) for row in backend_rows]
        axes[0].plot(xs, peak, marker="o", linewidth=2, color=COLORS[backend], label=backend)
        axes[1].plot(xs, delta, marker="o", linewidth=2, color=COLORS[backend], label=backend)
    axes[0].set_title("PyTorch Peak RSS")
    axes[0].set_xlabel("Input size")
    axes[0].set_ylabel("MiB")
    axes[1].set_title("PyTorch RSS Delta")
    axes[1].set_xlabel("Input size")
    axes[1].set_ylabel("MiB")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper left")
    fig.savefig(FIGURES / "torch_memory_scaling.pdf")


def plot_v2_ablation() -> None:
    rows = read_rows(RESULTS / "paper_v2_ablation_cpp.csv")
    problems = sorted({row["problem_name"] for row in rows})
    fig, axes = plt.subplots(1, len(problems), figsize=(3.6 * len(problems), 3.2), constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for axis, problem in zip(axes, problems):
        problem_rows = [row for row in rows if row["problem_name"] == problem]
        problem_rows.sort(key=lambda row: int(row["tile_out_width"]))
        xs = [int(row["tile_out_width"]) for row in problem_rows]
        ys = [float(row["mean_ms"]) for row in problem_rows]
        colors = ["#c0392b" if row["mode"] == "autotuned" else "#2d4f8b" for row in problem_rows]
        axis.scatter(xs, ys, c=colors, s=35)
        axis.plot(xs, ys, color="#2d4f8b", linewidth=1.5, alpha=0.8)
        axis.set_title(problem)
        axis.set_xlabel("Tile width")
        axis.set_ylabel("ms")
    fig.savefig(FIGURES / "v2_tile_ablation_cpp.pdf")


def build_tables() -> None:
    cpp_rows = read_rows(RESULTS / "cpp_size_scaling.csv")
    torch_rows = read_rows(RESULTS / "torch_size_scaling.csv")
    memory_rows = read_rows(RESULTS / "torch_memory_size_scaling.csv")

    cpp_by_problem: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in cpp_rows:
        cpp_by_problem[row["problem_name"]][row["backend"]] = row
    cpp_table = []
    for problem in sorted(cpp_by_problem.keys(), key=lambda name: int(name.split("-")[-1])):
        data = cpp_by_problem[problem]
        cpp_table.append(
            [
                problem.replace("scale-", ""),
                f"{float(data['im2col']['mean_ms']):.2f}",
                f"{float(data['dietconv-v1']['mean_ms']):.2f}",
                f"{float(data['dietconv-v2']['mean_ms']):.2f}",
                f"{float(data['dietconv-v2']['workspace_mib']):.3f}",
            ]
        )
    write_tex_table(
        TABLES / "cpp_runtime_summary.tex",
        ["Input", "im2col", "v1", "v2", "v2 ws (MiB)"],
        cpp_table,
        "Representative C++ runtime and explicit workspace results.",
        "tab:cpp-runtime",
    )

    torch_by_problem: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in torch_rows:
        torch_by_problem[row["problem_name"]][row["backend"]] = row
    torch_table = []
    for problem in sorted(torch_by_problem.keys(), key=lambda name: int(name.split("-")[-1])):
        data = torch_by_problem[problem]
        torch_table.append(
            [
                problem.replace("scale-", ""),
                f"{float(data['torch-native']['mean_ms']):.2f}",
                f"{float(data['torch-unfold']['mean_ms']):.2f}",
                f"{float(data['dietconv-v1-compiled']['mean_ms']):.2f}",
                f"{float(data['dietconv-v2-compiled']['mean_ms']):.2f}",
            ]
        )
    write_tex_table(
        TABLES / "torch_runtime_summary.tex",
        ["Input", "native", "unfold", "v1", "v2"],
        torch_table,
        "Representative PyTorch runtime results.",
        "tab:torch-runtime",
    )

    memory_by_problem: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in memory_rows:
        memory_by_problem[row["problem_name"]][row["backend"]] = row
    memory_table = []
    for problem in sorted(memory_by_problem.keys(), key=lambda name: int(name.split("-")[-1])):
        data = memory_by_problem[problem]
        memory_table.append(
            [
                problem.replace("scale-", ""),
                f"{float(data['torch-native']['rss_delta_mib']):.2f}",
                f"{float(data['torch-unfold']['rss_delta_mib']):.2f}",
                f"{float(data['dietconv-v1-compiled']['rss_delta_mib']):.2f}",
                f"{float(data['dietconv-v2-compiled']['rss_delta_mib']):.2f}",
            ]
        )
    write_tex_table(
        TABLES / "torch_memory_summary.tex",
        ["Input", "native", "unfold", "v1", "v2"],
        memory_table,
        "Representative isolated-process RSS deltas for PyTorch backends.",
        "tab:torch-memory",
    )


def main() -> None:
    ensure_dirs()
    publication_style()
    plot_cpp_size()
    plot_torch_size()
    plot_torch_memory()
    plot_v2_ablation()
    build_tables()
    print(f"Wrote paper assets to {PAPER}")


if __name__ == "__main__":
    main()
