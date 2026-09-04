from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw
import torch
from kneed import KneeLocator
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx


DATASETS = ["Cora", "Citeseer", "Pubmed"]
CENTRALITIES = ["pagerank", "betweenness", "closeness", "eigenvector"]
DISPLAY_CENTRALITY = {
    "pagerank": "PageRank",
    "betweenness": "Betweenness",
    "closeness": "Closeness",
    "eigenvector": "Eigenvector",
}
BLUE = "#1a5fa8"
ORANGE = "#e05a00"
FINAL_RE = re.compile(r"Final Best: AUC=([0-9.]+), AP=([0-9.]+), MRR=([0-9.]+)")
EPOCH_RE = re.compile(r"Epoch\s+\d+:\s+AUC=([0-9.]+),\s+AP=([0-9.]+),\s+MRR=([0-9.]+)")
CSV_FIELDS = [
    "section",
    "dataset",
    "centrality",
    "elbow_k",
    "elbow_score",
    "alpha",
    "xmin",
    "r_vs_exp",
    "p_value",
    "verdict",
    "k",
    "seed",
    "val_auc",
    "test_auc",
    "mean_val_auc",
    "std_val_auc",
    "mean_test_auc",
    "std_test_auc",
    "optimal_k",
    "auc_at_elbow",
    "auc_at_optimal",
    "auc_gap",
    "match",
]


@dataclass
class CurveResult:
    dataset: str
    centrality: str
    elbow_k: int
    elbow_score: float
    scores: np.ndarray


@dataclass
class PowerlawResult:
    dataset: str
    centrality: str
    alpha: float
    xmin: float
    r_vs_exp: float
    p_value: float
    verdict: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_planetoid_graph(dataset: str, data_root: Path) -> nx.Graph:
    pyg_dataset = Planetoid(root=str(data_root / dataset), name=dataset)
    data = pyg_dataset[0]
    graph = to_networkx(data, to_undirected=True, remove_self_loops=True)
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def compute_centrality(graph: nx.Graph, centrality: str) -> Dict[int, float]:
    if centrality == "pagerank":
        return nx.pagerank(graph, alpha=0.85)
    if centrality == "betweenness":
        return nx.betweenness_centrality(graph, normalized=True)
    if centrality == "closeness":
        return nx.closeness_centrality(graph)
    if centrality == "eigenvector":
        try:
            return nx.eigenvector_centrality(graph, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            return nx.eigenvector_centrality_numpy(graph)
    raise ValueError(f"Unsupported centrality: {centrality}")


def ranked_scores(scores: Dict[int, float] | Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores.values()) if isinstance(scores, dict) else list(scores), dtype=np.float64)
    values = values[np.isfinite(values)]
    values = np.clip(values, 0.0, None)
    return np.sort(values)[::-1]


def detect_elbow(scores: np.ndarray) -> Tuple[int, float]:
    x = np.arange(1, len(scores) + 1)
    locator = KneeLocator(
        x,
        scores,
        curve="convex",
        direction="decreasing",
        interp_method="polynomial",
    )
    elbow = locator.knee
    if elbow is None:
        elbow = max(1, int(round(math.sqrt(len(scores)))))
    elbow = int(max(1, min(len(scores), elbow)))
    return elbow, float(scores[elbow - 1])


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def save_table_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def reset_csv(path: Path) -> None:
    if path.exists():
        path.unlink()


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def run_experiment_1(args: argparse.Namespace) -> Dict[Tuple[str, str], CurveResult]:
    print("\n[Experiment 1] Centrality ranked curves + elbow detection")
    results: Dict[Tuple[str, str], CurveResult] = {}
    for dataset in args.datasets:
        graph = load_planetoid_graph(dataset, Path(args.data_root))
        print(f"  loaded {dataset}: |V|={graph.number_of_nodes()} |E|={graph.number_of_edges()}")
        for centrality in args.centralities:
            values = compute_centrality(graph, centrality)
            scores = ranked_scores(values)
            elbow_k, elbow_score = detect_elbow(scores)
            results[(dataset, centrality)] = CurveResult(dataset, centrality, elbow_k, elbow_score, scores)

    set_plot_style()
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), squeeze=False)
    for r, dataset in enumerate(args.datasets):
        for c, centrality in enumerate(args.centralities):
            ax = axes[r][c]
            item = results[(dataset, centrality)]
            x = np.arange(1, len(item.scores) + 1)
            ax.plot(x, item.scores, color=BLUE, linewidth=1.4)
            ax.fill_between(x, item.scores, color=BLUE, alpha=0.12)
            ax.axvline(item.elbow_k, color=ORANGE, linestyle="--", linewidth=1.1)
            ax.scatter([item.elbow_k], [item.elbow_score], color=ORANGE, s=22, zorder=5)
            ax.annotate(
                f"K={item.elbow_k}",
                xy=(item.elbow_k, item.elbow_score),
                xytext=(5, 5),
                textcoords="offset points",
                color=ORANGE,
                fontsize=7,
            )
            ax.set_title(f"{dataset} x {DISPLAY_CENTRALITY[centrality]}")
            ax.set_xlabel("Node rank")
            ax.set_ylabel("Centrality score")
            ax.grid(False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(Path(args.figures_dir) / f"fig_ranked_curves.{ext}", dpi=150)
    plt.close(fig)

    rows = [
        {
            "section": "elbow",
            "dataset": item.dataset,
            "centrality": item.centrality,
            "elbow_k": item.elbow_k,
            "elbow_score": item.elbow_score,
        }
        for item in results.values()
    ]
    save_table_csv(Path(args.output_csv), rows)

    print("Dataset   | Centrality  | Elbow K")
    print("----------|-------------|--------")
    for dataset in args.datasets:
        for centrality in args.centralities:
            item = results[(dataset, centrality)]
            print(f"{dataset:<9} | {DISPLAY_CENTRALITY[centrality]:<11} | {item.elbow_k}")
    return results


def fit_powerlaw(scores: np.ndarray) -> PowerlawResult:
    raise RuntimeError("fit_powerlaw requires dataset and centrality; use run_experiment_2")


def run_experiment_2(args: argparse.Namespace, curves: Dict[Tuple[str, str], CurveResult]) -> Dict[Tuple[str, str], PowerlawResult]:
    print("\n[Experiment 2] Power-law / heavy-tail verification")
    results: Dict[Tuple[str, str], PowerlawResult] = {}
    for key, item in curves.items():
        scores_nonzero = item.scores[item.scores > 0]
        fit = powerlaw.Fit(scores_nonzero, discrete=False, verbose=False)
        r_vs_exp, p_value = fit.distribution_compare("power_law", "exponential")
        verdict = "power-law \u2713" if r_vs_exp > 0 and p_value < 0.05 else "exponential \u25b3"
        results[key] = PowerlawResult(
            item.dataset,
            item.centrality,
            float(fit.power_law.alpha),
            float(fit.power_law.xmin),
            float(r_vs_exp),
            float(p_value),
            verdict,
        )

    set_plot_style()
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), squeeze=False)
    for r, dataset in enumerate(args.datasets):
        for c, centrality in enumerate(args.centralities):
            ax = axes[r][c]
            item = curves[(dataset, centrality)]
            stats = results[(dataset, centrality)]
            scores = item.scores[item.scores > 0]
            ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
            ax.loglog(ranks, scores, ".", color=BLUE, markersize=2.5, alpha=0.75, label="data")
            exponent = -1.0 / max(stats.alpha - 1.0, 1e-6)
            fit_line = scores[0] * np.power(ranks / ranks[0], exponent)
            ax.loglog(ranks, fit_line, "--", color=ORANGE, linewidth=1.2, label=f"alpha={stats.alpha:.2f}")
            ax.set_title(f"{dataset} x {DISPLAY_CENTRALITY[centrality]}")
            ax.set_xlabel("log(rank)")
            ax.set_ylabel("log(score)")
            ax.legend(frameon=False)
            ax.grid(True, linewidth=0.35, alpha=0.35)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(Path(args.figures_dir) / f"fig_loglog.{ext}", dpi=150)
    plt.close(fig)

    rows = [
        {
            "section": "powerlaw",
            "dataset": item.dataset,
            "centrality": item.centrality,
            "alpha": item.alpha,
            "xmin": item.xmin,
            "r_vs_exp": item.r_vs_exp,
            "p_value": item.p_value,
            "verdict": item.verdict,
        }
        for item in results.values()
    ]
    save_table_csv(Path(args.output_csv), rows)

    print("Dataset   | Centrality  | alpha | R (vs exp) | p | Verdict")
    print("----------|-------------|-------|------------|---|----------")
    for dataset in args.datasets:
        for centrality in args.centralities:
            item = results[(dataset, centrality)]
            print(
                f"{dataset:<9} | {DISPLAY_CENTRALITY[centrality]:<11} | "
                f"{item.alpha:.4f} | {item.r_vs_exp:.4f} | {item.p_value:.4g} | {item.verdict}"
            )
    return results


def parse_train_log(path: Path) -> Tuple[float | None, float | None]:
    best_val_auc = None
    test_auc_at_best = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                test_auc_at_best = float(epoch_match.group(1))
            final_match = FINAL_RE.search(line)
            if final_match:
                best_val_auc = float(final_match.group(1))
    return best_val_auc, test_auc_at_best


def run_command(command: List[str], log_path: Path, env: Dict[str, str]) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}). See {log_path}")


def generate_attributes(args: argparse.Namespace, dataset: str, centrality: str, k: int) -> Tuple[Path, Path]:
    attr_dir = Path(args.sweep_attr_dir) / dataset / centrality / f"k{k}"
    ensure_dir(attr_dir)
    attr_file = attr_dir / "concat_all_top100.npy"
    meta_file = attr_dir / "meta_concat_all_top100.json"
    if attr_file.exists() and meta_file.exists() and not args.force:
        return attr_file, meta_file
    command = [
        sys.executable,
        "-m",
        "src.core.generate_attributes",
        "--dataset",
        dataset,
        "--data_root",
        args.data_root,
        "--output_dir",
        str(attr_dir),
        "--prefix",
        "concat_all",
        "--seed",
        str(args.seed),
        "--centrality_names",
        centrality,
        "--top_k_percent",
        "100",
        "--max_anchors",
        str(k),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    run_command(command, attr_dir / "generate.log", env)
    return attr_file, meta_file


def run_one_scout(args: argparse.Namespace, dataset: str, centrality: str, k: int, seed: int) -> Tuple[float, float]:
    attr_file, meta_file = generate_attributes(args, dataset, centrality, k)
    log_path = Path(args.sweep_result_dir) / dataset / centrality / f"k{k}" / f"seed{seed}.log"
    if log_path.exists() and not args.force:
        val_auc, test_auc = parse_train_log(log_path)
        if val_auc is not None and test_auc is not None:
            return val_auc, test_auc
    command = [
        sys.executable,
        "-m",
        "src.core.train_linkpred",
        "--dataset",
        dataset,
        "--data_root",
        args.data_root,
        "--attr_file",
        str(attr_file),
        "--meta_file",
        str(meta_file),
        "--model",
        args.model,
        "--decoder",
        args.decoder,
        "--hidden",
        str(args.hidden),
        "--layer",
        str(args.layer),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--val_every",
        str(args.val_every),
        "--neg_ratio",
        str(args.neg_ratio),
        "--mrr_neg_k",
        str(args.mrr_neg_k),
        "--seed",
        str(seed),
    ]
    if args.use_raw_feature:
        command.append("--use_raw_feature")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    run_command(command, log_path, env)
    val_auc, test_auc = parse_train_log(log_path)
    if val_auc is None or test_auc is None:
        raise RuntimeError(f"Could not parse AUC values from {log_path}")
    return val_auc, test_auc


def run_experiment_3(args: argparse.Namespace, curves: Dict[Tuple[str, str], CurveResult]) -> None:
    print("\n[Experiment 3] Elbow K vs task-optimal K")
    k_sweep_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)

    base_grid = list(args.k_range)
    for dataset in args.datasets:
        for centrality in args.centralities:
            elbow_k = curves[(dataset, centrality)].elbow_k
            grid = sorted(set(base_grid + [int(elbow_k)]))
            for k in grid:
                val_values = []
                test_values = []
                for run_idx in range(args.runs):
                    val_auc, test_auc = run_one_scout(args, dataset, centrality, k, run_idx)
                    val_values.append(val_auc)
                    test_values.append(test_auc)
                    k_sweep_rows.append(
                        {
                            "section": "ksweep_run",
                            "dataset": dataset,
                            "centrality": centrality,
                            "k": k,
                            "seed": run_idx,
                            "val_auc": val_auc,
                            "test_auc": test_auc,
                        }
                    )
                row = {
                    "dataset": dataset,
                    "centrality": centrality,
                    "k": k,
                    "mean_val_auc": mean(val_values),
                    "std_val_auc": stdev(val_values) if len(val_values) > 1 else 0.0,
                    "mean_test_auc": mean(test_values),
                    "std_test_auc": stdev(test_values) if len(test_values) > 1 else 0.0,
                }
                grouped[(dataset, centrality)].append(row)

    for key, rows in grouped.items():
        dataset, centrality = key
        rows = sorted(rows, key=lambda row: int(row["k"]))
        best = max(rows, key=lambda row: float(row["mean_test_auc"]))
        elbow_k = curves[(dataset, centrality)].elbow_k
        elbow_row = next(row for row in rows if int(row["k"]) == int(elbow_k))
        auc_gap = float(best["mean_test_auc"]) - float(elbow_row["mean_test_auc"])
        match = abs(int(elbow_k) - int(best["k"])) <= 2 and auc_gap < 0.01
        for row in rows:
            k_sweep_rows.append({"section": "ksweep_mean", **row})
        summary_rows.append(
            {
                "section": "ksweep_summary",
                "dataset": dataset,
                "centrality": centrality,
                "elbow_k": int(elbow_k),
                "optimal_k": int(best["k"]),
                "auc_at_elbow": float(elbow_row["mean_test_auc"]),
                "auc_at_optimal": float(best["mean_test_auc"]),
                "auc_gap": auc_gap,
                "match": "\u2713" if match else "x",
            }
        )

    save_table_csv(Path(args.output_csv), k_sweep_rows)
    save_table_csv(Path(args.output_csv), summary_rows)

    set_plot_style()
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), squeeze=False)
    summary_lookup = {(row["dataset"], row["centrality"]): row for row in summary_rows}
    for r, dataset in enumerate(args.datasets):
        for c, centrality in enumerate(args.centralities):
            ax = axes[r][c]
            rows = sorted(grouped[(dataset, centrality)], key=lambda row: int(row["k"]))
            xs = [int(row["k"]) for row in rows]
            ys = [float(row["mean_test_auc"]) for row in rows]
            summary = summary_lookup[(dataset, centrality)]
            elbow_k = int(summary["elbow_k"])
            optimal_k = int(summary["optimal_k"])
            opt_y = float(summary["auc_at_optimal"])
            ax.plot(xs, ys, color=BLUE, marker="o", markersize=3, linewidth=1.3)
            ax.scatter([optimal_k], [opt_y], color=BLUE, marker="*", s=80, zorder=5, label=f"optimal K={optimal_k}")
            ax.axvline(elbow_k, color=ORANGE, linestyle="--", linewidth=1.1, label=f"elbow K={elbow_k}")
            ax.set_title(f"{dataset} x {DISPLAY_CENTRALITY[centrality]} (E={elbow_k}, O={optimal_k})")
            ax.set_xlabel("K")
            ax.set_ylabel("Mean test AUC")
            ax.grid(True, linewidth=0.35, alpha=0.35)
            ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(Path(args.figures_dir) / f"fig_ksweep.{ext}", dpi=150)
    plt.close(fig)

    print("Dataset   | Centrality  | Elbow K | Optimal K | AUC Gap | Match?")
    print("----------|-------------|---------|-----------|---------|-------")
    for dataset in args.datasets:
        for centrality in args.centralities:
            row = summary_lookup[(dataset, centrality)]
            print(
                f"{dataset:<9} | {DISPLAY_CENTRALITY[centrality]:<11} | "
                f"{row['elbow_k']:<7} | {row['optimal_k']:<9} | {float(row['auc_gap']):.4f} | {row['match']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCOUT elbow-rule visualization experiments")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--centralities", nargs="+", default=CENTRALITIES, choices=CENTRALITIES)
    parser.add_argument("--data_root", default="./datasets")
    parser.add_argument("--figures_dir", default="./figures")
    parser.add_argument("--output_csv", default="./figures/results_elbow_analysis.csv")
    parser.add_argument("--skip_ksweep", action="store_true")
    parser.add_argument("--cuda_visible_devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--k_range", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12, 15, 20, 25, 30])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep_attr_dir", default="./attrs/elbow_k_sweep")
    parser.add_argument("--sweep_result_dir", default="./results/elbow_k_sweep")
    parser.add_argument("--model", default="gcn", choices=["gcn", "sage"])
    parser.add_argument("--decoder", default="mlp", choices=["mlp", "inner"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--neg_ratio", type=float, default=1.0)
    parser.add_argument("--mrr_neg_k", type=int, default=100)
    parser.add_argument("--use_raw_feature", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(Path(args.figures_dir))
    reset_csv(Path(args.output_csv))
    curves = run_experiment_1(args)
    run_experiment_2(args, curves)
    if not args.skip_ksweep:
        run_experiment_3(args, curves)
    print(f"\nSaved figures and CSV under {args.figures_dir}")


if __name__ == "__main__":
    main()
