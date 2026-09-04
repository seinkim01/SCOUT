"""
C3 elbow-method analysis for SCOUT.

Produces centrality ranked curves, log-log curves, elbow metadata, and
power-law fit diagnostics for citation datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.core.elbow_selector import select_elbow_k
from src.core.generate_attributes import (
    CENTRALITIES,
    betweenness_auto,
    closeness_auto,
    eigenvector_centrality_power,
    load_train_adj,
    pagerank_cpu,
)


DISPLAY = {
    "pagerank": "PR",
    "betweenness": "BN",
    "closeness": "CL",
    "eigenvector": "EV",
}


@dataclass
class PowerLawStats:
    method: str
    n_positive: int
    alpha: float | None
    xmin: float | None
    loglikelihood_ratio: float | None
    p_value: float | None
    slope: float | None
    intercept: float | None
    r2: float | None
    note: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_centrality_map(args: argparse.Namespace, dataset: str) -> Dict[str, np.ndarray]:
    adj = load_train_adj(dataset, args.data_root, args.seed)
    out: Dict[str, np.ndarray] = {}
    for name in args.centrality_names:
        if name == "pagerank":
            out[name] = pagerank_cpu(adj, alpha=args.pr_alpha, tol=args.pr_tol, max_iter=args.pr_max_iter)
        elif name == "betweenness":
            out[name] = betweenness_auto(adj, k=args.btw_k, seed=args.seed)
        elif name == "closeness":
            out[name] = closeness_auto(adj, k=args.clo_k, seed=args.seed)
        elif name == "eigenvector":
            out[name] = eigenvector_centrality_power(adj, iters=args.eig_iters, tol=args.eig_tol)
        else:
            raise ValueError(f"Unsupported centrality: {name}")
    return out


def ranked(values: Iterable[float]) -> np.ndarray:
    vals = np.asarray(list(values), dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    vals = np.clip(vals, a_min=0.0, a_max=None)
    return np.sort(vals)[::-1]


def fit_powerlaw(values: np.ndarray) -> PowerLawStats:
    vals = ranked(values)
    pos = vals[vals > 0]
    if pos.size < 8:
        return PowerLawStats(
            method="none",
            n_positive=int(pos.size),
            alpha=None,
            xmin=None,
            loglikelihood_ratio=None,
            p_value=None,
            slope=None,
            intercept=None,
            r2=None,
            note="Too few positive centrality scores.",
        )

    try:
        import powerlaw  # type: ignore

        fit = powerlaw.Fit(pos, discrete=False, verbose=False)
        ratio, p_value = fit.distribution_compare("power_law", "lognormal")
        return PowerLawStats(
            method="powerlaw",
            n_positive=int(pos.size),
            alpha=float(fit.power_law.alpha),
            xmin=float(fit.power_law.xmin),
            loglikelihood_ratio=float(ratio),
            p_value=float(p_value),
            slope=None,
            intercept=None,
            r2=None,
            note="R>0 favors power-law over lognormal; p>=0.05 means the sign is not statistically decisive.",
        )
    except Exception as exc:
        ranks = np.arange(1, pos.size + 1, dtype=np.float64)
        x = np.log10(ranks)
        y = np.log10(pos + 1e-300)
        slope, intercept = np.polyfit(x, y, deg=1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return PowerLawStats(
            method="loglog_linear_fallback",
            n_positive=int(pos.size),
            alpha=None,
            xmin=None,
            loglikelihood_ratio=None,
            p_value=None,
            slope=float(slope),
            intercept=float(intercept),
            r2=float(r2),
            note=f"Install `powerlaw` for likelihood-ratio tests. Fallback reason: {type(exc).__name__}: {exc}",
        )


def plot_ranked_grid(
    score_maps: Dict[str, Dict[str, np.ndarray]],
    elbow_rows: List[Dict[str, object]],
    out_path: str,
    *,
    loglog: bool,
) -> None:
    datasets = list(score_maps.keys())
    centralities = list(next(iter(score_maps.values())).keys())
    elbow_lookup = {
        (str(row["dataset"]), str(row["centrality"])): int(row["elbow_k"])
        for row in elbow_rows
    }

    fig, axes = plt.subplots(
        len(datasets),
        len(centralities),
        figsize=(4.0 * len(centralities), 3.0 * len(datasets)),
        squeeze=False,
    )
    for r, dataset in enumerate(datasets):
        for c, cent in enumerate(centralities):
            ax = axes[r][c]
            vals = ranked(score_maps[dataset][cent])
            ranks = np.arange(1, vals.size + 1)
            if loglog:
                mask = vals > 0
                ax.loglog(ranks[mask], vals[mask], linewidth=1.5)
            else:
                ax.plot(ranks, vals, linewidth=1.5)

            elbow_k = elbow_lookup[(dataset, cent)]
            if 1 <= elbow_k <= vals.size:
                ax.scatter([elbow_k], [vals[elbow_k - 1]], marker="v", s=48, color="#d62728", zorder=5)
                ax.axvline(elbow_k, color="#d62728", linewidth=0.8, alpha=0.35)
                ax.text(
                    0.98,
                    0.92,
                    f"K={elbow_k}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                )

            ax.set_title(f"{dataset} / {DISPLAY.get(cent, cent)}", fontsize=10)
            if r == len(datasets) - 1:
                ax.set_xlabel("Rank")
            if c == 0:
                ax.set_ylabel("Centrality")
            ax.grid(True, linewidth=0.35, alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SCOUT C3 elbow-method analysis artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=["Cora", "Citeseer", "Pubmed"])
    parser.add_argument("--centrality_names", nargs="+", default=CENTRALITIES, choices=CENTRALITIES)
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--output_dir", type=str, default="./results/c3_elbow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_anchors", type=int, default=8)
    parser.add_argument("--max_anchors", type=int, default=None)
    parser.add_argument("--pr_alpha", type=float, default=0.85)
    parser.add_argument("--pr_tol", type=float, default=1e-6)
    parser.add_argument("--pr_max_iter", type=int, default=100)
    parser.add_argument("--clo_k", type=int, default=1024)
    parser.add_argument("--btw_k", type=int, default=1024)
    parser.add_argument("--eig_iters", type=int, default=100)
    parser.add_argument("--eig_tol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    score_maps: Dict[str, Dict[str, np.ndarray]] = {}
    rows: List[Dict[str, object]] = []

    for dataset in args.datasets:
        centralities = compute_centrality_map(args, dataset)
        score_maps[dataset] = centralities
        np.savez_compressed(
            os.path.join(args.output_dir, f"centrality_scores_{dataset}.npz"),
            **centralities,
        )

        for cent_name, values in centralities.items():
            vals = ranked(values)
            elbow = select_elbow_k(vals, min_k=args.min_anchors, max_k=args.max_anchors)
            stats = fit_powerlaw(vals)
            rows.append(
                {
                    "dataset": dataset,
                    "centrality": cent_name,
                    "n_nodes": int(vals.size),
                    "elbow_k": int(elbow.k),
                    "elbow_percent": float(elbow.percent),
                    "score_at_elbow": float(elbow.score_at_elbow),
                    "max_distance": float(elbow.max_distance),
                    **asdict(stats),
                }
            )

    plot_ranked_grid(
        score_maps,
        rows,
        os.path.join(args.output_dir, "centrality_ranked_curves_12panel.png"),
        loglog=False,
    )
    plot_ranked_grid(
        score_maps,
        rows,
        os.path.join(args.output_dir, "centrality_loglog_curves_12panel.png"),
        loglog=True,
    )
    write_csv(os.path.join(args.output_dir, "c3_elbow_powerlaw_summary.csv"), rows)
    with open(os.path.join(args.output_dir, "c3_elbow_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output_dir": args.output_dir, "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
