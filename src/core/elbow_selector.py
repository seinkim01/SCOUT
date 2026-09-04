"""
Utilities for selecting an anchor-count from centrality distributions.

SCOUT's README describes an elbow-based anchor-count detector grounded in
power-law centrality curves. This module implements a lightweight and
reproducible approximation using a Kneedle-style maximum-distance criterion on
the log-rank / log-centrality curve.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class ElbowResult:
    k: int
    percent: float
    elbow_index: int
    score_at_elbow: float
    max_distance: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "k": int(self.k),
            "percent": float(self.percent),
            "elbow_index": int(self.elbow_index),
            "score_at_elbow": float(self.score_at_elbow),
            "max_distance": float(self.max_distance),
        }


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = arr.min()
    hi = arr.max()
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def select_elbow_k(
    values: Iterable[float],
    *,
    min_k: int = 8,
    max_k: int | None = None,
    eps: float = 1e-12,
) -> ElbowResult:
    """
    Choose an elbow on a descending centrality curve.

    Args:
        values: Centrality scores for all nodes.
        min_k: Smallest allowed anchor count.
        max_k: Largest allowed anchor count.
        eps: Numerical stabilizer for log computations.
    """
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.ndim != 1:
        vals = vals.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Cannot select elbow from an empty score list.")

    vals = np.sort(np.clip(vals, a_min=0.0, a_max=None))[::-1]
    n = vals.size
    min_k = max(1, min(min_k, n))
    max_k = n if max_k is None else max(min_k, min(max_k, n))

    x = np.log(np.arange(1, n + 1, dtype=np.float64))
    y = np.log(vals + eps)
    x_norm = _normalize(x)
    y_norm = _normalize(y)

    # Straight line between endpoints of the normalized curve.
    y_line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    distance = y_line - y_norm

    candidate = np.arange(n)
    candidate = candidate[(candidate + 1 >= min_k) & (candidate + 1 <= max_k)]
    if candidate.size == 0:
        candidate = np.arange(n)

    best_idx = int(candidate[np.argmax(distance[candidate])])
    best_k = best_idx + 1
    return ElbowResult(
        k=best_k,
        percent=100.0 * best_k / n,
        elbow_index=best_idx,
        score_at_elbow=float(vals[best_idx]),
        max_distance=float(distance[best_idx]),
    )


def aggregate_anchor_percent(
    per_measure_results: Dict[str, ElbowResult],
    *,
    strategy: str = "mean",
) -> float:
    """
    Aggregate per-centrality elbow percentages into one graph-level percentage.
    """
    if not per_measure_results:
        raise ValueError("per_measure_results must not be empty.")

    percents = np.asarray(
        [res.percent for res in per_measure_results.values()],
        dtype=np.float64,
    )
    if strategy == "mean":
        return float(np.mean(percents))
    if strategy == "median":
        return float(np.median(percents))
    if strategy.startswith("measure:"):
        name = strategy.split(":", 1)[1]
        if name not in per_measure_results:
            raise KeyError(f"Unknown measure for aggregation: {name}")
        return float(per_measure_results[name].percent)
    raise ValueError(f"Unsupported aggregation strategy: {strategy}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="SCOUT elbow selector utility")
    parser.add_argument("--scores_json", type=str, required=True)
    parser.add_argument("--min_k", type=int, default=8)
    parser.add_argument("--max_k", type=int, default=None)
    parser.add_argument(
        "--aggregate_strategy",
        type=str,
        default="mean",
        help="mean | median | measure:<name>",
    )
    args = parser.parse_args()

    with open(args.scores_json, "r", encoding="utf-8") as f:
        score_map = json.load(f)

    results: Dict[str, ElbowResult] = {}
    for name, scores in score_map.items():
        results[name] = select_elbow_k(
            scores,
            min_k=args.min_k,
            max_k=args.max_k,
        )

    aggregate = aggregate_anchor_percent(results, strategy=args.aggregate_strategy)
    payload = {
        "aggregate_strategy": args.aggregate_strategy,
        "graph_level_top_k_percent": aggregate,
        "per_measure": {name: res.to_dict() for name, res in results.items()},
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    _main()
