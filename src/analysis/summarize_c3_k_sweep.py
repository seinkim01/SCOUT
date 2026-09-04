"""
Summarize SCOUT C3 K-sweep logs into elbow-vs-optimal-K tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


FINAL_RE = re.compile(r"Final Best: AUC=([0-9.]+), AP=([0-9.]+), MRR=([0-9.]+)")


def read_elbows(path: str) -> Dict[Tuple[str, str], int]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {
        (str(row["dataset"]), str(row["centrality"])): int(row["elbow_k"])
        for row in rows
    }


def parse_log(path: str) -> Tuple[float | None, float | None, float | None]:
    auc = ap = mrr = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = FINAL_RE.search(line)
            if match:
                auc = float(match.group(1))
                ap = float(match.group(2))
                mrr = float(match.group(3))
    return auc, ap, mrr


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize K-sweep logs for SCOUT C3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep_dir", type=str, default="./results/c3_k_sweep")
    parser.add_argument("--elbow_summary", type=str, default="./results/c3_elbow/c3_elbow_summary.json")
    parser.add_argument("--output_csv", type=str, default="./results/c3_k_sweep/c3_elbow_vs_optimal_k.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    elbows = read_elbows(args.elbow_summary)
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)

    for root, _, files in os.walk(args.sweep_dir):
        for name in files:
            if not name.endswith(".log"):
                continue
            path = os.path.join(root, name)
            rel_parts = os.path.relpath(path, args.sweep_dir).split(os.sep)
            if len(rel_parts) < 4:
                continue
            dataset, centrality, k_part = rel_parts[:3]
            if not k_part.startswith("k"):
                continue
            try:
                k = int(k_part[1:])
            except ValueError:
                continue
            auc, ap, mrr = parse_log(path)
            if auc is None:
                continue
            grouped[(dataset, centrality)].append(
                {"dataset": dataset, "centrality": centrality, "k": k, "auc": auc, "ap": ap, "mrr": mrr}
            )

    detail_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: int(row["k"]))
        detail_rows.extend(rows)
        best = max(rows, key=lambda row: float(row["auc"]))
        dataset, centrality = key
        elbow_k = elbows.get(key)
        elbow_row = next((row for row in rows if int(row["k"]) == elbow_k), None)
        elbow_auc = float(elbow_row["auc"]) if elbow_row else None
        best_auc = float(best["auc"])
        summary_rows.append(
            {
                "dataset": dataset,
                "centrality": centrality,
                "elbow_k": elbow_k,
                "optimal_k": int(best["k"]),
                "elbow_auc": elbow_auc,
                "optimal_auc": best_auc,
                "auc_gap": None if elbow_auc is None else best_auc - elbow_auc,
                "n_finished_runs": len(rows),
            }
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    write_csv(args.output_csv, summary_rows)
    detail_csv = os.path.join(os.path.dirname(args.output_csv), "c3_k_sweep_all_runs.csv")
    write_csv(detail_csv, detail_rows)
    print(json.dumps({"summary_csv": args.output_csv, "detail_csv": detail_csv, "n_groups": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
