"""
Leakage-safe SCOUT positional attribute generator.

- Builds adjacency from the train split only.
- Centralities: pagerank, betweenness, closeness, eigenvector
- Similarities: RWR, AdaSim, RA, Jaccard
- Saves per-centrality/per-similarity blocks and one concatenated file
- Emits SCOUT-style meta JSON with 16 blocks:
  centrality x sim{1,2,3,4}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from ogb.linkproppred import PygLinkPropPredDataset
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected

from src.core.elbow_selector import aggregate_anchor_percent, select_elbow_k


CENTRALITIES = ["betweenness", "closeness", "eigenvector", "pagerank"]
SIMILARITIES = ["rwr", "adasim", "ra", "jaccard"]
SIM_TAG = {"rwr": "sim1", "adasim": "sim2", "ra": "sim3", "jaccard": "sim4"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_top_percent(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _save_array(arr: np.ndarray, out_path: str, save_fp16: bool) -> None:
    arr = arr.astype(np.float16 if save_fp16 else np.float32, copy=False)
    np.save(out_path, arr)
    logging.info(
        "[SAVE] %s | dtype=%s | shape=%s | min=%.4g | max=%.4g",
        out_path,
        arr.dtype,
        arr.shape,
        float(arr.min()) if arr.size else 0.0,
        float(arr.max()) if arr.size else 0.0,
    )


def check_numeric(name: str, x: np.ndarray) -> None:
    if not np.isfinite(x).all():
        raise ValueError(f"[{name}] contains non-finite values")


def to_undirected_no_self(a: sp.csr_matrix) -> sp.csr_matrix:
    a = a.tocsr()
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def load_train_adj_ogb(dataset: str, root: str) -> sp.csr_matrix:
    dset = PygLinkPropPredDataset(name=dataset, root=root)
    split_edge = dset.get_edge_split()
    data = dset[0]
    num_nodes = int(data.num_nodes)

    train_e = np.asarray(split_edge["train"]["edge"], dtype=np.int64).T
    undirected = {"ogbl-collab", "ogbl-ppa", "ogbl-ddi"}
    if dataset in undirected:
        te = to_undirected(torch.from_numpy(train_e), num_nodes=num_nodes)
        train_e = te.cpu().numpy()

    a = sp.csr_matrix(
        (np.ones(train_e.shape[1], dtype=np.float32), (train_e[0], train_e[1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    a = to_undirected_no_self(a)
    logging.info("[LOAD-TRAIN] OGB %s | N=%d | undirected train-edges=%d", dataset, num_nodes, a.nnz // 2)
    return a


def load_train_adj_pyg(dataset: str, root: str, split_seed: int) -> sp.csr_matrix:
    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        dset = Planetoid(root=root, name=dataset)
    elif dataset == "CoauthorCS":
        dset = Coauthor(root=os.path.join(root, "CoauthorCS"), name="CS")
    elif dataset == "AmazonComputers":
        dset = Amazon(root=os.path.join(root, "AmazonComputers"), name="Computers")
    elif dataset == "AmazonPhoto":
        dset = Amazon(root=os.path.join(root, "AmazonPhoto"), name="Photo")
    else:
        raise ValueError(f"Unsupported PyG dataset: {dataset}")

    data = dset[0]
    torch.manual_seed(split_seed)
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        split_labels=True,
    )
    train_data, _, _ = transform(data)
    num_nodes = int(train_data.num_nodes)
    edge_index = train_data.edge_index.cpu().numpy()
    a = sp.csr_matrix(
        (np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    a = to_undirected_no_self(a)
    logging.info("[LOAD-TRAIN] PyG %s | N=%d | undirected train-edges=%d", dataset, num_nodes, a.nnz // 2)
    return a


def load_train_adj(dataset: str, root: str, split_seed: int) -> sp.csr_matrix:
    if dataset.startswith("ogbl-"):
        return load_train_adj_ogb(dataset, root)
    return load_train_adj_pyg(dataset, root, split_seed)


def _normalize_rows_csr(a: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(a.sum(axis=1)).ravel()
    d[d == 0] = 1.0
    return a.multiply(1.0 / d[:, None])


def pagerank_cpu(a: sp.csr_matrix, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    n = a.shape[0]
    d = np.asarray(a.sum(axis=1)).ravel()
    d[d == 0] = 1.0
    p = a.T.multiply(1.0 / d)
    pr = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - alpha) / n
    for _ in range(max_iter):
        prev = pr
        pr = alpha * (p @ prev) + teleport
        if np.abs(pr - prev).sum() < tol:
            break
    return pr


def eigenvector_centrality_power(a: sp.csr_matrix, iters: int = 100, tol: float = 1e-6) -> np.ndarray:
    x = np.ones(a.shape[0], dtype=np.float64)
    x /= np.linalg.norm(x, 2)
    for _ in range(iters):
        x_new = a @ x
        norm = np.linalg.norm(x_new, 2)
        if norm == 0:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new
            break
        x = x_new
    x /= (x.max() + 1e-12)
    return x


def closeness_auto(a: sp.csr_matrix, k: int = 1024, seed: int = 42) -> np.ndarray:
    n = a.shape[0]
    indptr, indices = a.indptr, a.indices

    def bfs_from(src: int) -> np.ndarray:
        dist = np.full(n, -1, dtype=np.int32)
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in indices[indptr[u]:indptr[u + 1]]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    rng = np.random.default_rng(seed)
    srcs = rng.choice(n, size=min(k, n), replace=False)
    dist_sum = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    for s in srcs:
        d = bfs_from(int(s))
        m = d >= 0
        dist_sum[m] += d[m]
        count[m] += 1
    avg = np.zeros(n, dtype=np.float64)
    np.divide(dist_sum, np.maximum(count, 1), out=avg, where=count > 0)
    clos = np.zeros(n, dtype=np.float64)
    m = avg > 0
    clos[m] = 1.0 / avg[m]
    if clos.max() > 0:
        clos /= clos.max()
    return clos


def betweenness_auto(a: sp.csr_matrix, k: int = 1024, seed: int = 42) -> np.ndarray:
    import networkx as nx

    g = nx.from_scipy_sparse_array(a, create_using=nx.Graph)
    rng = np.random.default_rng(seed)
    nodes = list(g.nodes())
    src = rng.choice(nodes, size=min(k, len(nodes)), replace=False)
    btw = nx.betweenness_centrality_subset(g, sources=src, targets=nodes, normalized=True)
    return np.asarray([btw[i] for i in range(a.shape[0])], dtype=np.float64)


def compute_centralities(args: argparse.Namespace, a: sp.csr_matrix) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in args.centrality_names:
        if name == "pagerank":
            out[name] = pagerank_cpu(a, alpha=args.pr_alpha, tol=args.pr_tol, max_iter=args.pr_max_iter)
        elif name == "betweenness":
            out[name] = betweenness_auto(a, k=args.btw_k, seed=args.seed)
        elif name == "closeness":
            out[name] = closeness_auto(a, k=args.clo_k, seed=args.seed)
        elif name == "eigenvector":
            out[name] = eigenvector_centrality_power(a, iters=args.eig_iters, tol=args.eig_tol)
        else:
            raise ValueError(f"Unsupported centrality: {name}")
    return out


def select_anchor_sets(
    centralities: Dict[str, np.ndarray],
    *,
    top_k_percent: float | None,
    min_anchors: int,
    max_anchors: int | None,
    aggregate_strategy: str,
) -> Tuple[float, Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    elbow_meta: Dict[str, Dict[str, float]] = {}
    if top_k_percent is None:
        elbow_results = {
            name: select_elbow_k(values, min_k=min_anchors, max_k=max_anchors)
            for name, values in centralities.items()
        }
        top_k_percent = aggregate_anchor_percent(elbow_results, strategy=aggregate_strategy)
        elbow_meta = {name: res.to_dict() for name, res in elbow_results.items()}

    anchors: Dict[str, np.ndarray] = {}
    for name, values in centralities.items():
        k = max(1, int(round(len(values) * float(top_k_percent) / 100.0)))
        k = max(min_anchors, k)
        if max_anchors is not None:
            k = min(max_anchors, k)
        order = np.argsort(values)[::-1]
        anchors[name] = np.asarray(order[:k], dtype=np.int64)
    return float(top_k_percent), anchors, elbow_meta


def rwr_anchor(a: sp.csr_matrix, anchors: np.ndarray, restart: float, tol: float, max_iter: int, batch: int) -> np.ndarray:
    n = a.shape[0]
    p = _normalize_rows_csr(a)
    pt = p.T.tocsr()
    out = np.zeros((n, len(anchors)), dtype=np.float32)
    for s in range(0, len(anchors), batch):
        ba = anchors[s:s + batch]
        q = sp.csr_matrix(
            (np.ones(len(ba), dtype=np.float32), (ba, np.arange(len(ba)))),
            shape=(n, len(ba)),
        )
        q_dense = q.toarray().astype(np.float32)
        x = q_dense.copy()
        for _ in range(max_iter):
            prev = x
            x = restart * q_dense + (1.0 - restart) * (pt @ x)
            if np.abs(x - prev).sum() < tol:
                break
        col_sum = np.abs(x).sum(axis=0, keepdims=True) + 1e-8
        x = x / col_sum
        out[:, s:s + len(ba)] = x
        for j, anchor in enumerate(ba):
            out[int(anchor), s + j] = 1.0
    return out


def adasim_anchor(a: sp.csr_matrix, anchors: np.ndarray, depth: int) -> np.ndarray:
    n = a.shape[0]
    out = np.zeros((n, len(anchors)), dtype=np.float32)
    neighs = [a[i].indices for i in range(n)]
    deg = np.asarray(a.sum(axis=1)).ravel()
    deg = np.maximum(deg, 1.0)
    inv_log_deg = 1.0 / np.log1p(deg)

    for j, anchor in enumerate(anchors):
        scores = np.zeros(n, dtype=np.float32)
        frontier = {int(anchor)}
        visited = {int(anchor)}
        for hop in range(1, depth + 1):
            next_frontier = set()
            for u in frontier:
                weight_u = inv_log_deg[u] / hop
                for v in neighs[u]:
                    if v not in visited:
                        scores[v] += weight_u * inv_log_deg[v]
                        next_frontier.add(int(v))
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        # Graph-level clipping + z-score normalization.
        if np.any(scores):
            nz = scores[scores > 0]
            clip_hi = np.percentile(nz, 99.0)
            scores = np.clip(scores, 0.0, clip_hi)
            mean = scores.mean()
            std = scores.std()
            if std > 0:
                scores = (scores - mean) / std
            scores = np.maximum(scores, 0.0)
        out[:, j] = scores

    return out


def jaccard_anchor(a: sp.csr_matrix, anchors: np.ndarray, batch: int) -> np.ndarray:
    b = (a > 0).astype(np.float32).tocsr()
    deg = np.asarray(b.sum(axis=1)).ravel()
    out = np.zeros((a.shape[0], len(anchors)), dtype=np.float32)
    for s in range(0, len(anchors), batch):
        ba = anchors[s:s + batch]
        cn = (b @ b[ba, :].T).toarray()
        denom = deg[:, None] + deg[ba][None, :] - cn
        out[:, s:s + len(ba)] = cn / (denom + 1e-8)
    return out


def ra_anchor(a: sp.csr_matrix, anchors: np.ndarray, batch: int) -> np.ndarray:
    b = (a > 0).astype(np.float32).tocsr()
    deg = np.asarray(b.sum(axis=1)).ravel()
    deg = np.maximum(deg, 1.0)
    d_inv = sp.diags(1.0 / deg)
    out = np.zeros((a.shape[0], len(anchors)), dtype=np.float32)
    for s in range(0, len(anchors), batch):
        ba = anchors[s:s + batch]
        block = (b @ (d_inv @ b[ba, :].T)).toarray().astype(np.float32)
        col_min = block.min(axis=0, keepdims=True)
        col_max = block.max(axis=0, keepdims=True)
        denom = np.maximum(col_max - col_min, 1e-8)
        block = (block - col_min) / denom
        out[:, s:s + len(ba)] = block
    return out


MEASURE_FNS = {
    "rwr": lambda a, anchors, args: rwr_anchor(
        a, anchors, restart=args.rwr_restart, tol=args.rwr_tol, max_iter=args.rwr_max_iter, batch=args.batch
    ),
    "adasim": lambda a, anchors, args: adasim_anchor(a, anchors, depth=args.adasim_depth),
    "ra": lambda a, anchors, args: ra_anchor(a, anchors, batch=args.batch),
    "jaccard": lambda a, anchors, args: jaccard_anchor(a, anchors, batch=args.batch),
}


def build_attributes(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    a = load_train_adj(args.dataset, args.data_root, args.seed)
    ensure_dir(args.output_dir)

    centralities = compute_centralities(args, a)
    top_k_percent, anchors_by_centrality, elbow_meta = select_anchor_sets(
        centralities,
        top_k_percent=args.top_k_percent,
        min_anchors=args.min_anchors,
        max_anchors=args.max_anchors,
        aggregate_strategy=args.aggregate_strategy,
    )

    blocks: List[np.ndarray] = []
    measure_names: List[str] = []
    block_dims: List[int] = []
    anchors_per_measure: List[List[int]] = []
    files: Dict[str, str] = {}
    measure_times: Dict[str, float] = {}
    percent_tag = format_top_percent(top_k_percent)

    total_start = time.time()
    for centrality_name in args.centrality_names:
        anchors = anchors_by_centrality[centrality_name]
        for sim_name in args.similarity_names:
            key = f"{centrality_name}_{sim_name}"
            start = time.time()
            block = MEASURE_FNS[sim_name](a, anchors, args)
            measure_times[key] = time.time() - start
            check_numeric(key, block)

            block_tag = f"{centrality_name}_{sim_name}_top{percent_tag}.npy"
            block_path = os.path.join(args.output_dir, block_tag)
            _save_array(block, block_path, save_fp16=args.save_fp16)
            files[key] = block_path

            blocks.append(block.astype(np.float32, copy=False))
            measure_names.append(f"{centrality_name}_{SIM_TAG[sim_name]}")
            block_dims.append(int(block.shape[1]))
            anchors_per_measure.append(anchors.tolist())

    attr = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    check_numeric("attr_all", attr)

    attr_file = os.path.join(args.output_dir, f"{args.prefix}_top{percent_tag}.npy")
    meta_file = os.path.join(args.output_dir, f"meta_{args.prefix}_top{percent_tag}.json")
    _save_array(attr, attr_file, save_fp16=args.save_fp16)
    files["attr_all"] = attr_file

    total_time = time.time() - total_start
    meta = {
        "dataset": args.dataset,
        "used_graph": "train_only",
        "split_seed": int(args.seed),
        "n_nodes": int(a.shape[0]),
        "n_edges_undirected": int(a.nnz // 2),
        "top_k_percent": float(top_k_percent),
        "measure_names": measure_names,
        "block_dims": block_dims,
        "anchors_per_measure": anchors_per_measure,
        "centrality_names": list(args.centrality_names),
        "similarity_names": list(args.similarity_names),
        "params": {
            "batch": int(args.batch),
            "rwr_restart": float(args.rwr_restart),
            "rwr_tol": float(args.rwr_tol),
            "rwr_max_iter": int(args.rwr_max_iter),
            "adasim_depth": int(args.adasim_depth),
            "pr_alpha": float(args.pr_alpha),
            "pr_tol": float(args.pr_tol),
            "pr_max_iter": int(args.pr_max_iter),
            "clo_k": int(args.clo_k),
            "btw_k": int(args.btw_k),
            "eig_iters": int(args.eig_iters),
            "eig_tol": float(args.eig_tol),
            "aggregate_strategy": args.aggregate_strategy,
            "min_anchors": int(args.min_anchors),
            "max_anchors": None if args.max_anchors is None else int(args.max_anchors),
        },
        "centrality_anchor_meta": {
            name: {
                "n_anchors": int(len(anchors_by_centrality[name])),
                "anchors": anchors_by_centrality[name].tolist(),
            }
            for name in args.centrality_names
        },
        "elbow_meta": elbow_meta,
        "files": files,
        "times": measure_times,
        "total_time": total_time,
        "note": "Generated for SCOUT (train-only graph, 16-block aspect matrix).",
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logging.info("[SAVE] meta -> %s", meta_file)

    return {
        "attr_file": attr_file,
        "meta_file": meta_file,
        "top_k_percent": float(top_k_percent),
        "shape": list(attr.shape),
        "measure_names": measure_names,
        "block_dims": block_dims,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe SCOUT attribute generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="concat_all")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--centrality_names", nargs="+", default=CENTRALITIES, choices=CENTRALITIES)
    parser.add_argument("--similarity_names", nargs="+", default=SIMILARITIES, choices=SIMILARITIES)

    parser.add_argument("--top_k_percent", type=float, default=None)
    parser.add_argument("--aggregate_strategy", type=str, default="mean")
    parser.add_argument("--min_anchors", type=int, default=8)
    parser.add_argument("--max_anchors", type=int, default=None)

    parser.add_argument("--pr_alpha", type=float, default=0.85)
    parser.add_argument("--pr_tol", type=float, default=1e-6)
    parser.add_argument("--pr_max_iter", type=int, default=100)
    parser.add_argument("--clo_k", type=int, default=1024)
    parser.add_argument("--btw_k", type=int, default=1024)
    parser.add_argument("--eig_iters", type=int, default=100)
    parser.add_argument("--eig_tol", type=float, default=1e-6)

    parser.add_argument("--rwr_restart", type=float, default=0.15)
    parser.add_argument("--rwr_tol", type=float, default=1e-6)
    parser.add_argument("--rwr_max_iter", type=int, default=50)
    parser.add_argument("--adasim_depth", type=int, default=3)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--save_fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    payload = build_attributes(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
