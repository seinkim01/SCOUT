# src/utils/data_loader.py
"""
Dataset loaders for SCOUT.

Two entry points, one per downstream task:

* ``load_dataset``                    -> link prediction  (RandomLinkSplit / OGB edge split)
* ``load_node_classification_dataset`` -> node classification (train/val/test node masks)

Both accept ``use_raw_feature``: when ``False`` the original node features are
replaced with an all-ones column so the model relies purely on the SCOUT
positional attributes.

Supported: Planetoid (Cora, Citeseer, Pubmed), Amazon (Computers, Photo),
Coauthor CS, and OGB link-prediction datasets (``ogbl-*``).
"""

import os

import torch
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset

PLANETOID = ("Cora", "Citeseer", "Pubmed")


def _load_raw(name: str, root: str):
    """Return the raw single-graph ``Data`` object for a supported dataset.

    The Planetoid ``root`` is passed straight through (PyG appends ``<name>/``
    itself), so ``--data_root ./datasets`` yields ``datasets/<Name>/`` for both
    this loader and ``src.core.generate_attributes``.
    """
    if name in PLANETOID:
        return Planetoid(root=root, name=name)[0]
    if name == "CoauthorCS":
        return Coauthor(root=os.path.join(root, "CoauthorCS"), name="CS")[0]
    if name == "AmazonComputers":
        return Amazon(root=os.path.join(root, "AmazonComputers"), name="Computers")[0]
    if name == "AmazonPhoto":
        return Amazon(root=os.path.join(root, "AmazonPhoto"), name="Photo")[0]
    raise ValueError(f"Unsupported dataset: {name}")


def _maybe_replace_features(data, use_raw_feature: bool):
    if not use_raw_feature or getattr(data, "x", None) is None:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
    return data


# ============================================================
# Link prediction
# ============================================================

def load_dataset(name: str, root: str, use_raw_feature: bool = False):
    """Load a dataset for link prediction and return ``(train, val, test, num_nodes)``."""
    # ---------------- Planetoid / Amazon / Coauthor ----------------
    if name in PLANETOID or name in ("CoauthorCS", "AmazonComputers", "AmazonPhoto"):
        data = _maybe_replace_features(_load_raw(name, root), use_raw_feature)
        transform = RandomLinkSplit(
            is_undirected=True,
            add_negative_train_samples=False,
            num_val=0.05,
            num_test=0.1,
        )
        train, val, test = transform(data)
        return train, val, test, data.num_nodes

    # ---------------- OGB link prediction ----------------
    if name.startswith("ogbl-"):
        dataset = PygLinkPropPredDataset(name=name, root=root)
        split_edge = dataset.get_edge_split()
        data = dataset[0]

        if data.x is None or not use_raw_feature:
            data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)

        def make_split(pos_edge, neg_edge):
            d = data.clone()
            if pos_edge.dim() == 2 and pos_edge.size(1) == 2:
                pos_edge = pos_edge.t()
            if neg_edge.dim() == 2 and neg_edge.size(1) == 2:
                neg_edge = neg_edge.t()
            d.edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
            d.edge_label = torch.cat([
                torch.ones(pos_edge.size(1)),
                torch.zeros(neg_edge.size(1)),
            ])
            return d

        if "edge_neg" in split_edge["train"]:
            train_neg = split_edge["train"]["edge_neg"]
        else:
            train_neg = negative_sampling(
                data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=split_edge["train"]["edge"].size(0),
            )

        train_data = make_split(split_edge["train"]["edge"], train_neg)
        val_data = make_split(split_edge["valid"]["edge"], split_edge["valid"]["edge_neg"])
        test_data = make_split(split_edge["test"]["edge"], split_edge["test"]["edge_neg"])
        return train_data, val_data, test_data, data.num_nodes

    raise ValueError(f"Unsupported dataset: {name}")


# ============================================================
# Node classification
# ============================================================

def _per_class_split(y: torch.Tensor, train_per_class: int = 20,
                     val_per_class: int = 30, seed: int = 1):
    """Planetoid-style deterministic split for datasets without built-in masks."""
    g = torch.Generator().manual_seed(seed)
    num_nodes = y.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in y.unique():
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0), generator=g)]
        train_mask[idx[:train_per_class]] = True
        val_mask[idx[train_per_class:train_per_class + val_per_class]] = True
        test_mask[idx[train_per_class + val_per_class:]] = True
    return train_mask, val_mask, test_mask


def load_node_classification_dataset(name: str, root: str,
                                     use_raw_feature: bool = False, seed: int = 1):
    """Load a dataset for node classification.

    Returns a single PyG ``Data`` object carrying ``x``, ``y``, ``edge_index``
    and boolean ``train_mask`` / ``val_mask`` / ``test_mask``.
    """
    data = _maybe_replace_features(_load_raw(name, root), use_raw_feature)

    if not hasattr(data, "train_mask") or data.train_mask is None:
        data.train_mask, data.val_mask, data.test_mask = _per_class_split(data.y, seed=seed)

    # Planetoid masks can be 2-D (public split matrix); keep the first column.
    for m in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, m)
        if mask.dim() > 1:
            setattr(data, m, mask[:, 0])

    data.num_classes = int(data.y.max().item()) + 1
    return data
