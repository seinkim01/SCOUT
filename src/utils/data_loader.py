# src/utils/data_loader.py
"""
Dataset Loader for SCOUT
Supports Planetoid (Cora, Citeseer, Pubmed), Amazon, Coauthor, and OGB datasets.
"""

import os
import torch
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset


def load_dataset(name: str, root: str, use_raw_feature: bool = False):
    """Load dataset and return (train, val, test, num_nodes)."""
    # ---------------- Planetoid ----------------
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=os.path.join(root, name), name=name)
        data = dataset[0]
        if not use_raw_feature:
            data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)

        transform = RandomLinkSplit(
            is_undirected=True,
            add_negative_train_samples=False,
            num_val=0.05,
            num_test=0.1
        )
        train, val, test = transform(data)
        return train, val, test, data.num_nodes

    # ---------------- Coauthor / Amazon ----------------
    elif name == "CoauthorCS":
        dataset = Coauthor(root=os.path.join(root, "CoauthorCS"), name="CS")
        data = dataset[0]
    elif name == "AmazonComputers":
        dataset = Amazon(root=os.path.join(root, "AmazonComputers"), name="Computers")
        data = dataset[0]
    elif name == "AmazonPhoto":
        dataset = Amazon(root=os.path.join(root, "AmazonPhoto"), name="Photo")
        data = dataset[0]

    # ---------------- OGB Link Prediction ----------------
    elif name.startswith("ogbl-"):
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
                torch.zeros(neg_edge.size(1))
            ])
            return d

        if 'edge_neg' in split_edge['train']:
            train_neg = split_edge['train']['edge_neg']
        else:
            num_nodes = data.num_nodes
            num_pos = split_edge['train']['edge'].size(0)
            train_neg = negative_sampling(
                data.edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_pos
            )

        train_data = make_split(split_edge['train']['edge'], train_neg)
        val_data = make_split(split_edge['valid']['edge'], split_edge['valid']['edge_neg'])
        test_data = make_split(split_edge['test']['edge'], split_edge['test']['edge_neg'])
        return train_data, val_data, test_data, data.num_nodes

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # ---------------- Coauthor/Amazon split ----------------
    if not use_raw_feature:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)

    transform = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=False,
        num_val=0.05, num_test=0.1
    )
    train, val, test = transform(data)
    return train, val, test, data.num_nodes
