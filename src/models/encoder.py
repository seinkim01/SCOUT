# src/models/encoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv

class Encoder(nn.Module):
    """
    GNN encoder supporting GCN and GraphSAGE backbones.
    """
    def __init__(self, model, in_dim, hidden, dropout=0.5, num_layers=2):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        if model == "gcn":
            self.layers = nn.ModuleList([GCNConv(in_dim, hidden)])
            for _ in range(num_layers - 1):
                self.layers.append(GCNConv(hidden, hidden))
        elif model == "sage":
            self.layers = nn.ModuleList([SAGEConv(in_dim, hidden)])
            for _ in range(num_layers - 1):
                self.layers.append(SAGEConv(hidden, hidden))
        else:
            raise ValueError(f"Unsupported model: {model}")

    def forward(self, edge_index, x):
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(h, edge_index)
            if i < self.num_layers - 1:
                h = h.relu()
                h = self.dropout(h)
        return h
