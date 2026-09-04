# src/models/heads.py
"""
Task heads for SCOUT.

`MLPClassifier` maps node embeddings produced by `src.models.encoder.Encoder`
to class logits for node classification. Link prediction uses the decoders in
`src.models.decoder` instead.
"""

import torch.nn as nn


class MLPClassifier(nn.Module):
    """Two-layer MLP node-classification head on top of GNN embeddings."""

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z):
        return self.mlp(z)
