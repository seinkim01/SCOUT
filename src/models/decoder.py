# src/models/decoder.py
import torch
import torch.nn as nn

class InnerProductDecoder(nn.Module):
    """Standard inner-product link predictor."""
    def forward(self, x_i, x_j):
        return (x_i * x_j).sum(dim=1)


class MLPDecoder(nn.Module):
    """MLP-based decoder for link prediction."""
    def __init__(self, hidden_dim, num_layers=3, dropout=0.5):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_i, x_j):
        x = x_i * x_j
        return self.mlp(x).view(-1)
