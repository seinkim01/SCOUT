# src/models/attr_gate.py
"""
MeasureAttentionGateV3: Attribute block attention gate for SCOUT
Supports measure-wise or element-wise gating and node-shared or node-specific weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeasureAttentionGateV3(nn.Module):
    def __init__(self, block_dims, att_dim=128, temperature=1.0,
                 gate_type="measure", weight_mode="same"):
        """
        Args:
            block_dims (list[int]): Dimensionality of each positional aspect block
            att_dim (int): Attention projection dimension
            temperature (float): Softmax temperature
            gate_type (str): "measure" or "element"
            weight_mode (str): "same" (shared) or "different" (node-specific)
        """
        super().__init__()
        self.block_dims = block_dims
        self.att_dim = att_dim
        self.temperature = temperature
        self.gate_type = gate_type
        self.weight_mode = weight_mode

        self.query = nn.Parameter(torch.randn(1, att_dim))
        self.proj = nn.ModuleList([nn.Linear(d, att_dim, bias=False) for d in block_dims])
        for p in self.proj:
            nn.init.xavier_uniform_(p.weight)

    def forward(self, measure_blocks):
        scores = []

        if self.weight_mode == "same":
            # === Graph-level attention (shared weights) ===
            for block, proj in zip(measure_blocks, self.proj):
                if self.gate_type == "measure":
                    k = proj(block.mean(dim=0, keepdim=True))
                else:
                    k = proj(block).mean(dim=0, keepdim=True)
                s = (self.query @ k.T) / self.temperature
                scores.append(s.squeeze())
            scores = torch.stack(scores)
            att = torch.softmax(scores, dim=0)
            out = torch.cat([block * att[i] for i, block in enumerate(measure_blocks)], dim=1)
            return out, att

        elif self.weight_mode == "different":
            # === Node-level attention (node-specific weights) ===
            node_scores = []
            for block, proj in zip(measure_blocks, self.proj):
                k = proj(block)
                s = (self.query @ k.T) / self.temperature
                node_scores.append(s.squeeze(0))
            node_scores = torch.stack(node_scores, dim=1)
            att = torch.softmax(node_scores, dim=1)

            out_blocks = []
            for i, block in enumerate(measure_blocks):
                out_blocks.append(block * att[:, i].unsqueeze(-1))
            out = torch.cat(out_blocks, dim=1)
            return out, att
