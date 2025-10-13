# ============================================================
# SCOUT
# Structure-Aware Aspect and Anchor-Count Selection for
# Node Attribute Augmentation via Positional Information
#
# File: src/core/train_linkpred.py
# Description: Link prediction training script for SCOUT
# ============================================================

import os
import json
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling

# Internal imports
from src.utils.data_loader import load_dataset
from src.models.attr_gate import MeasureAttentionGateV3
from src.models.encoder import Encoder
from src.models.decoder import MLPDecoder, InnerProductDecoder


# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed: int = 1):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Training / Evaluation
# ============================================================

def train_epoch(enc, gate, dec, data, measure_blocks, opt, crit, args):
    """Perform one epoch of training."""
    enc.train()
    gate.train()
    dec.train()
    opt.zero_grad()

    # Gated attributes
    gated, _ = gate(measure_blocks)
    z = enc(data.edge_index.to(gated.device), gated)

    # Positive & negative edges
    pos = data.edge_label_index[:, data.edge_label == 1].to(z.device)
    num_pos = pos.size(1)
    num_neg = int(args.neg_ratio * num_pos)
    neg = negative_sampling(
        data.edge_index, num_nodes=data.num_nodes, num_neg_samples=num_neg
    ).to(z.device)

    # Edge labels
    ei = torch.cat([pos, neg], dim=1)
    lbl = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)]).to(z.device)

    # Forward & optimize
    logit = dec(z[ei[0]], z[ei[1]])
    loss = crit(logit, lbl)
    loss.backward()
    opt.step()

    return loss.item()


@torch.no_grad()
def evaluate(enc, gate, dec, data, measure_blocks, args):
    """Evaluate model performance (AUC, AP, MRR)."""
    enc.eval()
    gate.eval()
    dec.eval()

    gated, _ = gate(measure_blocks)
    z = enc(data.edge_index.to(gated.device), gated)

    pos = data.edge_label_index[:, data.edge_label == 1]
    neg = data.edge_label_index[:, data.edge_label == 0]

    pl = dec(z[pos[0]], z[pos[1]])
    nl = dec(z[neg[0]], z[neg[1]])

    logit = torch.cat([pl, nl])
    lbl = torch.cat([torch.ones(pl.size(0)), torch.zeros(nl.size(0))]).to(z.device)

    prob = torch.sigmoid(logit).cpu().numpy()
    labels = lbl.cpu().numpy()

    auc = roc_auc_score(labels, prob)
    ap = average_precision_score(labels, prob)

    # Compute MRR (Mean Reciprocal Rank)
    mrrs = []
    K = args.mrr_neg_k
    for i in range(pos.size(1)):
        u, v = pos[0, i].item(), pos[1, i].item()
        s_pos = pl[i].item()
        v_neg = torch.randint(0, data.num_nodes, (K,), device=z.device)
        u_rep = torch.full((K,), u, device=z.device)
        s_neg = dec(z[u_rep], z[v_neg]).cpu()
        rank = 0.5 * ((s_neg >= s_pos).sum().item() + (s_neg > s_pos).sum().item()) + 1.0
        mrrs.append(1.0 / rank)

    return auc, ap, float(np.mean(mrrs))


# ============================================================
# Main Training Pipeline
# ============================================================

def run(args):
    """Run link prediction training."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ------------------ Load dataset ------------------
    train, val, test, num_nodes = load_dataset(
        args.dataset, args.data_root, use_raw_feature=args.use_raw_feature
    )

    # ------------------ Load SCOUT attributes ------------------
    arr = np.load(args.attr_file)
    with open(args.meta_file) as f:
        meta = json.load(f)

    block_dims = meta["block_dims"]
    attr_gen = torch.tensor(arr, dtype=torch.float32).to(device)

    # Split into per-measure blocks
    measure_blocks, offset = [], 0
    for d in block_dims:
        measure_blocks.append(attr_gen[:, offset:offset + d])
        offset += d

    # ------------------ Initialize modules ------------------
    gate = MeasureAttentionGateV3(block_dims, att_dim=args.att_dim).to(device)
    enc = Encoder(args.model, sum(block_dims), args.hidden,
                  dropout=args.dropout, num_layers=args.layer).to(device)
    dec = MLPDecoder(args.hidden).to(device) if args.decoder == "mlp" else InnerProductDecoder().to(device)

    # ------------------ Optimization setup ------------------
    params = list(enc.parameters()) + list(dec.parameters()) + list(gate.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(args.neg_ratio), device=device))

    # ------------------ Training loop ------------------
    best_auc, best_ap, best_mrr = 0, 0, 0
    for ep in range(1, args.epochs + 1):
        _ = train_epoch(enc, gate, dec, train, measure_blocks, opt, crit, args)

        if ep % args.val_every == 0:
            val_auc, _, _ = evaluate(enc, gate, dec, val, measure_blocks, args)
            if val_auc > best_auc:
                best_auc = val_auc
                t_auc, t_ap, t_mrr = evaluate(enc, gate, dec, test, measure_blocks, args)
                best_ap, best_mrr = t_ap, t_mrr
                logging.info(f"Epoch {ep:04d}: AUC={t_auc:.4f}, AP={t_ap:.4f}, MRR={t_mrr:.4f}")

    logging.info(f"🏁 Final Best: AUC={best_auc:.4f}, AP={best_ap:.4f}, MRR={best_mrr:.4f}")


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SCOUT Link Prediction Trainer")
    # Dataset / Path
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_root", type=str, default="./datasets")
    p.add_argument("--attr_file", type=str, required=True)
    p.add_argument("--meta_file", type=str, required=True)

    # Model
    p.add_argument("--model", type=str, default="gcn", choices=["gcn", "sage"])
    p.add_argument("--decoder", type=str, default="mlp", choices=["mlp", "inner"])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--att_dim", type=int, default=128)
    p.add_argument("--layer", type=int, default=2)

    # Optimization
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--val_every", type=int, default=10)

    # Evaluation
    p.add_argument("--neg_ratio", type=float, default=1.0)
    p.add_argument("--mrr_neg_k", type=int, default=100)

    # Misc
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--use_raw_feature", action="store_true",
                   help="Use original node features (default: off)")

    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run(args)
