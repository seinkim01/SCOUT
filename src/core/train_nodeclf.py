# ============================================================
# SCOUT
# Structure-Aware Aspect and Anchor-Count Selection for
# Node Attribute Augmentation via Positional Information
#
# File: src/core/train_nodeclf.py
# Description: Node classification training script for SCOUT
#
# Mirrors src/core/train_linkpred.py: the SCOUT positional attributes are
# gated by MeasureAttentionGateV3, encoded by a GCN/GraphSAGE backbone, and
# read out by a 2-layer MLP classification head. Transductive setting with
# the standard train/val/test node masks.
# ============================================================

import argparse
import json
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.utils.data_loader import load_node_classification_dataset
from src.models.attr_gate import MeasureAttentionGateV3
from src.models.encoder import Encoder
from src.models.heads import MLPClassifier


# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed: int = 1):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_measure_blocks(attr_file: str, meta_file: str, device):
    """Load the SCOUT attribute tensor and split it into per-measure blocks."""
    arr = np.load(attr_file)
    with open(meta_file) as f:
        meta = json.load(f)

    attr = torch.tensor(arr, dtype=torch.float32).to(device)
    blocks, offset = [], 0
    for d in meta["block_dims"]:
        blocks.append(attr[:, offset:offset + d])
        offset += d
    return blocks, meta


# ============================================================
# Training / Evaluation
# ============================================================

def encode(enc, gate, head, data, measure_blocks, args):
    """Shared forward pass: gate -> encoder -> classification logits."""
    gated, _ = gate(measure_blocks)
    x_in = gated
    if args.use_raw_feature and getattr(data, "x", None) is not None:
        raw_x = data.x.to(gated.device).float()
        x_in = torch.cat([raw_x, gated], dim=1)
    z = enc(data.edge_index.to(gated.device), x_in)
    return head(z)


def train_epoch(enc, gate, head, data, measure_blocks, opt, crit, args):
    """Perform one epoch of training on the labelled train nodes."""
    enc.train(); gate.train(); head.train()
    opt.zero_grad()

    logits = encode(enc, gate, head, data, measure_blocks, args)
    mask = data.train_mask.to(logits.device)
    loss = crit(logits[mask], data.y.to(logits.device)[mask])
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(enc, gate, head, data, measure_blocks, args):
    """Return (acc, macro_f1) for train / val / test splits."""
    enc.eval(); gate.eval(); head.eval()

    logits = encode(enc, gate, head, data, measure_blocks, args)
    pred = logits.argmax(dim=1).cpu()
    y = data.y.cpu()

    out = {}
    for split in ("train", "val", "test"):
        mask = getattr(data, f"{split}_mask").cpu()
        acc = float((pred[mask] == y[mask]).float().mean())
        f1 = f1_score(y[mask].numpy(), pred[mask].numpy(), average="macro")
        out[split] = (acc, float(f1))
    return out


# ============================================================
# Main Training Pipeline
# ============================================================

def run(args):
    """Run node classification training."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ------------------ Load dataset ------------------
    data = load_node_classification_dataset(
        args.dataset, args.data_root, use_raw_feature=args.use_raw_feature, seed=args.seed
    )
    logging.info(
        f"{args.dataset}: N={data.num_nodes}, classes={data.num_classes}, "
        f"train/val/test={int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}"
    )

    # ------------------ Load SCOUT attributes ------------------
    measure_blocks, meta = load_measure_blocks(args.attr_file, args.meta_file, device)
    block_dims = meta["block_dims"]

    # ------------------ Initialize modules ------------------
    gate = MeasureAttentionGateV3(block_dims, att_dim=args.att_dim).to(device)
    in_dim = sum(block_dims)
    if args.use_raw_feature and getattr(data, "x", None) is not None:
        in_dim += int(data.x.size(-1))
    enc = Encoder(args.model, in_dim, args.hidden,
                  dropout=args.dropout, num_layers=args.layer).to(device)
    head = MLPClassifier(args.hidden, args.hidden, data.num_classes,
                         dropout=args.dropout).to(device)

    # ------------------ Optimization setup ------------------
    params = list(enc.parameters()) + list(head.parameters()) + list(gate.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    # ------------------ Training loop ------------------
    best_val, best_test, best_epoch = 0.0, (0.0, 0.0), 0
    patience = 0
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(enc, gate, head, data, measure_blocks, opt, crit, args)

        if ep % args.val_every == 0:
            scores = evaluate(enc, gate, head, data, measure_blocks, args)
            val_acc = scores["val"][0]
            if val_acc > best_val:
                best_val, best_test, best_epoch = val_acc, scores["test"], ep
                patience = 0
                logging.info(
                    f"Epoch {ep:04d}: loss={loss:.4f} val_acc={val_acc:.4f} "
                    f"test_acc={best_test[0]:.4f} test_f1={best_test[1]:.4f}"
                )
            else:
                patience += 1
                if args.patience and patience >= args.patience:
                    logging.info(f"Early stopping at epoch {ep} (no val improvement).")
                    break

    logging.info(
        f"Final Best (epoch {best_epoch}): val_acc={best_val:.4f} "
        f"test_acc={best_test[0]:.4f} test_macro_f1={best_test[1]:.4f}"
    )


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SCOUT Node Classification Trainer")
    # Dataset / Path
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_root", type=str, default="./datasets")
    p.add_argument("--attr_file", type=str, required=True)
    p.add_argument("--meta_file", type=str, required=True)

    # Model
    p.add_argument("--model", type=str, default="gcn", choices=["gcn", "sage"])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--att_dim", type=int, default=128)
    p.add_argument("--layer", type=int, default=2)

    # Optimization
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stop after this many validations without improvement (0 disables).")

    # Misc
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--use_raw_feature", action="store_true",
                   help="Concatenate original node features (default: off)")

    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run(args)
