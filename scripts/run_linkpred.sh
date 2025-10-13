#!/bin/bash
# ============================================================
# SCOUT: Structure-Aware Aspect and Anchor-Count Selection
# for Positional Attribute Augmentation (WWW 2026)
#
# File: scripts/run_linkpred.sh
# Description:
#   Unified script for Link Prediction experiments
#   - (1) Without original node features
#   - (2) With original node features
# ============================================================

# ---------------- GPU & Base Settings ----------------
export CUDA_VISIBLE_DEVICES=1
DATASET="Cora"
DATA_ROOT="./datasets"
ATTR_FILE="./attrs/Cora_concat_centrality/concat_all_top10.684.npy"
META_FILE="./attrs/Cora_concat_centrality/meta_concat_all_top10.684.json"

MODEL="gcn"
DECODER="mlp"
HIDDEN=64
LAYER=2
DROPOUT=0.5
EPOCHS=2000
LR=0.001
WEIGHT_DECAY=0.0005
VAL_EVERY=10
NEG_RATIO=1.0
MRR_NEG_K=100

LOG_DIR="./logs"
mkdir -p $LOG_DIR


# ============================================================
# ① Without Original Node Features
# ============================================================
echo "Running SCOUT Link Prediction (w/o original features)..."
python -m src.core.train_linkpred \
  --dataset $DATASET \
  --data_root $DATA_ROOT \
  --attr_file $ATTR_FILE \
  --meta_file $META_FILE \
  --model $MODEL \
  --decoder $DECODER \
  --hidden $HIDDEN \
  --layer $LAYER \
  --dropout $DROPOUT \
  --epochs $EPOCHS \
  --lr $LR \
  --weight_decay $WEIGHT_DECAY \
  --val_every $VAL_EVERY \
  --neg_ratio $NEG_RATIO \
  --mrr_neg_k $MRR_NEG_K \
  | tee $LOG_DIR/${DATASET,,}_wofeat_${MODEL}.log


# ============================================================
# ② With Original Node Features
# ============================================================
echo "Running SCOUT Link Prediction (w/ original features)..."
python -m src.core.train_linkpred \
  --dataset $DATASET \
  --data_root $DATA_ROOT \
  --attr_file $ATTR_FILE \
  --meta_file $META_FILE \
  --model $MODEL \
  --decoder $DECODER \
  --hidden $HIDDEN \
  --layer $LAYER \
  --dropout $DROPOUT \
  --epochs $EPOCHS \
  --lr $LR \
  --weight_decay $WEIGHT_DECAY \
  --val_every $VAL_EVERY \
  --neg_ratio $NEG_RATIO \
  --mrr_neg_k $MRR_NEG_K \
  --use_raw_feature \
  | tee $LOG_DIR/${DATASET,,}_wfeat_${MODEL}.log

echo "All link prediction experiments completed."
