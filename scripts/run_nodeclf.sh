#!/bin/bash
# ============================================================
# SCOUT: Structure-Aware Aspect and Anchor-Count Selection
# for Positional Attribute Augmentation (WWW 2026)
#
# File: scripts/run_nodeclf.sh
# Description:
#   Unified script for Node Classification experiments
#   - (1) Without original node features
#   - (2) With original node features
# ============================================================

# ---------------- GPU & Base Settings ----------------
export CUDA_VISIBLE_DEVICES=0
DATASET="Cora"
DATA_ROOT="./datasets"
ATTR_FILE="./attrs/Cora_concat_centrality/concat_all_top10.684.npy"
META_FILE="./attrs/Cora_concat_centrality/meta_concat_all_top10.684.json"

MODEL="gcn"
HIDDEN=64
LAYER=2
DROPOUT=0.5
EPOCHS=1000
LR=0.01
WEIGHT_DECAY=0.0005
VAL_EVERY=10
PATIENCE=20

LOG_DIR="./logs"
mkdir -p $LOG_DIR


# ============================================================
# (1) Without Original Node Features
# ============================================================
echo "Running SCOUT Node Classification (w/o original features)..."
python -m src.core.train_nodeclf \
  --dataset $DATASET \
  --data_root $DATA_ROOT \
  --attr_file $ATTR_FILE \
  --meta_file $META_FILE \
  --model $MODEL \
  --hidden $HIDDEN \
  --layer $LAYER \
  --dropout $DROPOUT \
  --epochs $EPOCHS \
  --lr $LR \
  --weight_decay $WEIGHT_DECAY \
  --val_every $VAL_EVERY \
  --patience $PATIENCE \
  | tee $LOG_DIR/${DATASET,,}_nodeclf_wofeat_${MODEL}.log


# ============================================================
# (2) With Original Node Features
# ============================================================
echo "Running SCOUT Node Classification (w/ original features)..."
python -m src.core.train_nodeclf \
  --dataset $DATASET \
  --data_root $DATA_ROOT \
  --attr_file $ATTR_FILE \
  --meta_file $META_FILE \
  --model $MODEL \
  --hidden $HIDDEN \
  --layer $LAYER \
  --dropout $DROPOUT \
  --epochs $EPOCHS \
  --lr $LR \
  --weight_decay $WEIGHT_DECAY \
  --val_every $VAL_EVERY \
  --patience $PATIENCE \
  --use_raw_feature \
  | tee $LOG_DIR/${DATASET,,}_nodeclf_wfeat_${MODEL}.log

echo "All node classification experiments completed."
