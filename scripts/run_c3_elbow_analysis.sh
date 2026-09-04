#!/bin/bash

set -euo pipefail

DATASETS="${DATASETS:-Cora Citeseer Pubmed}"
DATA_ROOT="${DATA_ROOT:-./datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/c3_elbow}"
SEED="${SEED:-42}"
MIN_ANCHORS="${MIN_ANCHORS:-8}"
MAX_ANCHORS="${MAX_ANCHORS:-}"
PYTHON="${PYTHON:-python3}"

args=(
  -m src.analysis.c3_elbow_analysis
  --datasets $DATASETS
  --data_root "$DATA_ROOT"
  --output_dir "$OUTPUT_DIR"
  --seed "$SEED"
  --min_anchors "$MIN_ANCHORS"
)

if [[ -n "$MAX_ANCHORS" ]]; then
  args+=(--max_anchors "$MAX_ANCHORS")
fi

"$PYTHON" "${args[@]}"
