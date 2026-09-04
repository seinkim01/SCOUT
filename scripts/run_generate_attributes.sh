#!/bin/bash

set -euo pipefail

DATASET="${DATASET:-Cora}"
DATA_ROOT="${DATA_ROOT:-./datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-./attrs/${DATASET}_concat_centrality}"
PREFIX="${PREFIX:-concat_all}"
SEED="${SEED:-1}"

python -m src.core.generate_attributes \
  --dataset "$DATASET" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --prefix "$PREFIX" \
  --seed "$SEED"
