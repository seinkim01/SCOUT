#!/bin/bash

set -euo pipefail

DATASETS="${DATASETS:-Cora Citeseer Pubmed}"
CENTRALITIES="${CENTRALITIES:-pagerank betweenness closeness eigenvector}"
K_GRID="${K_GRID:-8 16 32 64 128 256}"
DATA_ROOT="${DATA_ROOT:-./datasets}"
ATTR_ROOT="${ATTR_ROOT:-./attrs/c3_k_sweep}"
RESULT_ROOT="${RESULT_ROOT:-./results/c3_k_sweep}"
ELBOW_SUMMARY="${ELBOW_SUMMARY:-./results/c3_elbow/c3_elbow_summary.json}"
SEED="${SEED:-42}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-gcn}"
DECODER="${DECODER:-mlp}"
HIDDEN="${HIDDEN:-64}"
LAYER="${LAYER:-2}"
DROPOUT="${DROPOUT:-0.5}"
EPOCHS="${EPOCHS:-2000}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
VAL_EVERY="${VAL_EVERY:-10}"
NEG_RATIO="${NEG_RATIO:-1.0}"
MRR_NEG_K="${MRR_NEG_K:-100}"
USE_RAW_FEATURE="${USE_RAW_FEATURE:-0}"
PYTHON="${PYTHON:-python3}"

export CUDA_VISIBLE_DEVICES

for dataset in $DATASETS; do
  for centrality in $CENTRALITIES; do
    pair_grid="$K_GRID"
    if [[ -f "$ELBOW_SUMMARY" ]]; then
      elbow_k="$("$PYTHON" - "$ELBOW_SUMMARY" "$dataset" "$centrality" <<'PY'
import json
import sys

path, dataset, centrality = sys.argv[1:4]
with open(path, "r", encoding="utf-8") as f:
    rows = json.load(f)
for row in rows:
    if row.get("dataset") == dataset and row.get("centrality") == centrality:
        print(int(row["elbow_k"]))
        break
PY
)"
      if [[ -n "$elbow_k" ]]; then
        pair_grid="$(printf '%s\n%s\n' "$K_GRID" "$elbow_k" | tr ' ' '\n' | awk 'NF' | sort -n | uniq | tr '\n' ' ')"
      fi
    fi

    for k in $pair_grid; do
      attr_dir="$ATTR_ROOT/$dataset/$centrality/k$k"
      log_dir="$RESULT_ROOT/$dataset/$centrality/k$k"
      mkdir -p "$attr_dir" "$log_dir"

      "$PYTHON" -m src.core.generate_attributes \
        --dataset "$dataset" \
        --data_root "$DATA_ROOT" \
        --output_dir "$attr_dir" \
        --prefix "concat_all" \
        --seed "$SEED" \
        --centrality_names "$centrality" \
        --top_k_percent 100 \
        --max_anchors "$k"

      meta_file="$(find "$attr_dir" -maxdepth 1 -name 'meta_concat_all_top100.json' | head -1)"
      attr_file="$(find "$attr_dir" -maxdepth 1 -name 'concat_all_top100.npy' | head -1)"

      train_args=(
        -m src.core.train_linkpred
        --dataset "$dataset"
        --data_root "$DATA_ROOT"
        --attr_file "$attr_file"
        --meta_file "$meta_file"
        --model "$MODEL"
        --decoder "$DECODER"
        --hidden "$HIDDEN"
        --layer "$LAYER"
        --dropout "$DROPOUT"
        --epochs "$EPOCHS"
        --lr "$LR"
        --weight_decay "$WEIGHT_DECAY"
        --val_every "$VAL_EVERY"
        --neg_ratio "$NEG_RATIO"
        --mrr_neg_k "$MRR_NEG_K"
        --seed "$SEED"
      )
      if [[ "$USE_RAW_FEATURE" == "1" ]]; then
        train_args+=(--use_raw_feature)
      fi

      "$PYTHON" "${train_args[@]}" 2>&1 | tee "$log_dir/train.log"
    done
  done
done

"$PYTHON" -m src.analysis.summarize_c3_k_sweep \
  --sweep_dir "$RESULT_ROOT" \
  --elbow_summary "$ELBOW_SUMMARY" \
  --output_csv "$RESULT_ROOT/c3_elbow_vs_optimal_k.csv"
