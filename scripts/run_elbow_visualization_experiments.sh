#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="${DATA_ROOT:-./datasets}"
FIGURES_DIR="${FIGURES_DIR:-./figures}"
EPOCHS="${EPOCHS:-2000}"
RUNS="${RUNS:-3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES

"${PYTHON_BIN}" -m src.analysis.elbow_visualization_experiments \
  --data_root "${DATA_ROOT}" \
  --figures_dir "${FIGURES_DIR}" \
  --output_csv "${FIGURES_DIR}/results_elbow_analysis.csv" \
  --epochs "${EPOCHS}" \
  --runs "${RUNS}" \
  --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
  ${EXTRA_ARGS}
