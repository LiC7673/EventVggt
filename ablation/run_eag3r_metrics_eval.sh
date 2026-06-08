#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-7}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/ablation/results/eag3r_metrics}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/ablation/eag3r_eval_manifest.json}"
MAX_BATCHES="${MAX_BATCHES:-}"

cd "${ROOT_DIR}"
ARGS=()
if [[ -n "${MAX_BATCHES}" ]]; then
  ARGS+=(--max-batches "${MAX_BATCHES}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" python ablation/eag3r_metrics_eval.py \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --device cuda:0 \
  --split test \
  --num-workers 0 \
  --skip-missing \
  "${ARGS[@]}" \
  "$@"

echo "[done] ${OUT_DIR}/summary.csv"
