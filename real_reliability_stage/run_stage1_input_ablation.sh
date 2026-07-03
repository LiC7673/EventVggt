#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-7}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/abl_event_exp/real_reliability_stage/reliability_net/checkpoint-best.pth}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/abl_event_exp/real_reliability_stage/labels}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/abl_event_exp/real_reliability_stage/stage1_input_ablation}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PREVIEW_COUNT="${PREVIEW_COUNT:-12}"
MAX_BATCHES="${MAX_BATCHES:-}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] Stage-1 checkpoint missing: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${DATA_DIR}/manifest_test.json" ]]; then
  echo "[error] Stage-1 test manifest missing: ${DATA_DIR}/manifest_test.json" >&2
  exit 1
fi

ARGS=()
if [[ -n "${MAX_BATCHES}" ]]; then
  ARGS+=(--max-batches "${MAX_BATCHES}")
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m real_reliability_stage.evaluate_stage1_input_ablation \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUT_DIR}" \
  --split test \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --preview-count "${PREVIEW_COUNT}" \
  --device cuda:0 \
  --amp bf16 \
  "${ARGS[@]}" \
  "$@"

echo "[done] ${OUT_DIR}/summary.json"
echo "[done] ${OUT_DIR}/condition_metrics.csv"
echo "[done] ${OUT_DIR}/preview/"
