#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-7}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/abl_event_exp/paper_scale_60_12_12}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${OUTPUT_ROOT}/scene_split.json}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_ROOT}/full_model_train60_val12/checkpoint-best.pth}"
RESULT_DIR="${RESULT_DIR:-${OUTPUT_ROOT}/final_test_12_scenes}"
MAX_BATCHES="${MAX_BATCHES:-}"
NUM_VIEWS="${NUM_VIEWS:-2}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] best checkpoint missing: ${CHECKPOINT}" >&2
  exit 1
fi
ARGS=()
if [[ -n "${MAX_BATCHES}" ]]; then
  ARGS+=(--max-batches "${MAX_BATCHES}")
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m paper_scale_training.evaluate_heldout_test \
  --checkpoint "${CHECKPOINT}" \
  --scene-manifest "${SCENE_MANIFEST}" \
  --output-dir "${RESULT_DIR}" \
  --root "${DATA_ROOT}" \
  --num-views "${NUM_VIEWS}" \
  --ldr-event-id "${LDR_ID:-ev_5}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --device cuda:0 \
  "${ARGS[@]}"

echo "[done] ${RESULT_DIR}/summary.csv"
echo "[done] ${RESULT_DIR}/summary.json"
