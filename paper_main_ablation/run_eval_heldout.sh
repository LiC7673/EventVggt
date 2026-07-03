#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-7}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/paper_main_ablation/main_table_manifest.json}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/abl_event_exp/paper_main_table/results_heldout_scene12_15}"
ROOT_DATA="${ROOT_DATA:-/data1/lzh/dataset/reflective_raw}"
HELDOUT_INITIAL_SCENE_IDX="${HELDOUT_INITIAL_SCENE_IDX:-12}"
HELDOUT_SCENE_COUNT="${HELDOUT_SCENE_COUNT:-4}"
NUM_VIEWS="${NUM_VIEWS:-4}"
LDR_ID="${LDR_ID:-ev_5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_BATCHES="${MAX_BATCHES:-}"

ARGS=()
if [[ -n "${MAX_BATCHES}" ]]; then
  ARGS+=(--max-batches "${MAX_BATCHES}")
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m paper_main_ablation.evaluate_main_table \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --root "${ROOT_DATA}" \
  --split all \
  --initial-scene-idx "${HELDOUT_INITIAL_SCENE_IDX}" \
  --active-scene-count "${HELDOUT_SCENE_COUNT}" \
  --test-frame-count 120 \
  --num-views "${NUM_VIEWS}" \
  --ldr-event-id "${LDR_ID}" \
  --event-resize-bins 10 \
  --batch-size 1 \
  --num-workers "${NUM_WORKERS}" \
  --device cuda:0 \
  --amp bf16 \
  "${ARGS[@]}" \
  "$@"

python -m paper_main_ablation.summarize_main_table \
  --input "${OUT_DIR}/summary.csv" \
  --output "${OUT_DIR}/paper_main_table.csv"

echo "[done] ${OUT_DIR}/summary.csv"
echo "[done] ${OUT_DIR}/summary.json"
echo "[done] ${OUT_DIR}/paper_main_table.csv"
