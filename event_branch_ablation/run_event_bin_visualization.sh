#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
DECOMPOSE_GPU="${DECOMPOSE_GPU:-2}"
GEOMETRY_GPU="${GEOMETRY_GPU:-3}"
INITIAL_SCENE_IDX="${INITIAL_SCENE_IDX:-12}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-1}"
NUM_VIEWS="${NUM_VIEWS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
LDR_ID="${LDR_ID:-ev_5}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/abl_event_exp/event_bin_visualization_${RUN_ID}}"

DECOMPOSE_CKPT="${DECOMPOSE_CKPT:-${ROOT_DIR}/abl_event_exp/full_to_additive_tokens_img_reliability_v4_stable_scene12/checkpoint-last.pth}"
GEOMETRY_CKPT="${GEOMETRY_CKPT:-${ROOT_DIR}/abl_event_exp/geometry_motion_full_img_reliability_v5_stable_scene12/checkpoint-last.pth}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}"

COMMON_ARGS=(
  --root "${DATA_ROOT}"
  --split test
  --initial-scene-idx "${INITIAL_SCENE_IDX}"
  --active-scene-count "${ACTIVE_SCENE_COUNT}"
  --num-views "${NUM_VIEWS}"
  --max-samples "${MAX_SAMPLES}"
  --ldr-event-id "${LDR_ID}"
  --num-workers 0
  --amp bf16
)

PIDS=()
NAMES=()
if [[ -f "${DECOMPOSE_CKPT}" ]]; then
  echo "[visualize] decomposition checkpoint on GPU ${DECOMPOSE_GPU}"
  env CUDA_VISIBLE_DEVICES="${DECOMPOSE_GPU}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -m event_branch_ablation.visualize_checkpoint_event_bins \
    --checkpoint "${DECOMPOSE_CKPT}" --model-kind decomposition \
    --output-dir "${OUTPUT_ROOT}/full_to_additive" "${COMMON_ARGS[@]}" \
    > "${OUTPUT_ROOT}/full_to_additive.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("full_to_additive")
else
  echo "[skip] missing checkpoint: ${DECOMPOSE_CKPT}" >&2
fi

if [[ -f "${GEOMETRY_CKPT}" ]]; then
  echo "[visualize] geometry checkpoint on GPU ${GEOMETRY_GPU}"
  env CUDA_VISIBLE_DEVICES="${GEOMETRY_GPU}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -m event_branch_ablation.visualize_checkpoint_event_bins \
    --checkpoint "${GEOMETRY_CKPT}" --model-kind geometry \
    --output-dir "${OUTPUT_ROOT}/geometry_motion" "${COMMON_ARGS[@]}" \
    > "${OUTPUT_ROOT}/geometry_motion.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("geometry_motion")
else
  echo "[skip] missing checkpoint: ${GEOMETRY_CKPT}" >&2
fi

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "[error] no checkpoint found" >&2
  exit 1
fi

STATUS=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[done] ${NAMES[$index]}"
  else
    echo "[fail] ${NAMES[$index]}; see ${OUTPUT_ROOT}/${NAMES[$index]}.log" >&2
    STATUS=1
  fi
done

echo "[outputs] ${OUTPUT_ROOT}"
exit "${STATUS}"
