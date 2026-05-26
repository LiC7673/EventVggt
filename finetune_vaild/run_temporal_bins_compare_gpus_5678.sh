#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Validation uses one GPU per model; these are physical GPUs 5 and 6.
UNIFORM_GPU="${UNIFORM_GPU:-4}"
TEMPORAL_GPU="${TEMPORAL_GPU:-5}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
SPLIT="${SPLIT:-test}"
NUM_VIEWS="${NUM_VIEWS:-4}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-4}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-1}"
EVENT_THRESHOLD="${EVENT_THRESHOLD:-0.20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_PARENT="${OUTPUT_PARENT:-${ROOT_DIR}/finetune_vaild/results}"

UNIFORM_CKPT="${UNIFORM_CKPT:-${ROOT_DIR}/checkpoints/mul_loss_detail_gt_uniform/checkpoint-last.pth}"
TEMPORAL_CKPT="${TEMPORAL_CKPT:-${ROOT_DIR}/checkpoints/mul_loss_detail_gt_temporal_bins/checkpoint-last.pth}"

for ckpt in "$UNIFORM_CKPT" "$TEMPORAL_CKPT"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt"
    exit 1
  fi
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_PARENT}/temporal_bins_compare_${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

run_one() {
  local method="$1"
  local variant="$2"
  local ckpt="$3"
  local gpu="$4"
  shift 4
  local log_file="${LOG_DIR}/${method}_gpu_${gpu}.log"

  echo "[CUDA ${gpu}] start ${method}"
  (
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" exp_test/visualize_normal_error_event_corr.py \
      --root "$DATA_ROOT" \
      --checkpoint "$ckpt" \
      --model-variant "$variant" \
      --split "$SPLIT" \
      --output-dir "${RUN_DIR}/${method}" \
      --num-views "$NUM_VIEWS" \
      --active-scene-count "$ACTIVE_SCENE_COUNT" \
      --samples-per-scene "$SAMPLES_PER_SCENE" \
      --event-support-mode temporal_polarity \
      --event-threshold "$EVENT_THRESHOLD" \
      --gt-normal-source depth \
      --num-workers "$NUM_WORKERS" \
      "$@"
  ) >"$log_file" 2>&1
}

run_one "detail_gt_uniform" "base" "$UNIFORM_CKPT" "$UNIFORM_GPU" "$@" &
UNIFORM_PID="$!"
run_one "detail_gt_temporal_bins" "temporal_bins" "$TEMPORAL_CKPT" "$TEMPORAL_GPU" "$@" &
TEMPORAL_PID="$!"

STATUS=0
wait "$UNIFORM_PID" || STATUS=1
wait "$TEMPORAL_PID" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "Validation failed; inspect ${LOG_DIR}"
  exit "$STATUS"
fi

"$PYTHON_BIN" "${ROOT_DIR}/finetune_vaild/summarize_normal_error_comparison.py" \
  --run-dir "$RUN_DIR" \
  --methods "detail_gt_uniform" "detail_gt_temporal_bins"

echo "Results saved in ${RUN_DIR}"
