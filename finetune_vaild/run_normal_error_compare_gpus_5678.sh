#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
# CUDA device indices are zero-based: IDs 4,5,6,7 are physical GPUs 5-8.
GPU_LIST="${GPU_LIST:-4,5,6,7}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
SPLIT="${SPLIT:-test}"
NUM_VIEWS="${NUM_VIEWS:-4}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-4}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-1}"
EVENT_THRESHOLD="${EVENT_THRESHOLD:-0.20}"
EVENT_HIGH_FRACTION="${EVENT_HIGH_FRACTION:-0.20}"
EVENT_LOW_FRACTION="${EVENT_LOW_FRACTION:-0.20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL_VARIANT="${MODEL_VARIANT:-base}"
OUTPUT_PARENT="${OUTPUT_PARENT:-${ROOT_DIR}/finetune_vaild/results}"

BASELINE_CKPT="${BASELINE_CKPT:-${ROOT_DIR}/checkpoints/mul_loss_baseline/checkpoint-last.pth}"
DETAIL_GT_CKPT="${DETAIL_GT_CKPT:-${ROOT_DIR}/checkpoints/mul_loss_detail_gt/checkpoint-last.pth}"
DETAIL_GT_SALIENT_CKPT="${DETAIL_GT_SALIENT_CKPT:-${ROOT_DIR}/checkpoints/mul_loss_detail_gt_salient/checkpoint-last.pth}"
MV_ALL_DETAIL_GT_CKPT="${MV_ALL_DETAIL_GT_CKPT:-${ROOT_DIR}/checkpoints/mul_loss_mv_all_detail_gt/checkpoint-last.pth}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
METHODS=("baseline" "detail_gt" "detail_gt_salient" "mv_all_detail_gt")
CHECKPOINTS=("$BASELINE_CKPT" "$DETAIL_GT_CKPT" "$DETAIL_GT_SALIENT_CKPT" "$MV_ALL_DETAIL_GT_CKPT")

if (( ${#GPUS[@]} < ${#METHODS[@]} )); then
  echo "Need four GPUs, got GPU_LIST=${GPU_LIST}"
  exit 1
fi

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt"
    echo "Override its environment variable when the checkpoint directory differs."
    exit 1
  fi
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_PARENT}/normal_error_compare_${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "Validation run: ${RUN_DIR}"
echo "GPU assignment: baseline=${GPUS[0]}, detail_gt=${GPUS[1]}, detail_gt_salient=${GPUS[2]}, mv_all_detail_gt=${GPUS[3]}"
echo "Dataset: root=${DATA_ROOT}, split=${SPLIT}, scenes=${ACTIVE_SCENE_COUNT}, samples/scene=${SAMPLES_PER_SCENE}, views=${NUM_VIEWS}"
echo "Event threshold=${EVENT_THRESHOLD}, high/low fractions=${EVENT_HIGH_FRACTION}/${EVENT_LOW_FRACTION}"

PIDS=()
for idx in "${!METHODS[@]}"; do
  method="${METHODS[$idx]}"
  gpu="${GPUS[$idx]}"
  ckpt="${CHECKPOINTS[$idx]}"
  out_dir="${RUN_DIR}/${method}"
  log_file="${LOG_DIR}/${method}_gpu_${gpu}.log"

  echo "[GPU ${gpu}] start ${method}: ${ckpt}"
  (
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" exp_test/visualize_normal_error_event_corr.py \
      --root "$DATA_ROOT" \
      --checkpoint "$ckpt" \
      --split "$SPLIT" \
      --output-dir "$out_dir" \
      --num-views "$NUM_VIEWS" \
      --active-scene-count "$ACTIVE_SCENE_COUNT" \
      --samples-per-scene "$SAMPLES_PER_SCENE" \
      --event-support-mode temporal_polarity \
      --event-threshold "$EVENT_THRESHOLD" \
      --event-high-fraction "$EVENT_HIGH_FRACTION" \
      --event-low-fraction "$EVENT_LOW_FRACTION" \
      --gt-normal-source depth \
      --num-workers "$NUM_WORKERS" \
      --model-variant "$MODEL_VARIANT" \
      "$@"
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then
    echo "[GPU ${GPUS[$idx]}] done ${METHODS[$idx]}"
  else
    echo "[GPU ${GPUS[$idx]}] fail ${METHODS[$idx]} (see ${LOG_DIR}/${METHODS[$idx]}_gpu_${GPUS[$idx]}.log)"
    STATUS=1
  fi
done

if [[ "$STATUS" -ne 0 ]]; then
  echo "Some validation jobs failed. Check ${LOG_DIR}"
  exit "$STATUS"
fi

"$PYTHON_BIN" "${ROOT_DIR}/finetune_vaild/summarize_normal_error_comparison.py" \
  --run-dir "$RUN_DIR" \
  --methods "${METHODS[@]}"

echo "All validation results are in ${RUN_DIR}"
