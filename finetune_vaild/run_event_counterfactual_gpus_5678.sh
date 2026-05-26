#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Physical GPUs 5,6,7,8 are zero-based CUDA device IDs 4,5,6,7.
GPU_LIST="${GPU_LIST:-4,5,6,7}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
NUM_VIEWS="${NUM_VIEWS:-4}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-4}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_PARENT="${OUTPUT_PARENT:-${ROOT_DIR}/finetune_vaild/results}"

BASELINE_CKPT="${BASELINE_CKPT:-${ROOT_DIR}/checkpoints/mul_loss_baseline/checkpoint-last.pth}"
UNIFORM_CKPT="${UNIFORM_CKPT:-${ROOT_DIR}/checkpoints/mul_loss_detail_gt_uniform/checkpoint-last.pth}"
SELECTIVE_CKPT="${SELECTIVE_CKPT:-${ROOT_DIR}/checkpoints/mul_loss_detail_gt_selective_event/checkpoint-last.pth}"
TEMPORAL_GATED_CKPT="${TEMPORAL_GATED_CKPT:-${ROOT_DIR}/checkpoints/mul_loss_detail_gt_temporal_gated/checkpoint-last.pth}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
METHODS=("baseline" "detail_gt_uniform" "detail_gt_selective_event" "detail_gt_temporal_gated")
VARIANTS=("base" "base" "base" "temporal_gated_detail")
CHECKPOINTS=("$BASELINE_CKPT" "$UNIFORM_CKPT" "$SELECTIVE_CKPT" "$TEMPORAL_GATED_CKPT")

if (( ${#GPUS[@]} < ${#METHODS[@]} )); then
  echo "Need four CUDA device IDs, got GPU_LIST=${GPU_LIST}"
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_PARENT}/event_counterfactual_${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

PIDS=()
RUN_METHODS=()
RUN_GPUS=()
for idx in "${!METHODS[@]}"; do
  method="${METHODS[$idx]}"
  variant="${VARIANTS[$idx]}"
  ckpt="${CHECKPOINTS[$idx]}"
  gpu="${GPUS[$idx]}"
  if [[ ! -f "$ckpt" ]]; then
    echo "[CUDA ${gpu}] skip ${method}: missing ${ckpt}"
    continue
  fi

  echo "[CUDA ${gpu}] probe ${method} (${variant})"
  (
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" finetune_vaild/verify_event_input_counterfactual.py \
      --root "$DATA_ROOT" \
      --checkpoint "$ckpt" \
      --model-variant "$variant" \
      --output-dir "${RUN_DIR}/${method}" \
      --split test \
      --num-views "$NUM_VIEWS" \
      --active-scene-count "$ACTIVE_SCENE_COUNT" \
      --samples-per-scene "$SAMPLES_PER_SCENE" \
      --event-support-mode temporal_polarity \
      --refiner-residual-scale 0.01 \
      --event-gate-downsample 4 \
      --num-workers "$NUM_WORKERS" \
      "$@"
  ) >"${LOG_DIR}/${method}_gpu_${gpu}.log" 2>&1 &
  PIDS+=("$!")
  RUN_METHODS+=("$method")
  RUN_GPUS+=("$gpu")
done

if (( ${#PIDS[@]} == 0 )); then
  echo "No checkpoints found. Override BASELINE_CKPT, UNIFORM_CKPT, SELECTIVE_CKPT or TEMPORAL_GATED_CKPT."
  exit 1
fi

STATUS=0
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then
    echo "[CUDA ${RUN_GPUS[$idx]}] done ${RUN_METHODS[$idx]}"
  else
    echo "[CUDA ${RUN_GPUS[$idx]}] failed ${RUN_METHODS[$idx]} (see ${LOG_DIR})"
    STATUS=1
  fi
done

if [[ "$STATUS" -ne 0 ]]; then
  exit "$STATUS"
fi

"$PYTHON_BIN" "${ROOT_DIR}/finetune_vaild/summarize_event_counterfactual.py" --run-dir "$RUN_DIR"
echo "Visualizations and JSON summaries saved in ${RUN_DIR}"
