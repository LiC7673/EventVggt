#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
VERIFY_GPU="${VERIFY_GPU:-${GPUS##*,}}"
PORT="${PORT:-29928}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
NUM_VIEWS="${NUM_VIEWS:-4}"
INITIAL_SCENE_IDX="${INITIAL_SCENE_IDX:-0}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-12}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-10}"
LDR_ID="${LDR_ID:-ev_5}"
EPOCHS="${EPOCHS:-20}"
EXP_NAME="${EXP_NAME:-mul_loss_detail_gt_temporal_detail_img_reliability_scene12}"
CKPT="${CKPT:-${ROOT_DIR}/checkpoints/${EXP_NAME}/checkpoint-last.pth}"
VERIFY_AFTER_TRAIN="${VERIFY_AFTER_TRAIN:-true}"
VERIFY_OUT="${VERIFY_OUT:-${ROOT_DIR}/finetune_vaild/results/${EXP_NAME}_counterfactual}"
GRID_OUT="${GRID_OUT:-${ROOT_DIR}/exp_test/grid_source_diagnostics/${EXP_NAME}}"

PRETRAINED_ARGS=()
if [[ -n "${PRETRAINED:-}" ]]; then
  PRETRAINED_ARGS=("pretrained=${PRETRAINED}")
fi

cd "${ROOT_DIR}"
echo "[train] temporal-detail image-guided reliability on GPUs ${GPUS}, epochs=${EPOCHS}"
echo "[train] scenes=[${INITIAL_SCENE_IDX}, $((INITIAL_SCENE_IDX + ACTIVE_SCENE_COUNT - 1))], count=${ACTIVE_SCENE_COUNT}"
CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "${PORT}" \
  mul_loss_fine/finetune_mul_loss_detail_gt_temporal_detail_img_reliability.py \
  num_workers="${NUM_WORKERS}" \
  pin_mem="${PIN_MEM}" \
  epochs="${EPOCHS}" \
  exp_name="${EXP_NAME}" \
  data.num_views="${NUM_VIEWS}" \
  data.initial_scene_idx="${INITIAL_SCENE_IDX}" \
  data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
  data.test_frame_count="${TEST_FRAME_COUNT}" \
  data.ldr_event_id="${LDR_ID}" \
  "${PRETRAINED_ARGS[@]}" \
  "$@"

if [[ "${VERIFY_AFTER_TRAIN}" == "true" ]]; then
  echo "[verify] event counterfactual on GPU ${VERIFY_GPU}"
  CUDA_VISIBLE_DEVICES="${VERIFY_GPU}" python finetune_vaild/verify_event_input_counterfactual.py \
    --checkpoint "${CKPT}" \
    --model-variant temporal_detail \
    --event-hidden-dim 16 \
    --event-num-bins 10 \
    --refiner-residual-scale 0.035 \
    --event-delta-highpass-kernel 9 \
    --event-delta-patch-zero-mean \
    --event-delta-patch-size 14 \
    --event-delta-abs-limit 0.025 \
    --event-reliability-gate-enabled \
    --event-reliability-gate-floor 0.20 \
    --event-reliability-init-bias 0.0 \
    --num-views "${NUM_VIEWS}" \
    --ldr-event-id "${LDR_ID}" \
    --samples-per-scene 1 \
    --max-visualizations 4 \
    --output-dir "${VERIFY_OUT}"

  echo "[diagnose] grid source on GPU ${VERIFY_GPU}"
  CUDA_VISIBLE_DEVICES="${VERIFY_GPU}" python exp_test/diagnose_grid_source.py \
    --checkpoint "${CKPT}" \
    --model-variant temporal_detail \
    --event-hidden-dim 16 \
    --event-num-bins 10 \
    --refiner-residual-scale 0.035 \
    --event-delta-highpass-kernel 9 \
    --event-delta-patch-zero-mean \
    --event-delta-patch-size 14 \
    --event-delta-abs-limit 0.025 \
    --event-reliability-gate-enabled \
    --event-reliability-gate-floor 0.20 \
    --event-reliability-init-bias 0.0 \
    --num-views "${NUM_VIEWS}" \
    --ldr-event-id "${LDR_ID}" \
    --samples-per-scene 1 \
    --max-samples 4 \
    --visual-samples 4 \
    --output-dir "${GRID_OUT}"

  echo "[done] counterfactual summary: ${VERIFY_OUT}/summary.json"
  echo "[done] grid summary: ${GRID_OUT}/grid_source_summary.json"
fi
