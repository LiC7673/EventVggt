#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
VERIFY_GPU="${VERIFY_GPU:-${GPUS##*,}}"
PORT="${PORT:-29920}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
NUM_VIEWS="${NUM_VIEWS:-4}"
GEO_TEACHER_LDR_ID="${GEO_TEACHER_LDR_ID:-ev_10}"
GEO_STUDENT_LDR_IDS="${GEO_STUDENT_LDR_IDS:-ev_2,ev_5}"
EXPOSURES_PER_SAMPLE="${EXPOSURES_PER_SAMPLE:-2}"
EVAL_LDR_ID="${EVAL_LDR_ID:-ev_5}"
EXP_NAME="${EXP_NAME:-mul_loss_detail_gt_event_after_head_degrid}"
CKPT="${CKPT:-${ROOT_DIR}/checkpoints/${EXP_NAME}/checkpoint-last.pth}"
VERIFY_AFTER_TRAIN="${VERIFY_AFTER_TRAIN:-true}"
VERIFY_OUT="${VERIFY_OUT:-${ROOT_DIR}/finetune_vaild/results/${EXP_NAME}_counterfactual}"
GRID_OUT="${GRID_OUT:-${ROOT_DIR}/exp_test/grid_source_diagnostics/${EXP_NAME}}"

PRETRAINED_ARGS=()
if [[ -n "${PRETRAINED:-}" ]]; then
  PRETRAINED_ARGS=("pretrained=${PRETRAINED}")
fi

cd "${ROOT_DIR}"
echo "[train] event-after-head-degrid on GPUs ${GPUS}"
CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "${PORT}" \
  mul_loss_fine/finetune_mul_loss_detail_gt_event_after_head_degrid.py \
  num_workers="${NUM_WORKERS}" \
  pin_mem="${PIN_MEM}" \
  exp_name="${EXP_NAME}" \
  "+data.geo_teacher_ldr_id=${GEO_TEACHER_LDR_ID}" \
  "+data.geo_student_ldr_ids=[${GEO_STUDENT_LDR_IDS}]" \
  "+data.geo_exposures_per_sample=${EXPOSURES_PER_SAMPLE}" \
  "+data.geo_scenes_per_batch=1" \
  "+data.geo_num_views=${NUM_VIEWS}" \
  "+data.eval_ldr_event_id=${EVAL_LDR_ID}" \
  "${PRETRAINED_ARGS[@]}" \
  "$@"

if [[ "${VERIFY_AFTER_TRAIN}" == "true" ]]; then
  echo "[verify] event counterfactual on GPU ${VERIFY_GPU}"
  CUDA_VISIBLE_DEVICES="${VERIFY_GPU}" python finetune_vaild/verify_event_input_counterfactual.py \
    --checkpoint "${CKPT}" \
    --model-variant temporal_reliability_v2 \
    --event-hidden-dim 32 \
    --event-num-bins 10 \
    --refiner-residual-scale 0.035 \
    --event-gate-downsample 1 \
    --event-gate-smooth-kernel 3 \
    --event-reliability-floor 0.18 \
    --event-reliability-init-bias -0.5 \
    --proposal-depth-lowpass \
    --no-proposal-use-depth-hf \
    --event-proposal-weight 0.65 \
    --event-delta-highpass-kernel 9 \
    --event-delta-patch-zero-mean \
    --event-delta-patch-size 14 \
    --final-degrid-strength 0.0 \
    --final-degrid-kernel 9 \
    --num-views "${NUM_VIEWS}" \
    --ldr-event-id "${EVAL_LDR_ID}" \
    --samples-per-scene 1 \
    --max-visualizations 4 \
    --output-dir "${VERIFY_OUT}"

  echo "[diagnose] grid source on GPU ${VERIFY_GPU}"
  CUDA_VISIBLE_DEVICES="${VERIFY_GPU}" python exp_test/diagnose_grid_source.py \
    --checkpoint "${CKPT}" \
    --model-variant temporal_reliability_v2 \
    --event-hidden-dim 32 \
    --event-num-bins 10 \
    --refiner-residual-scale 0.035 \
    --event-gate-downsample 1 \
    --event-gate-smooth-kernel 3 \
    --event-reliability-floor 0.18 \
    --event-reliability-init-bias -0.5 \
    --proposal-depth-lowpass \
    --no-proposal-use-depth-hf \
    --event-proposal-weight 0.65 \
    --event-delta-highpass-kernel 9 \
    --event-delta-patch-zero-mean \
    --event-delta-patch-size 14 \
    --final-degrid-strength 0.0 \
    --final-degrid-kernel 9 \
    --num-views "${NUM_VIEWS}" \
    --ldr-event-id "${EVAL_LDR_ID}" \
    --samples-per-scene 1 \
    --max-samples 4 \
    --visual-samples 4 \
    --output-dir "${GRID_OUT}"

  echo "[done] counterfactual summary: ${VERIFY_OUT}/summary.json"
  echo "[done] grid summary: ${GRID_OUT}/grid_source_summary.json"
fi
