#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
VERIFY_GPU="${VERIFY_GPU:-${GPUS##*,}}"
PORT="${PORT:-29914}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
GEO_TEACHER_LDR_ID="${GEO_TEACHER_LDR_ID:-ev_10}"
GEO_STUDENT_LDR_IDS="${GEO_STUDENT_LDR_IDS:-ev_2,ev_5}"
EVAL_LDR_ID="${EVAL_LDR_ID:-ev_5}"
NUM_VIEWS="${NUM_VIEWS:-4}"
EXPOSURES_PER_SAMPLE="${EXPOSURES_PER_SAMPLE:-2}"
VERIFY_AFTER_TRAIN="${VERIFY_AFTER_TRAIN:-true}"
EXP_NAME="${EXP_NAME:-mul_loss_detail_gt_reliability_filter}"
CKPT="${CKPT:-${ROOT_DIR}/checkpoints/${EXP_NAME}/checkpoint-last.pth}"
VERIFY_OUT="${VERIFY_OUT:-${ROOT_DIR}/finetune_vaild/results/${EXP_NAME}_counterfactual}"

cd "${ROOT_DIR}"
echo "[train] reliability-filter event training on GPUs ${GPUS}"
CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "${PORT}" \
  mul_loss_fine/finetune_mul_loss_detail_gt_reliability_filter.py \
  num_workers="${NUM_WORKERS}" \
  pin_mem="${PIN_MEM}" \
  exp_name="${EXP_NAME}" \
  "+data.geo_teacher_ldr_id=${GEO_TEACHER_LDR_ID}" \
  "+data.geo_student_ldr_ids=[${GEO_STUDENT_LDR_IDS}]" \
  "+data.geo_exposures_per_sample=${EXPOSURES_PER_SAMPLE}" \
  "+data.geo_scenes_per_batch=1" \
  "+data.geo_num_views=${NUM_VIEWS}" \
  "+data.eval_ldr_event_id=${EVAL_LDR_ID}" \
  "$@"

if [[ "${VERIFY_AFTER_TRAIN}" == "true" ]]; then
  echo "[verify] real/zero/reverse/swap events on GPU ${VERIFY_GPU}"
  CUDA_VISIBLE_DEVICES="${VERIFY_GPU}" python finetune_vaild/verify_event_input_counterfactual.py \
    --checkpoint "${CKPT}" \
    --model-variant temporal_reliability_v2 \
    --event-hidden-dim 16 \
    --event-num-bins 10 \
    --refiner-residual-scale 0.03 \
    --event-gate-downsample 2 \
    --proposal-depth-lowpass \
    --event-proposal-weight 0.0 \
    --num-views "${NUM_VIEWS}" \
    --ldr-event-id "${EVAL_LDR_ID}" \
    --samples-per-scene 1 \
    --max-visualizations 4 \
    --sensitivity-epsilon-deg 1e-4 \
    --output-dir "${VERIFY_OUT}"
  echo "[done] verification summary: ${VERIFY_OUT}/summary.json"
fi
