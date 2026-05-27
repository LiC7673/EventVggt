#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
VERIFY_GPU="${VERIFY_GPU:-${GPUS##*,}}"
PORT="${PORT:-29910}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
LDR_TRAIN_IDS="${LDR_TRAIN_IDS:-ev_2,ev_5,ev_10}"
EVAL_LDR_ID="${EVAL_LDR_ID:-ev_5}"
NUM_VIEWS="${NUM_VIEWS:-4}"
VERIFY_AFTER_TRAIN="${VERIFY_AFTER_TRAIN:-true}"
EXP_NAME="${EXP_NAME:-mul_loss_detail_gt_temporal_reliability_v2}"
CKPT="${CKPT:-${ROOT_DIR}/checkpoints/${EXP_NAME}/checkpoint-last.pth}"
VERIFY_OUT="${VERIFY_OUT:-${ROOT_DIR}/finetune_vaild/results/${EXP_NAME}_counterfactual}"

cd "${ROOT_DIR}"
echo "[train] temporal reliability V2 on GPUs ${GPUS}"
CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "${PORT}" \
  mul_loss_fine/finetune_mul_loss_detail_gt_temporal_reliability_v2.py \
  num_workers="${NUM_WORKERS}" \
  pin_mem="${PIN_MEM}" \
  exp_name="${EXP_NAME}" \
  "+data.eval_ldr_event_id=${EVAL_LDR_ID}" \
  "+data.mul_ldr_train_ids=[${LDR_TRAIN_IDS}]" \
  "+data.mul_ldr_exposures_per_sample=2" \
  "+data.mul_ldr_scenes_per_batch=1" \
  "+data.mul_ldr_num_views=${NUM_VIEWS}" \
  "$@"

if [[ "${VERIFY_AFTER_TRAIN}" == "true" ]]; then
  echo "[verify] real/zero/reverse/swap events on GPU ${VERIFY_GPU}"
  CUDA_VISIBLE_DEVICES="${VERIFY_GPU}" python finetune_vaild/verify_event_input_counterfactual.py \
    --checkpoint "${CKPT}" \
    --model-variant temporal_reliability_v2 \
    --event-hidden-dim 16 \
    --event-num-bins 10 \
    --refiner-residual-scale 0.015 \
    --event-gate-downsample 4 \
    --num-views "${NUM_VIEWS}" \
    --ldr-event-id "${EVAL_LDR_ID}" \
    --samples-per-scene 1 \
    --max-visualizations 4 \
    --sensitivity-epsilon-deg 1e-4 \
    --output-dir "${VERIFY_OUT}"
  python - "${VERIFY_OUT}/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    summary = json.load(handle)
if not summary.get("event_output_sensitivity_detected", False):
    raise SystemExit("Event counterfactual failed: zeroing events did not measurably change the model output.")
zero = summary["comparisons"]["real_vs_zero"]
reverse = summary["comparisons"]["real_vs_reverse_time"]
print(
    "[verify] event path active; real-vs-zero normal advantage(deg)="
    f"{zero['normal_error_advantage_deg']:.6f}, "
    "real-vs-reverse normal advantage(deg)="
    f"{reverse['normal_error_advantage_deg']:.6f}"
)
PY
  echo "[done] verification summary: ${VERIFY_OUT}/summary.json"
fi
