#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
PORT_BASE="${PORT_BASE:-29940}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
NUM_VIEWS="${NUM_VIEWS:-4}"
LDR_ID="${LDR_ID:-ev_5}"
EPOCHS="${EPOCHS:-20}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-3}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-10}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-true}"
VARIANTS="${VARIANTS:-rgb_baseline,rgb_detail_gt,raw_event,raw_event_detail_gt,multildr,multildr_detail_gt,full_img_reliability}"

IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"

cd "${ROOT_DIR}"
echo "[ablation] root=${ROOT_DIR}"
echo "[ablation] variants=${VARIANTS}"
echo "[ablation] gpus=${GPUS}, epochs=${EPOCHS}, num_views=${NUM_VIEWS}, ldr=${LDR_ID}"

idx=0
for variant in "${VARIANT_LIST[@]}"; do
  variant="$(echo "${variant}" | xargs)"
  [[ -z "${variant}" ]] && continue
  port=$((PORT_BASE + idx))
  exp_name="ablation_${variant}"
  echo
  echo "================================================================================"
  echo "[train] ${variant} -> ${exp_name} on GPUs ${GPUS}, port=${port}"
  echo "================================================================================"
  CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
  "${ACCELERATE_BIN}" launch \
    --multi_gpu \
    --num_processes 2 \
    --num_machines 1 \
    --main_process_port "${port}" \
    ablation/finetune_paper_ablation.py \
    +ablation_variant="${variant}" \
    exp_name="${exp_name}" \
    epochs="${EPOCHS}" \
    eval_every_steps="${EVAL_EVERY_STEPS}" \
    +skip_final_eval="${SKIP_FINAL_EVAL}" \
    num_workers="${NUM_WORKERS}" \
    pin_mem="${PIN_MEM}" \
    data.num_views="${NUM_VIEWS}" \
    data.ldr_event_id="${LDR_ID}" \
    data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
    data.test_frame_count="${TEST_FRAME_COUNT}" \
    "$@"
  idx=$((idx + 1))
done

echo
echo "[ablation] all requested variants finished."
echo "[ablation] run metrics with:"
echo "  python ablation/eag3r_metrics_eval.py --manifest ablation/eag3r_eval_manifest.json --device cuda:0"
