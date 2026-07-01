#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-2,3}"
PORT="${PORT:-29941}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
NUM_VIEWS="${NUM_VIEWS:-4}"
INITIAL_SCENE_IDX="${INITIAL_SCENE_IDX:-12}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-12}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-10}"
LDR_ID="${LDR_ID:-5}"
EPOCHS="${EPOCHS:-20}"
EXP_NAME="${EXP_NAME:-original_event_img_reliability_scene12}"
VIS_PANEL_WIDTH="${VIS_PANEL_WIDTH:-180}"
VIS_SAVE_EVERY="${VIS_SAVE_EVERY:-500}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PRETRAINED_ARGS=()
if [[ -n "${PRETRAINED:-}" ]]; then
  PRETRAINED_ARGS=("pretrained=${PRETRAINED}")
fi

echo "[train] original-event image-guided reliability on GPUs ${GPUS}, epochs=${EPOCHS}"
echo "[train] scenes=[${INITIAL_SCENE_IDX}, $((INITIAL_SCENE_IDX + ACTIVE_SCENE_COUNT - 1))], count=${ACTIVE_SCENE_COUNT}"
echo "[train] output=abl_event_exp/${EXP_NAME}"

CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "${PORT}" \
  event_branch_ablation/finetune_original_event_img_reliability.py \
  num_workers="${NUM_WORKERS}" \
  pin_mem="${PIN_MEM}" \
  epochs="${EPOCHS}" \
  exp_name="${EXP_NAME}" \
  data.num_views="${NUM_VIEWS}" \
  data.initial_scene_idx="${INITIAL_SCENE_IDX}" \
  data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
  data.test_frame_count="${TEST_FRAME_COUNT}" \
  data.ldr_event_id="${LDR_ID}" \
  vis.event_bins_enabled=true \
  vis.event_bins_num_views="${NUM_VIEWS}" \
  vis.event_bin_panel_width="${VIS_PANEL_WIDTH}" \
  vis.save_every_steps="${VIS_SAVE_EVERY}" \
  "${PRETRAINED_ARGS[@]}" \
  "$@"

echo "[done] checkpoint: abl_event_exp/${EXP_NAME}/checkpoint-last.pth"
echo "[done] event bins: abl_event_exp/${EXP_NAME}/train_vis/event_bins/"
