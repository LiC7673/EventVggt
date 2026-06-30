#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPU_CANDIDATES="${GPU_CANDIDATES:-2,3,4,5,6,7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-12}"
NUM_VIEWS="${NUM_VIEWS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LDR_ID="${LDR_ID:-ev_5}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
EVENT_FLOOR="${EVENT_FLOOR:-0.0}"
PORT="${PORT:-30420}"
STAGE1_DIR="${STAGE1_DIR:-${ROOT_DIR}/abl_event_exp/additive_decomposer_stage1_v2_scene12}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_DIR}/checkpoint-best.pth}"
STAGE2_EXP="${STAGE2_EXP:-two_stage_frozen_geometry_full_img_reliability_v4_stable_scene12}"
STAGE2_DIR="${ROOT_DIR}/abl_event_exp/${STAGE2_EXP}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/abl_event_exp/two_stage_logs_$(date +%Y%m%d_%H%M%S)}"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

if [[ -n "${GPUS:-}" ]]; then
  SELECTED_GPUS="${GPUS}"
else
  SELECTED=()
  IFS=',' read -r -a CANDIDATES <<< "${GPU_CANDIDATES}"
  for gpu in "${CANDIDATES[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      SELECTED+=("${gpu}")
    else
      echo "[gpu-skip] GPU ${gpu}: free=${free_mib:-unknown} MiB"
    fi
  done
  SELECTED_GPUS="$(IFS=','; echo "${SELECTED[*]}")"
fi
IFS=',' read -r -a GPU_ARRAY <<< "${SELECTED_GPUS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ARRAY[@]}}"
if (( NUM_PROCESSES > ${#GPU_ARRAY[@]} )); then
  NUM_PROCESSES="${#GPU_ARRAY[@]}"
fi
if (( NUM_PROCESSES < 2 )); then
  echo "[error] At least two free GPUs are required; selected=${SELECTED_GPUS:-none}" >&2
  exit 1
fi
STAGE1_GPU="${STAGE1_GPU:-${GPU_ARRAY[0]}}"
echo "[gpu-select] stage1=${STAGE1_GPU}, stage2=${SELECTED_GPUS} (${NUM_PROCESSES} ranks)"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[stage1] pretrain additive decomposer"
  env CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" PYTHONUNBUFFERED=1 \
    python -m event_filter_two_stage.train_stage1_additive_decomposer \
    --root "${DATA_ROOT}" --out-dir "${STAGE1_DIR}" \
    --ldr-event-id "${LDR_ID}" --num-views "${NUM_VIEWS}" \
    --active-scene-count "${ACTIVE_SCENE_COUNT}" --num-workers "${NUM_WORKERS}" \
    --epochs "${STAGE1_EPOCHS}" --amp \
    2>&1 | tee "${LOG_DIR}/stage1_decomposer.log"
else
  echo "[stage1] reuse ${STAGE1_CKPT}"
fi
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[error] Stage 1 did not produce ${STAGE1_CKPT}" >&2
  exit 1
fi

echo "[stage2] frozen Stage-1 geometry stream + full-img reliability"
env CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${ACCELERATE_BIN}" launch --multi_gpu \
  --num_processes "${NUM_PROCESSES}" --num_machines 1 \
  --main_process_port "${PORT}" --mixed_precision bf16 --dynamo_backend no \
  event_filter_two_stage/finetune_stage2_frozen_geometry_stream.py \
  exp_name="${STAGE2_EXP}" epochs="${STAGE2_EPOCHS}" \
  hydra.run.dir="${STAGE2_DIR}" \
  data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" data.ldr_event_id="${LDR_ID}" \
  data.initial_scene_idx=0 data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
  num_workers="${NUM_WORKERS}" pin_mem=false \
  +model.decomposition_checkpoint="${STAGE1_CKPT}" \
  +model.geometry_event_floor="${EVENT_FLOOR}" \
  2>&1 | tee "${LOG_DIR}/stage2_full_img_reliability.log"

echo "[done] Stage 1: ${STAGE1_CKPT}"
echo "[done] Stage 2: ${STAGE2_DIR}/checkpoint-last.pth"
