#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPU_CANDIDATES="${GPU_CANDIDATES:-2,3,4,5,6,7}"
AUTO_SELECT_FREE_GPUS="${AUTO_SELECT_FREE_GPUS:-true}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
PORT_STAGE2="${PORT_STAGE2:-30220}"
PORT_STAGE3="${PORT_STAGE3:-30221}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
SCENE_COUNT="${SCENE_COUNT:-12}"
NUM_VIEWS="${NUM_VIEWS:-4}"
LDR_ID="${LDR_ID:-ev_5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-10}"
STAGE1_DIR="${STAGE1_DIR:-${ROOT_DIR}/checkpoints/reliability_net_stage1_scene12}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_DIR}/checkpoint-best.pth}"
STAGE2_EXP="${STAGE2_EXP:-staged_reliability_stage2_frozen_scene12}"
STAGE3_EXP="${STAGE3_EXP:-staged_reliability_stage3_joint_scene12}"
STAGE2_CKPT="${ROOT_DIR}/checkpoints/${STAGE2_EXP}/checkpoint-last.pth"
STAGE2_DONE="${ROOT_DIR}/checkpoints/${STAGE2_EXP}/.stage_complete"
STAGE3_CKPT="${ROOT_DIR}/checkpoints/${STAGE3_EXP}/checkpoint-last.pth"
STAGE3_DONE="${ROOT_DIR}/checkpoints/${STAGE3_EXP}/.stage_complete"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/ablation_logs/staged_reliability_$(date +%Y%m%d_%H%M%S)}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

# DDP replicates the complete model on every selected GPU. One occupied card
# is enough to kill the whole job, so filter physical GPUs by free memory.
if [[ -n "${GPUS:-}" ]]; then
  SELECTED_GPUS="${GPUS}"
elif [[ "${AUTO_SELECT_FREE_GPUS}" == "true" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  SELECTED=()
  IFS=',' read -r -a CANDIDATES <<< "${GPU_CANDIDATES}"
  for gpu in "${CANDIDATES[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      SELECTED+=("${gpu}")
    else
      echo "[gpu-skip] physical GPU ${gpu}: free=${free_mib:-unknown} MiB, required>=${MIN_FREE_MIB} MiB"
    fi
  done
  SELECTED_GPUS="$(IFS=','; echo "${SELECTED[*]}")"
else
  SELECTED_GPUS="${GPU_CANDIDATES}"
fi

IFS=',' read -r -a SELECTED_ARRAY <<< "${SELECTED_GPUS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#SELECTED_ARRAY[@]}}"
if (( NUM_PROCESSES > ${#SELECTED_ARRAY[@]} )); then
  echo "[gpu-adjust] processes ${NUM_PROCESSES} -> ${#SELECTED_ARRAY[@]} to match visible GPUs"
  NUM_PROCESSES="${#SELECTED_ARRAY[@]}"
fi
if (( NUM_PROCESSES < 2 )); then
  echo "[error] Need at least two free GPUs from ${GPU_CANDIDATES}; selected=${SELECTED_GPUS:-none}" >&2
  exit 1
fi
GPUS="${SELECTED_GPUS}"
STAGE1_GPU="${STAGE1_GPU:-${SELECTED_ARRAY[0]}}"
echo "[gpu-select] physical GPUs=${GPUS}, processes=${NUM_PROCESSES}, min_free=${MIN_FREE_MIB} MiB"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[stage1] ReliabilityNet checkpoint missing; training on physical GPU ${STAGE1_GPU}"
  if ! env GPU="${STAGE1_GPU}" DATA_ROOT="${DATA_ROOT}" OUT_DIR="${STAGE1_DIR}" \
    ACTIVE_SCENE_COUNT="${SCENE_COUNT}" EPOCHS="${STAGE1_EPOCHS}" \
    PYTHONUNBUFFERED=1 \
    bash reliability_pretrain/run_stage1_reliability_net.sh \
    2>&1 | tee "${LOG_DIR}/stage1_reliability.log"; then
    echo "[error] Stage 1 failed; log: ${LOG_DIR}/stage1_reliability.log" >&2
    exit 1
  fi
else
  echo "[stage1] reuse ${STAGE1_CKPT}"
fi

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[error] Stage 1 did not produce ${STAGE1_CKPT}" >&2
  exit 1
fi

if [[ "${SKIP_EXISTING}" == "true" && -f "${STAGE2_DONE}" && -f "${STAGE2_CKPT}" ]]; then
  echo "[stage2] reuse ${STAGE2_CKPT}"
else
  echo "[stage2] frozen ReliabilityNet, GPUs ${GPUS}"
  if ! env CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${ACCELERATE_BIN}" launch \
    --multi_gpu --num_processes "${NUM_PROCESSES}" --num_machines 1 \
    --main_process_port "${PORT_STAGE2}" --mixed_precision bf16 --dynamo_backend no \
    reliability_staged_finetune/finetune_stage2_frozen_reliability.py \
    exp_name="${STAGE2_EXP}" epochs="${STAGE2_EPOCHS}" \
    pretrained="${ROOT_DIR}/ckpt/model.pt" \
    data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" \
    data.ldr_event_id="${LDR_ID}" data.initial_scene_idx=0 \
    data.active_scene_count="${SCENE_COUNT}" num_workers="${NUM_WORKERS}" pin_mem=false \
    +data.additive_event_root=events_additive \
    +model.stage1_reliability_checkpoint="${STAGE1_CKPT}" \
    2>&1 | tee "${LOG_DIR}/stage2_frozen.log"; then
    echo "[error] Stage 2 failed; log: ${LOG_DIR}/stage2_frozen.log" >&2
    exit 1
  fi
  if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "[error] Stage 2 exited without writing ${STAGE2_CKPT}" >&2
    exit 1
  fi
  touch "${STAGE2_DONE}"
fi

if [[ ! -f "${STAGE2_CKPT}" ]]; then
  echo "[error] Stage 2 did not produce ${STAGE2_CKPT}" >&2
  exit 1
fi

if [[ "${SKIP_EXISTING}" == "true" && -f "${STAGE3_DONE}" && -f "${STAGE3_CKPT}" ]]; then
  echo "[stage3] already complete"
else
  echo "[stage3] joint low-LR finetuning, GPUs ${GPUS}"
  if ! env CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${ACCELERATE_BIN}" launch \
    --multi_gpu --num_processes "${NUM_PROCESSES}" --num_machines 1 \
    --main_process_port "${PORT_STAGE3}" --mixed_precision bf16 --dynamo_backend no \
    reliability_staged_finetune/finetune_stage3_joint_reliability.py \
    exp_name="${STAGE3_EXP}" epochs="${STAGE3_EPOCHS}" lr=1e-5 \
    pretrained="${STAGE2_CKPT}" \
    data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" \
    data.ldr_event_id="${LDR_ID}" data.initial_scene_idx=0 \
    data.active_scene_count="${SCENE_COUNT}" num_workers="${NUM_WORKERS}" pin_mem=false \
    +data.additive_event_root=events_additive \
    +model.stage1_reliability_checkpoint="${STAGE1_CKPT}" \
    +train.reliability_lr_scale=0.25 \
    +loss.joint_reliability_weight=0.30 \
    2>&1 | tee "${LOG_DIR}/stage3_joint.log"; then
    echo "[error] Stage 3 failed; log: ${LOG_DIR}/stage3_joint.log" >&2
    exit 1
  fi
  if [[ ! -f "${STAGE3_CKPT}" ]]; then
    echo "[error] Stage 3 exited without writing ${STAGE3_CKPT}" >&2
    exit 1
  fi
  touch "${STAGE3_DONE}"
fi

echo "[done] stage2=${STAGE2_CKPT}"
echo "[done] stage3=${STAGE3_CKPT}"
echo "[logs] ${LOG_DIR}"
