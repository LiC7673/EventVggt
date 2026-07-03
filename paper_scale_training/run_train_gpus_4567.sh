#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPUS="${GPUS:-4,5,6,7}"
PORT="${PORT:-30460}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/abl_event_exp/paper_scale_60_12_12}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${OUTPUT_ROOT}/scene_split.json}"
STAGE1_LABEL_DIR="${STAGE1_LABEL_DIR:-${OUTPUT_ROOT}/reliability_labels}"
STAGE1_NET_DIR="${STAGE1_NET_DIR:-${OUTPUT_ROOT}/reliability_net}"
RELIABILITY_CKPT="${RELIABILITY_CKPT:-${STAGE1_NET_DIR}/checkpoint-best.pth}"
STAGE1_GPU="${STAGE1_GPU:-4}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE1_LDR_IDS="${STAGE1_LDR_IDS:-ev_2 ev_5 ev_10}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_VIEWS="${NUM_VIEWS:-2}"
LDR_IDS="${LDR_IDS:-[ev_2,ev_5,ev_10]}"
EVAL_LDR_ID="${EVAL_LDR_ID:-ev_5}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/train_gpus_4567.log}"

mkdir -p "${OUTPUT_ROOT}"
if [[ ! -f "${SCENE_MANIFEST}" ]]; then
  echo "[split] creating immutable 60/12/12 scene manifest"
  SPLIT_ARGS=()
  if [[ -n "${SCENE_METADATA:-}" ]]; then
    SPLIT_ARGS+=(--metadata "${SCENE_METADATA}")
  fi
  python -m paper_scale_training.make_scene_split \
    --root "${DATA_ROOT}" \
    --output "${SCENE_MANIFEST}" \
    --train-count 60 \
    --val-count 12 \
    --test-count 12 \
    --seed "${SPLIT_SEED:-2026}" \
    "${SPLIT_ARGS[@]}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${RELIABILITY_CKPT}" ]]; then
  echo "[stage1] rendering leakage-free reliability labels from train/val scenes"
  read -r -a STAGE1_LDR_ARRAY <<< "${STAGE1_LDR_IDS}"
  CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m paper_scale_training.render_reliability_scene_split \
    --root "${DATA_ROOT}" \
    --scene-manifest "${SCENE_MANIFEST}" \
    --output-dir "${STAGE1_LABEL_DIR}" \
    --train-ldr-ids "${STAGE1_LDR_ARRAY[@]}" \
    --val-ldr-id "${EVAL_LDR_ID}" \
    --event-bins 10 \
    --num-workers "${NUM_WORKERS}" \
    --preview-count 8 \
    > "${OUTPUT_ROOT}/stage1_render.log" 2>&1

  echo "[stage1] training ReliabilityNet on physical GPU ${STAGE1_GPU}"
  CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m real_reliability_stage.train_reliability_net \
    --data-dir "${STAGE1_LABEL_DIR}" \
    --out-dir "${STAGE1_NET_DIR}" \
    --num-bins 10 \
    --batch-size "${STAGE1_BATCH_SIZE:-4}" \
    --num-workers "${NUM_WORKERS}" \
    --epochs "${STAGE1_EPOCHS}" \
    --lr "${STAGE1_LR:-1e-4}" \
    --base-channels 32 \
    --preview-count 4 \
    --pin-memory \
    --amp \
    > "${OUTPUT_ROOT}/stage1_train.log" 2>&1
fi

if [[ ! -f "${RELIABILITY_CKPT}" ]]; then
  echo "[error] Stage-1 ReliabilityNet did not produce: ${RELIABILITY_CKPT}" >&2
  exit 1
fi

echo "[train] GPUs=${GPUS}; scenes=60/12/12; views=${NUM_VIEWS}; epochs=${EPOCHS}; patience=${PATIENCE}"
CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "${PORT}" \
  paper_scale_training/finetune_full_60_12_12.py \
  +paper_scale_output_root="${OUTPUT_ROOT}" \
  +paper_scale_epochs="${EPOCHS}" \
  +paper_scale_early_stopping_patience="${PATIENCE}" \
  +data.scene_split_manifest="${SCENE_MANIFEST}" \
  +data.mul_ldr_train_ids="${LDR_IDS}" \
  +data.eval_ldr_event_id="${EVAL_LDR_ID}" \
  +model.reliability_checkpoint="${RELIABILITY_CKPT}" \
  data.root="${DATA_ROOT}" \
  data.num_views="${NUM_VIEWS}" \
  num_workers="${NUM_WORKERS}" \
  pin_mem=true \
  print_freq=100 \
  log_freq=100 \
  save_every_steps=2000 \
  vis.save_every_steps=5000 \
  > "${LOG_FILE}" 2>&1

echo "[done] ${OUTPUT_ROOT}/full_model_train60_val12/checkpoint-best.pth"
echo "[next] GPU=7 bash paper_scale_training/run_test_once.sh"
