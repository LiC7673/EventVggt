#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-2}"
GPUS="${GPUS:-${GPU}}"
PORT="${PORT:-29731}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
CONFIG="${CONFIG:-config/finetune_event.yaml}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
BASE_CKPT="${BASE_CKPT:-${ROOT}/ckpt/model.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/ablation}"
EPOCHS_PROXY="${EPOCHS_PROXY:-3}"
EPOCHS_CONTRIBUTION="${EPOCHS_CONTRIBUTION:-5}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-5}"
LR_STAGE1="${LR_STAGE1:-1e-4}"
LR_STAGE2="${LR_STAGE2:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_VIEWS="${NUM_VIEWS:-4}"
TRAIN_SCENES="${TRAIN_SCENES:-20}"
TEST_SCENES="${TEST_SCENES:-5}"
TEST_START="${TEST_START:-20}"
LDR_TRAIN="${LDR_TRAIN:-ev_10}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "${OUTPUT_ROOT}"/{rgb_only,raw_event,ours,no_multildr,saturation_mask}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "[error] missing base checkpoint: ${BASE_CKPT}" >&2
  exit 1
fi

echo "[test] ablation invariants"
python -m ab_st1_st2.test_ablation

train_stage1() {
  local method="$1"
  local out="${OUTPUT_ROOT}/${method}/stage1"
  mkdir -p "${out}/logs"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m ab_st1_st2.train_stage1 \
    --method "${method}" \
    --config "${CONFIG}" \
    --pretrained "${BASE_CKPT}" \
    --output "${out}" \
    --phase all \
    --epochs-proxy "${EPOCHS_PROXY}" \
    --epochs-contribution "${EPOCHS_CONTRIBUTION}" \
    --epochs-joint 0 \
    --lr "${LR_STAGE1}" \
    --num-workers "${NUM_WORKERS}" \
    --exposures 0,1,2,5,10 \
    --pair-mode all \
    "data.root=${DATA_ROOT}" \
    "data.num_views=${NUM_VIEWS}" \
    "data.train_initial_scene_idx=0" \
    "data.train_scene_count=${TRAIN_SCENES}" \
    "data.test_initial_scene_idx=${TEST_START}" \
    "data.test_scene_count=${TEST_SCENES}" \
    "data.train_holdout_frame_count=0" \
    "data.heldout_test_frame_count=120" \
    2>&1 | tee "${out}/logs/training.log"
}

if [[ "${RUN_STAGE1}" == "1" ]]; then
  echo "[Stage1] geometry-aware Multi-LDR contribution"
  train_stage1 ours
  echo "[Stage1] no-Multi-LDR single-exposure contribution"
  train_stage1 no_multildr
fi

OURS_STAGE1="${OUTPUT_ROOT}/ours/stage1/checkpoint-best.pth"
NO_MULTI_STAGE1="${OUTPUT_ROOT}/no_multildr/stage1/checkpoint-best.pth"
for required in "${OURS_STAGE1}" "${NO_MULTI_STAGE1}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[error] missing Stage1 checkpoint: ${required}" >&2
    exit 1
  fi
done

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_PROCESSES="${#GPU_ARRAY[@]}"
ACCELERATE_MODE=()
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  ACCELERATE_MODE+=(--multi_gpu)
fi

train_stage2() {
  local method="$1"
  local stage1="$2"
  local method_port="$3"
  local out="${OUTPUT_ROOT}/${method}"
  mkdir -p "${out}/logs"
  CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
  "${ACCELERATE_BIN}" launch \
    "${ACCELERATE_MODE[@]}" \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port "${method_port}" \
    ab_st1_st2/finetune_stage2.py \
    "pretrained=${BASE_CKPT}" \
    "epochs=${EPOCHS_STAGE2}" \
    "lr=${LR_STAGE2}" \
    "num_workers=${NUM_WORKERS}" \
    "data.root=${DATA_ROOT}" \
    "data.num_views=${NUM_VIEWS}" \
    "data.ldr_event_id=${LDR_TRAIN}" \
    "+data.ablation_train_initial_scene_idx=0" \
    "+data.ablation_train_scene_count=${TRAIN_SCENES}" \
    "+data.ablation_test_initial_scene_idx=${TEST_START}" \
    "+data.ablation_test_scene_count=${TEST_SCENES}" \
    "+model.ablation_method=${method}" \
    "+model.stage1_contribution_checkpoint=${stage1}" \
    "+skip_final_eval=true" \
    2>&1 | tee "${out}/logs/training.log"
}

if [[ "${RUN_STAGE2}" == "1" ]]; then
  train_stage2 raw_event "${OURS_STAGE1}" "${PORT}"
  train_stage2 ours "${OURS_STAGE1}" "$((PORT + 1))"
  train_stage2 no_multildr "${NO_MULTI_STAGE1}" "$((PORT + 2))"
  train_stage2 saturation_mask "${OURS_STAGE1}" "$((PORT + 3))"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  for method in rgb_only raw_event ours no_multildr saturation_mask; do
    mkdir -p "${OUTPUT_ROOT}/${method}/logs"
    stage1="${OURS_STAGE1}"
    checkpoint="${OUTPUT_ROOT}/${method}/stage2/checkpoint-last.pth"
    if [[ "${method}" == "no_multildr" ]]; then
      stage1="${NO_MULTI_STAGE1}"
    fi
    if [[ "${method}" == "rgb_only" ]]; then
      checkpoint="${BASE_CKPT}"
      ln -sfn "${BASE_CKPT}" "${OUTPUT_ROOT}/rgb_only/checkpoint.pth"
    else
      ln -sfn "${checkpoint}" "${OUTPUT_ROOT}/${method}/checkpoint.pth"
    fi
    if [[ ! -f "${checkpoint}" ]]; then
      echo "[error] missing evaluation checkpoint: ${checkpoint}" >&2
      exit 1
    fi
    echo "[evaluate] ${method}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -m ab_st1_st2.evaluate \
      --method "${method}" \
      --checkpoint "${checkpoint}" \
      --stage1-checkpoint "${stage1}" \
      --output-dir "${OUTPUT_ROOT}/${method}" \
      --root "${DATA_ROOT}" \
      --test-initial-scene-idx "${TEST_START}" \
      --test-scene-count "${TEST_SCENES}" \
      --num-views "${NUM_VIEWS}" \
      --num-workers "${NUM_WORKERS}" \
      2>&1 | tee "${OUTPUT_ROOT}/${method}/logs/evaluation.log"
  done
  python -m ab_st1_st2.aggregate --root "${OUTPUT_ROOT}"
fi

echo "[done] ${OUTPUT_ROOT}/ablation_results.csv"
