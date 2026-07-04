#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/abl_event_exp/multildr_token_strategy}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/ablation_logs/multildr_token_strategy_$(date +%Y%m%d_%H%M%S)}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${OUTPUT_ROOT}/scene_manifest.json}"
EPOCHS="${EPOCHS:-20}"
NUM_VIEWS="${NUM_VIEWS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
LDR_IDS="${LDR_IDS:-[ev_1,ev_2,ev_5,ev_10]}"
GPU_GROUPS=("2,3" "4,5" "6,7")
SCRIPTS=(
  "multildr_token_exp/finetune_random_ldr_full.py"
  "multildr_token_exp/finetune_paired_output_full.py"
  "multildr_token_exp/finetune_paired_token_full.py"
)
NAMES=("random_ldr_full" "paired_output_full" "paired_token_full")

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
if [[ ! -f "${SCENE_MANIFEST}" ]]; then
  python -m paper_main_ablation.make_eval_scene_manifest \
    --root "${DATA_ROOT}" \
    --output "${SCENE_MANIFEST}" \
    --train-scene-count 12 \
    --test-scene-count 4 \
    --train-ldr-levels 1 2 5 10 \
    --ldr-levels 1 2 5 10 \
    --num-views "${NUM_VIEWS}"
fi

pids=()
names=()
for index in 0 1 2; do
  name="${NAMES[index]}"
  checkpoint="${OUTPUT_ROOT}/${name}/checkpoint-last.pth"
  if [[ "${SKIP_EXISTING}" == "true" && -f "${checkpoint}" ]]; then
    echo "[skip] ${name}: ${checkpoint}"
    continue
  fi
  gpu_group="${GPU_GROUPS[index]}"
  log_file="${LOG_ROOT}/${name}_gpus_${gpu_group//,/_}.log"
  echo "[launch] ${name} on GPUs ${gpu_group}"
  CUDA_VISIBLE_DEVICES="${gpu_group}" HYDRA_FULL_ERROR=1 \
  "${ACCELERATE_BIN}" launch \
    --multi_gpu --num_processes 2 --num_machines 1 \
    --main_process_port "$((30520 + index))" \
    --mixed_precision bf16 --dynamo_backend no \
    "${SCRIPTS[index]}" \
    epochs="${EPOCHS}" num_workers="${NUM_WORKERS}" pin_mem="${PIN_MEM}" \
    data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" \
    data.test_frame_count=10 \
    +data.multildr_train_ids="${LDR_IDS}" \
    +data.multildr_scene_manifest="${SCENE_MANIFEST}" \
    +strategy_output_root="${OUTPUT_ROOT}" \
    +skip_final_eval=true \
    print_freq=100 log_freq=100 save_every_steps=2000 vis.save_every_steps=4000 \
    "$@" > "${log_file}" 2>&1 &
  pids+=("$!")
  names+=("${name}")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "[done] ${names[index]}"
  else
    echo "[fail] ${names[index]}; inspect ${LOG_ROOT}" >&2
    failed=1
  fi
done
[[ "${failed}" -eq 0 ]] || exit 1

echo "[done] checkpoints: ${OUTPUT_ROOT}"
