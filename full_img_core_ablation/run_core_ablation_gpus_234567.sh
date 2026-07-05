#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/abl_event_exp/full_img_core_ablation}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/ablation_logs/full_img_core_ablation_$(date +%Y%m%d_%H%M%S)}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${OUTPUT_ROOT}/scene_manifest.json}"
EPOCHS="${EPOCHS:-20}"
NUM_VIEWS="${NUM_VIEWS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
LDR_IDS="${LDR_IDS:-[ev_1,ev_2,ev_5,ev_10]}"
GPU_GROUPS=("2,3" "4,5" "6,7")
VARIANTS=(
  "temporal_detail_no_gate"
  "gate_no_img_supervision"
  "img_reliability_no_detail_gt"
  "full_img_reliability"
  "full_img_reliability_token_multildr"
)

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
if [[ ! -f "${SCENE_MANIFEST}" ]]; then
  python -m paper_main_ablation.make_eval_scene_manifest \
    --root "${DATA_ROOT}" --output "${SCENE_MANIFEST}" \
    --train-scene-count 12 --test-scene-count 4 \
    --train-ldr-levels 1 2 5 10 --ldr-levels 1 2 5 10 \
    --num-views "${NUM_VIEWS}"
fi

launch_variant() {
  local variant="$1"
  local gpu_group="$2"
  local job_index="$3"
  local checkpoint="${OUTPUT_ROOT}/${variant}/checkpoint-last.pth"
  local log_file="${LOG_ROOT}/${variant}_gpus_${gpu_group//,/_}.log"
  if [[ "${SKIP_EXISTING}" == "true" && -f "${checkpoint}" ]]; then
    echo "[skip] ${variant}: ${checkpoint}"
    return 2
  fi
  echo "[launch] ${variant} on GPUs ${gpu_group}"
  CUDA_VISIBLE_DEVICES="${gpu_group}" HYDRA_FULL_ERROR=1 \
  "${ACCELERATE_BIN}" launch \
    --multi_gpu --num_processes 2 --num_machines 1 \
    --main_process_port "$((30620 + job_index))" \
    --mixed_precision bf16 --dynamo_backend no \
    full_img_core_ablation/finetune_core_ablation.py \
    +core_ablation_variant="${variant}" \
    +core_ablation_output_root="${OUTPUT_ROOT}" \
    epochs="${EPOCHS}" num_workers="${NUM_WORKERS}" pin_mem="${PIN_MEM}" \
    data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" data.test_frame_count=10 \
    +data.multildr_train_ids="${LDR_IDS}" \
    +data.multildr_scene_manifest="${SCENE_MANIFEST}" \
    +strategy_output_root="${OUTPUT_ROOT}" \
    +skip_final_eval=true print_freq=100 log_freq=100 \
    save_every_steps=2000 vis.save_every_steps=4000 \
    "$@" > "${log_file}" 2>&1 &
  LAUNCHED_PID="$!"
}

variant_index=0
job_index=0
while [[ "${variant_index}" -lt "${#VARIANTS[@]}" ]]; do
  pids=()
  names=()
  for gpu_group in "${GPU_GROUPS[@]}"; do
    [[ "${variant_index}" -ge "${#VARIANTS[@]}" ]] && break
    variant="${VARIANTS[variant_index]}"
    variant_index=$((variant_index + 1))
    if launch_variant "${variant}" "${gpu_group}" "${job_index}"; then
      pids+=("${LAUNCHED_PID}")
      names+=("${variant}")
      job_index=$((job_index + 1))
    else
      status="$?"
      [[ "${status}" -eq 2 ]] || exit "${status}"
    fi
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
done

echo "[done] ${OUTPUT_ROOT}"
