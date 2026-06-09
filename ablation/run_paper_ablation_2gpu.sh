#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

# Default: consume GPUs 1,4,5,6,7 as three parallel jobs.
# Two jobs use 2 GPUs, and the last one uses GPU 7 with gradient accumulation.
GPU_GROUPS="${GPU_GROUPS:-1,4 5,6 7}"
PORT_BASE="${PORT_BASE:-29940}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
NUM_VIEWS="${NUM_VIEWS:-4}"
LDR_ID="${LDR_ID:-ev_5}"
EPOCHS="${EPOCHS:-20}"
ACCUM_ITER="${ACCUM_ITER:-1}"
SINGLE_GPU_ACCUM_ITER="${SINGLE_GPU_ACCUM_ITER:-2}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-3}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-10}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/ablation_logs/paper_ablation_parallel_$(date +%Y%m%d_%H%M%S)}"

# ablation_rgb_baseline has already finished, so it is excluded by default.
VARIANTS="${VARIANTS:-rgb_detail_gt,raw_event,raw_event_detail_gt,multildr,multildr_detail_gt,full_img_reliability}"

IFS=',' read -r -a RAW_VARIANT_LIST <<< "${VARIANTS}"
read -r -a GPU_GROUP_LIST <<< "${GPU_GROUPS}"
EXTRA_ARGS=("$@")

count_gpus() {
  local group="$1"
  if [[ -z "${group}" ]]; then
    echo 0
    return
  fi
  local no_commas="${group//,/}"
  echo $(( ${#group} - ${#no_commas} + 1 ))
}

trim() {
  local text="$1"
  echo "${text}" | xargs
}

launch_job() {
  local variant="$1"
  local gpu_group="$2"
  local job_index="$3"
  local exp_name="ablation_${variant}"
  local safe_group="${gpu_group//,/_}"
  local log_file="${LOG_ROOT}/${job_index}_${exp_name}_gpus_${safe_group}.log"
  local processes
  processes="$(count_gpus "${gpu_group}")"

  if [[ "${processes}" -lt 1 ]]; then
    echo "[error] empty GPU group: '${gpu_group}'" >&2
    return 1
  fi

  local job_accum_iter="${ACCUM_ITER}"
  if [[ "${processes}" -eq 1 ]]; then
    job_accum_iter="${SINGLE_GPU_ACCUM_ITER}"
  fi

  local launch_args=()
  if [[ "${processes}" -gt 1 ]]; then
    launch_args+=(--multi_gpu)
  fi
  launch_args+=(
    --num_processes "${processes}"
    --num_machines 1
    --main_process_port "$((PORT_BASE + job_index))"
  )

  echo "[launch] ${variant} -> ${exp_name}, GPUs=${gpu_group}, processes=${processes}, accum_iter=${job_accum_iter}"
  echo "         log=${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu_group}" HYDRA_FULL_ERROR=1 \
  "${ACCELERATE_BIN}" launch \
    "${launch_args[@]}" \
    ablation/finetune_paper_ablation.py \
    +ablation_variant="${variant}" \
    exp_name="${exp_name}" \
    epochs="${EPOCHS}" \
    accum_iter="${job_accum_iter}" \
    eval_every_steps="${EVAL_EVERY_STEPS}" \
    +skip_final_eval="${SKIP_FINAL_EVAL}" \
    num_workers="${NUM_WORKERS}" \
    pin_mem="${PIN_MEM}" \
    data.num_views="${NUM_VIEWS}" \
    data.ldr_event_id="${LDR_ID}" \
    data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
    data.test_frame_count="${TEST_FRAME_COUNT}" \
    "${EXTRA_ARGS[@]}" \
    > "${log_file}" 2>&1 &
}

cd "${ROOT_DIR}"
mkdir -p "${LOG_ROOT}"

FILTERED_VARIANTS=()
for raw_variant in "${RAW_VARIANT_LIST[@]}"; do
  variant="$(trim "${raw_variant}")"
  [[ -z "${variant}" ]] && continue

  exp_name="ablation_${variant}"
  ckpt_path="${ROOT_DIR}/checkpoints/${exp_name}/checkpoint-last.pth"
  if [[ "${SKIP_EXISTING}" == "true" && -f "${ckpt_path}" ]]; then
    echo "[skip] ${variant}: found ${ckpt_path}"
    continue
  fi
  FILTERED_VARIANTS+=("${variant}")
done

echo "[ablation] root=${ROOT_DIR}"
echo "[ablation] variants=${FILTERED_VARIANTS[*]:-<none>}"
echo "[ablation] gpu_groups=${GPU_GROUPS}, epochs=${EPOCHS}, num_views=${NUM_VIEWS}, ldr=${LDR_ID}"
echo "[ablation] skip_existing=${SKIP_EXISTING}, logs=${LOG_ROOT}"

if [[ "${#FILTERED_VARIANTS[@]}" -eq 0 ]]; then
  echo "[ablation] nothing to run."
  exit 0
fi

variant_idx=0
job_idx=0
while [[ "${variant_idx}" -lt "${#FILTERED_VARIANTS[@]}" ]]; do
  pids=()
  names=()

  for gpu_group in "${GPU_GROUP_LIST[@]}"; do
    if [[ "${variant_idx}" -ge "${#FILTERED_VARIANTS[@]}" ]]; then
      break
    fi

    variant="${FILTERED_VARIANTS[variant_idx]}"
    variant_idx=$((variant_idx + 1))
    launch_job "${variant}" "${gpu_group}" "${job_idx}"
    pids+=("$!")
    names+=("${variant}")
    job_idx=$((job_idx + 1))
  done

  status=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[fail] ${names[$i]} failed; see logs under ${LOG_ROOT}" >&2
      status=1
    else
      echo "[done] ${names[$i]}"
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
done

echo
echo "[ablation] all requested variants finished."
echo "[ablation] logs: ${LOG_ROOT}"
echo "[ablation] run metrics with:"
echo "  python ablation/eag3r_metrics_eval.py --manifest ablation/eag3r_eval_manifest.json --device cuda:0"
