#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPU_GROUPS="${GPU_GROUPS:-2,3 4,5 6,7}"
PORT_BASE="${PORT_BASE:-30020}"
EPOCHS="${EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
NUM_VIEWS="${NUM_VIEWS:-4}"
INITIAL_SCENE_IDX="${INITIAL_SCENE_IDX:-0}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-12}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-10}"
LDR_ID="${LDR_ID:-ev_5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/ablation_logs/scene12_ablation_$(date +%Y%m%d_%H%M%S)}"

# These six variants isolate the three paper claims:
# RGB baseline, detail supervision, raw event contribution, Multi-LDR
# invariance, their combination, and image-guided event reliability.
VARIANTS="${VARIANTS:-rgb_baseline,rgb_detail_gt,raw_event_detail_gt,multildr,multildr_detail_gt,full_img_reliability}"
IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"
read -r -a GROUP_LIST <<< "${GPU_GROUPS}"
EXTRA_ARGS=("$@")

cd "${ROOT_DIR}"
mkdir -p "${LOG_ROOT}"

launch_variant() {
  local variant="$1"
  local gpu_group="$2"
  local job_index="$3"
  local exp_name="ablation_${variant}_scene12"
  local checkpoint="${ROOT_DIR}/checkpoints/${exp_name}/checkpoint-last.pth"
  local safe_group="${gpu_group//,/_}"
  local log_file="${LOG_ROOT}/${job_index}_${exp_name}_gpus_${safe_group}.log"

  if [[ "${SKIP_EXISTING}" == "true" && -f "${checkpoint}" ]]; then
    echo "[skip] ${variant}: ${checkpoint}"
    return 2
  fi

  echo "[launch] ${variant}, GPUs=${gpu_group}, exp=${exp_name}"
  CUDA_VISIBLE_DEVICES="${gpu_group}" HYDRA_FULL_ERROR=1 \
  "${ACCELERATE_BIN}" launch \
    --multi_gpu \
    --num_processes 2 \
    --num_machines 1 \
    --main_process_port "$((PORT_BASE + job_index))" \
    ablation/finetune_paper_ablation.py \
    +ablation_variant="${variant}" \
    exp_name="${exp_name}" \
    epochs="${EPOCHS}" \
    eval_every_steps=0 \
    +skip_final_eval=true \
    num_workers="${NUM_WORKERS}" \
    pin_mem="${PIN_MEM}" \
    data.num_views="${NUM_VIEWS}" \
    data.ldr_event_id="${LDR_ID}" \
    data.initial_scene_idx="${INITIAL_SCENE_IDX}" \
    data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
    data.test_frame_count="${TEST_FRAME_COUNT}" \
    "${EXTRA_ARGS[@]}" \
    > "${log_file}" 2>&1 &
  LAUNCHED_PID="$!"
}

echo "[scene12-ablation] train scenes: initial=${INITIAL_SCENE_IDX}, count=${ACTIVE_SCENE_COUNT}"
echo "[scene12-ablation] GPU groups: ${GPU_GROUPS}"
echo "[scene12-ablation] logs: ${LOG_ROOT}"

variant_idx=0
job_idx=0
while [[ "${variant_idx}" -lt "${#VARIANT_LIST[@]}" ]]; do
  pids=()
  names=()

  for gpu_group in "${GROUP_LIST[@]}"; do
    [[ "${variant_idx}" -ge "${#VARIANT_LIST[@]}" ]] && break
    variant="$(echo "${VARIANT_LIST[variant_idx]}" | xargs)"
    variant_idx=$((variant_idx + 1))
    [[ -z "${variant}" ]] && continue

    if launch_variant "${variant}" "${gpu_group}" "${job_idx}"; then
      pids+=("${LAUNCHED_PID}")
      names+=("${variant}")
      job_idx=$((job_idx + 1))
    else
      code="$?"
      if [[ "${code}" -ne 2 ]]; then
        exit "${code}"
      fi
    fi
  done

  status=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[fail] ${names[$i]} failed; inspect ${LOG_ROOT}" >&2
      status=1
    else
      echo "[done] ${names[$i]}"
    fi
  done
  [[ "${status}" -ne 0 ]] && exit "${status}"
done

# Stage 2 depends on the Multi-LDR + detail checkpoint, so it starts only
# after phase 1 has completed. It uses GPUs 6,7 by default.
TEACHER="${ROOT_DIR}/checkpoints/ablation_multildr_detail_gt_scene12/checkpoint-last.pth"
STAGE2_CHECKPOINT="${ROOT_DIR}/checkpoints/multildr_detail_then_reliability_scene12/checkpoint-last.pth"
if [[ "${SKIP_EXISTING}" == "true" && -f "${STAGE2_CHECKPOINT}" ]]; then
  echo "[skip] combined stage 2: ${STAGE2_CHECKPOINT}"
else
  if [[ ! -f "${TEACHER}" ]]; then
    echo "[error] stage-2 teacher was not produced: ${TEACHER}" >&2
    exit 1
  fi
  echo "[stage2] frozen Multi-LDR detail teacher + event reliability on GPUs 6,7"
  GPUS=6,7 \
  PORT="$((PORT_BASE + job_idx + 10))" \
  EPOCHS="${STAGE2_EPOCHS}" \
  NUM_VIEWS="${NUM_VIEWS}" \
  INITIAL_SCENE_IDX="${INITIAL_SCENE_IDX}" \
  ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT}" \
  TEST_FRAME_COUNT="${TEST_FRAME_COUNT}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PIN_MEM="${PIN_MEM}" \
  EXP_NAME=multildr_detail_then_reliability_scene12 \
  TEACHER="${TEACHER}" \
  bash mul_loss_fine/run_multildr_detail_then_reliability_2gpu.sh \
    > "${LOG_ROOT}/stage2_multildr_detail_then_reliability_gpus_6_7.log" 2>&1
fi

echo "[done] all scene12 ablations finished"
echo "[next] GPU=7 bash ablation/run_scene12_heldout_eval.sh"
