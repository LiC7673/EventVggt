#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPU_GROUPS="${GPU_GROUPS:-2,3 4,5 6,7}"
VARIANTS="${VARIANTS:-a0_rgb_only,a1_direct_event,a2_wo_reliability,a3_wo_multildr,a4_wo_detail,a5_full}"
PORT_BASE="${PORT_BASE:-30240}"
EPOCHS="${EPOCHS:-20}"
NUM_VIEWS="${NUM_VIEWS:-4}"
TRAIN_INITIAL_SCENE_IDX="${TRAIN_INITIAL_SCENE_IDX:-0}"
TRAIN_SCENE_COUNT="${TRAIN_SCENE_COUNT:-12}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-10}"
LDR_ID="${LDR_ID:-ev_5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PIN_MEM="${PIN_MEM:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
RELIABILITY_CKPT="${RELIABILITY_CKPT:-${ROOT_DIR}/abl_event_exp/real_reliability_stage/reliability_net/checkpoint-best.pth}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/abl_event_exp/paper_module_ablation}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/ablation_logs/paper_module_ablation_$(date +%Y%m%d_%H%M%S)}"
MULTILDR_IDS="${MULTILDR_IDS:-[ev_2,ev_5,ev_10]}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${OUTPUT_ROOT}/scene_manifest.json}"

if [[ ! -f "${ROOT_DIR}/ckpt/model.pt" ]]; then
  echo "[error] original VGGT checkpoint missing: ${ROOT_DIR}/ckpt/model.pt" >&2
  exit 1
fi
if [[ ! -f "${RELIABILITY_CKPT}" ]]; then
  echo "[error] frozen ReliabilityNet checkpoint missing: ${RELIABILITY_CKPT}" >&2
  exit 1
fi

IFS=',' read -r -a VARIANT_ARRAY <<< "${VARIANTS}"
read -r -a GROUP_ARRAY <<< "${GPU_GROUPS}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"
EXTRA_ARGS=("$@")

if [[ ! -f "${SCENE_MANIFEST}" ]]; then
  python -m paper_main_ablation.make_eval_scene_manifest \
    --root "${DATA_ROOT}" \
    --output "${SCENE_MANIFEST}" \
    --train-scene-count "${TRAIN_SCENE_COUNT}" \
    --test-scene-count 4 \
    --train-ldr-levels 2 5 10 \
    --ldr-levels 1 2 5 10 \
    --num-views "${NUM_VIEWS}"
fi

launch_one() {
  local variant="$1"
  local gpu_group="$2"
  local job_index="$3"
  local checkpoint="${OUTPUT_ROOT}/${variant}/checkpoint-last.pth"
  local log_file="${LOG_ROOT}/${job_index}_${variant}_gpus_${gpu_group//,/_}.log"
  local process_count
  local accum_iter
  local -a launch_mode

  IFS=',' read -r -a local_gpus <<< "${gpu_group}"
  process_count="${#local_gpus[@]}"
  if [[ "${process_count}" -gt 1 ]]; then
    launch_mode=(--multi_gpu)
    accum_iter=1
  else
    launch_mode=()
    # Match the two-GPU jobs' effective batch size.
    accum_iter=2
  fi

  if [[ "${SKIP_EXISTING}" == "true" && -f "${checkpoint}" ]]; then
    echo "[skip] ${variant}: ${checkpoint}"
    return 2
  fi

  echo "[launch] ${variant} on GPUs ${gpu_group}"
  CUDA_VISIBLE_DEVICES="${gpu_group}" HYDRA_FULL_ERROR=1 \
  "${ACCELERATE_BIN}" launch \
    "${launch_mode[@]}" \
    --num_processes "${process_count}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port "$((PORT_BASE + job_index))" \
    paper_main_ablation/finetune_main_table.py \
    +main_table_variant="${variant}" \
    +main_table_output_root="${OUTPUT_ROOT}" \
    epochs="${EPOCHS}" \
    accum_iter="${accum_iter}" \
    num_workers="${NUM_WORKERS}" \
    pin_mem="${PIN_MEM}" \
    print_freq=100 \
    log_freq=100 \
    save_every_steps=2000 \
    vis.save_every_steps=4000 \
    data.num_views="${NUM_VIEWS}" \
    data.root="${DATA_ROOT}" \
    data.initial_scene_idx="${TRAIN_INITIAL_SCENE_IDX}" \
    data.active_scene_count="${TRAIN_SCENE_COUNT}" \
    data.test_frame_count="${TEST_FRAME_COUNT}" \
    data.ldr_event_id="${LDR_ID}" \
    +data.mul_ldr_train_ids="${MULTILDR_IDS}" \
    +data.module_scene_manifest="${SCENE_MANIFEST}" \
    +model.reliability_checkpoint="${RELIABILITY_CKPT}" \
    "${EXTRA_ARGS[@]}" \
    > "${log_file}" 2>&1 &
  LAUNCHED_PID="$!"
}

echo "[main-table] variants=${VARIANTS}"
echo "[main-table] train scenes=${TRAIN_INITIAL_SCENE_IDX}..$((TRAIN_INITIAL_SCENE_IDX + TRAIN_SCENE_COUNT - 1))"
echo "[main-table] outputs=${OUTPUT_ROOT}"
echo "[main-table] logs=${LOG_ROOT}"

variant_index=0
job_index=0
while [[ "${variant_index}" -lt "${#VARIANT_ARRAY[@]}" ]]; do
  pids=()
  names=()
  for gpu_group in "${GROUP_ARRAY[@]}"; do
    [[ "${variant_index}" -ge "${#VARIANT_ARRAY[@]}" ]] && break
    variant="${VARIANT_ARRAY[variant_index]}"
    variant_index=$((variant_index + 1))
    if launch_one "${variant}" "${gpu_group}" "${job_index}"; then
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

echo "[done] all leave-one-out module-ablation checkpoints are ready"
echo "[next] bash paper_main_ablation/run_ldr_scene_eval_gpus_234567.sh"
