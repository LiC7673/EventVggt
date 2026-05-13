#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
BASE_PORT="${BASE_PORT:-29500}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

SCRIPTS_DIR="fine_event"
SCRIPTS=(
  "${SCRIPTS_DIR}/finetune_ablation_twostage_stage1_freeze_stage2.py"
  "${SCRIPTS_DIR}/finetune_ablation_twostage_stage2_freeze_stage1.py"
  "${SCRIPTS_DIR}/finetune_event_two_stage_residual.py"
  "${SCRIPTS_DIR}/finetune_ablation_twostage_event_current_rgb.py"
  "${SCRIPTS_DIR}/finetune_ablation_twostage_event_global_rgb_current_rgb.py"
  "${SCRIPTS_DIR}/finetune_ablation_twostage_global_rgb_current_event.py"
  "${SCRIPTS_DIR}/finetune_ablation_twostage_single_frame_event.py"
)

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs configured. Set GPU_LIST, for example: GPU_LIST=0,1,2,3 bash run_twostage_2gpu_8gpu.sh"
  exit 1
fi

if (( GPUS_PER_JOB <= 0 )); then
  echo "GPUS_PER_JOB must be positive."
  exit 1
fi

if (( ${#GPUS[@]} % GPUS_PER_JOB != 0 )); then
  echo "GPU count (${#GPUS[@]}) must be divisible by GPUS_PER_JOB (${GPUS_PER_JOB})."
  exit 1
fi

for script in "${SCRIPTS[@]}"; do
  if [[ ! -f "${ROOT_DIR}/${script}" ]]; then
    echo "Missing script: ${ROOT_DIR}/${script}"
    exit 1
  fi
done

GPU_GROUPS=()
for ((i = 0; i < ${#GPUS[@]}; i += GPUS_PER_JOB)); do
  group=""
  for ((j = 0; j < GPUS_PER_JOB; j += 1)); do
    gpu="${GPUS[$((i + j))]}"
    if [[ -z "$group" ]]; then
      group="$gpu"
    else
      group="${group},${gpu}"
    fi
  done
  GPU_GROUPS+=("$group")
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/ablation_logs/twostage_2gpu_${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "Root: ${ROOT_DIR}"
echo "GPU list: ${GPU_LIST}"
echo "GPUs per job: ${GPUS_PER_JOB}"
echo "GPU groups: ${GPU_GROUPS[*]}"
echo "Logs: ${LOG_DIR}"
echo "Extra Hydra args: $*"
echo

PIDS=()

run_worker() {
  local worker_idx="$1"
  shift
  local gpu_group="${GPU_GROUPS[$worker_idx]}"
  local port=$((BASE_PORT + worker_idx))
  local script_idx script name log_file
  local worker_status=0

  for ((script_idx = worker_idx; script_idx < ${#SCRIPTS[@]}; script_idx += ${#GPU_GROUPS[@]})); do
    script="${SCRIPTS[$script_idx]}"
    name="${script%.py}"
    log_file="${LOG_DIR}/${script_idx}_${name}_gpus_${gpu_group//,/}.log"

    echo "[GPUs ${gpu_group}] start ${script} -> ${log_file}"
    mkdir -p "$(dirname "$log_file")"
    if (
      cd "$ROOT_DIR"
      CUDA_VISIBLE_DEVICES="$gpu_group" HYDRA_FULL_ERROR=1 "$ACCELERATE_BIN" launch \
        --multi_gpu \
        --num_processes "$GPUS_PER_JOB" \
        --num_machines 1 \
        --main_process_port "$port" \
        "$script" "$@"
    ) >"$log_file" 2>&1; then
      echo "[GPUs ${gpu_group}] done  ${script}"
    else
      worker_status=1
      echo "[GPUs ${gpu_group}] fail  ${script} (see ${log_file})"
    fi
  done

  return "$worker_status"
}

for worker_idx in "${!GPU_GROUPS[@]}"; do
  run_worker "$worker_idx" "$@" &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    STATUS=1
  fi
done

if [[ "$STATUS" -eq 0 ]]; then
  echo "All two-stage ablations finished. Logs are in ${LOG_DIR}"
else
  echo "Some two-stage ablations failed. Check logs in ${LOG_DIR}"
fi

exit "$STATUS"
