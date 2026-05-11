#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
SCRIPTS_DIR="fine_event"
SCRIPTS=(
  "${SCRIPTS_DIR}/finetune_ablation_rgb_coarse.py"
  "${SCRIPTS_DIR}/finetune_ablation_global_only.py"
  "${SCRIPTS_DIR}/finetune_ablation_local_only.py"
  "${SCRIPTS_DIR}/finetune_ablation_global_local.py"
  "${SCRIPTS_DIR}/finetune_ablation_tokens_4.py"
  "${SCRIPTS_DIR}/finetune_ablation_tokens_16.py"
  "${SCRIPTS_DIR}/finetune_ablation_tokens_64.py"
  "${SCRIPTS_DIR}/finetune_ablation_tokens_256.py"
  "${SCRIPTS_DIR}/finetune_ablation_event_h16.py"
  "${SCRIPTS_DIR}/finetune_ablation_event_h8.py"
  "${SCRIPTS_DIR}/finetune_ablation_event_h4.py"
  "${SCRIPTS_DIR}/finetune_ablation_event_h2.py"
)

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs configured. Set GPU_LIST, for example: GPU_LIST=0,1,2,3 bash run_ablation_8gpu.sh"
  exit 1
fi

for script in "${SCRIPTS[@]}"; do
  if [[ ! -f "${ROOT_DIR}/${script}" ]]; then
    echo "Missing script: ${ROOT_DIR}/${script}"
    exit 1
  fi
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/ablation_logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "Root: ${ROOT_DIR}"
echo "GPUs: ${GPU_LIST}"
echo "Logs: ${LOG_DIR}"
echo "Extra Hydra args: $*"
echo

PIDS=()

run_worker() {
  local worker_idx="$1"
  shift
  local gpu="${GPUS[$worker_idx]}"
  local script_idx script name log_file
  local worker_status=0

  for ((script_idx = worker_idx; script_idx < ${#SCRIPTS[@]}; script_idx += ${#GPUS[@]})); do
    script="${SCRIPTS[$script_idx]}"
    name="${script%.py}"
    log_file="${LOG_DIR}/${script_idx}_${name}_gpu${gpu}.log"
    
    echo "[GPU ${gpu}] start ${script} -> ${log_file}"
    mkdir -p "$(dirname "$log_file")"
    if (
      cd "$ROOT_DIR"
      CUDA_VISIBLE_DEVICES="$gpu" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" "$script" "$@"
    ) >"$log_file" 2>&1; then
      echo "[GPU ${gpu}] done  ${script}"
    else
      worker_status=1
      echo "[GPU ${gpu}] fail  ${script} (see ${log_file})"
    fi
  done

  return "$worker_status"
}

for worker_idx in "${!GPUS[@]}"; do
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
  echo "All ablations finished. Logs are in ${LOG_DIR}"
else
  echo "Some ablations failed. Check logs in ${LOG_DIR}"
fi

exit "$STATUS"
