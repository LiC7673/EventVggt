#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
CONTROL_GPUS="${CONTROL_GPUS:-5,6}"
SELECTIVE_EVENT_GPUS="${SELECTIVE_EVENT_GPUS:-7,8}"
BASE_PORT="${BASE_PORT:-29820}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
OMP_THREADS="${OMP_THREADS:-1}"

CONTROL_SCRIPT="mul_loss_fine/finetune_mul_loss_detail_gt_uniform.py"
SELECTIVE_EVENT_SCRIPT="mul_loss_fine/finetune_mul_loss_detail_gt_selective_event.py"

for script in "$CONTROL_SCRIPT" "$SELECTIVE_EVENT_SCRIPT"; do
  if [[ ! -f "${ROOT_DIR}/${script}" ]]; then
    echo "Missing script: ${ROOT_DIR}/${script}"
    exit 1
  fi
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/ablation_logs/detail_gt_event_pair_${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "GT detail event-weighting ablation"
echo "Uniform GT detail: GPUs ${CONTROL_GPUS}"
echo "Selective event-weighted GT detail: GPUs ${SELECTIVE_EVENT_GPUS}"
echo "Logs: ${LOG_DIR}"
echo "Extra Hydra args: $*"

run_one() {
  local name="$1"
  local script="$2"
  local gpu_group="$3"
  local port="$4"
  shift 4
  local log_file="${LOG_DIR}/${name}_gpus_${gpu_group//,/}.log"

  echo "[GPUs ${gpu_group}] start ${name} -> ${log_file}"
  (
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES="$gpu_group" \
    HYDRA_FULL_ERROR=1 \
    OMP_NUM_THREADS="$OMP_THREADS" \
    MKL_NUM_THREADS="$OMP_THREADS" \
    "$ACCELERATE_BIN" launch \
      --multi_gpu \
      --num_processes 2 \
      --num_machines 1 \
      --main_process_port "$port" \
      "$script" \
      num_workers="$NUM_WORKERS" \
      pin_mem="$PIN_MEM" \
      "$@"
  ) >"$log_file" 2>&1
}

run_one "detail_gt_uniform" "$CONTROL_SCRIPT" "$CONTROL_GPUS" "$BASE_PORT" "$@" &
CONTROL_PID="$!"
run_one "detail_gt_selective_event" "$SELECTIVE_EVENT_SCRIPT" "$SELECTIVE_EVENT_GPUS" "$((BASE_PORT + 1))" "$@" &
SELECTIVE_PID="$!"

STATUS=0
if wait "$CONTROL_PID"; then
  echo "[GPUs ${CONTROL_GPUS}] done detail_gt_uniform"
else
  echo "[GPUs ${CONTROL_GPUS}] fail detail_gt_uniform"
  STATUS=1
fi
if wait "$SELECTIVE_PID"; then
  echo "[GPUs ${SELECTIVE_EVENT_GPUS}] done detail_gt_selective_event"
else
  echo "[GPUs ${SELECTIVE_EVENT_GPUS}] fail detail_gt_selective_event"
  STATUS=1
fi

if [[ "$STATUS" -eq 0 ]]; then
  echo "Both training runs finished. Logs are in ${LOG_DIR}"
else
  echo "At least one training run failed. Check ${LOG_DIR}"
fi

exit "$STATUS"
