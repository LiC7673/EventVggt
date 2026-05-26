#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
# CUDA device indices are zero-based: IDs 4,5,6,7 are physical GPUs 5-8.
UNIFORM_GPUS="${UNIFORM_GPUS:-4,5}"
TEMPORAL_GPUS="${TEMPORAL_GPUS:-6,7}"
BASE_PORT="${BASE_PORT:-29840}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
OMP_THREADS="${OMP_THREADS:-1}"

UNIFORM_SCRIPT="mul_loss_fine/finetune_mul_loss_detail_gt_uniform.py"
TEMPORAL_SCRIPT="mul_loss_fine/finetune_mul_loss_detail_gt_temporal_bins.py"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/ablation_logs/temporal_bins_compare_${RUN_ID}"
mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"
  local script="$2"
  local gpu_group="$3"
  local port="$4"
  shift 4
  local log_file="${LOG_DIR}/${name}_gpus_${gpu_group//,/}.log"

  echo "[CUDA ${gpu_group}] start ${name} -> ${log_file}"
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

echo "Uniform GT-detail control: CUDA devices ${UNIFORM_GPUS}"
echo "Temporal-bin event input: CUDA devices ${TEMPORAL_GPUS}"
echo "Logs: ${LOG_DIR}"

run_one "detail_gt_uniform" "$UNIFORM_SCRIPT" "$UNIFORM_GPUS" "$BASE_PORT" "$@" &
UNIFORM_PID="$!"
run_one "detail_gt_temporal_bins" "$TEMPORAL_SCRIPT" "$TEMPORAL_GPUS" "$((BASE_PORT + 1))" "$@" &
TEMPORAL_PID="$!"

STATUS=0
if wait "$UNIFORM_PID"; then
  echo "[CUDA ${UNIFORM_GPUS}] done detail_gt_uniform"
else
  echo "[CUDA ${UNIFORM_GPUS}] fail detail_gt_uniform"
  STATUS=1
fi
if wait "$TEMPORAL_PID"; then
  echo "[CUDA ${TEMPORAL_GPUS}] done detail_gt_temporal_bins"
else
  echo "[CUDA ${TEMPORAL_GPUS}] fail detail_gt_temporal_bins"
  STATUS=1
fi

exit "$STATUS"
