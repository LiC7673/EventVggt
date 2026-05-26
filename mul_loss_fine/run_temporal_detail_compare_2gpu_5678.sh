#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
# Physical GPUs 5-8 correspond to zero-based CUDA IDs 4-7.
UNIFORM_GPUS="${UNIFORM_GPUS:-4,5}"
DETAIL_GPUS="${DETAIL_GPUS:-6,7}"
BASE_PORT="${BASE_PORT:-29860}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
OMP_THREADS="${OMP_THREADS:-1}"

UNIFORM_SCRIPT="mul_loss_fine/finetune_mul_loss_detail_gt_uniform.py"
DETAIL_SCRIPT="mul_loss_fine/finetune_mul_loss_detail_gt_temporal_detail.py"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/ablation_logs/temporal_detail_compare_${RUN_ID}"
mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"
  local script="$2"
  local gpu_group="$3"
  local port="$4"
  shift 4
  echo "[CUDA ${gpu_group}] start ${name}"
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
  ) >"${LOG_DIR}/${name}_gpus_${gpu_group//,/}.log" 2>&1
}

echo "Uniform GT-detail control: CUDA ${UNIFORM_GPUS}"
echo "Dense temporal event detail: CUDA ${DETAIL_GPUS}"
echo "Logs: ${LOG_DIR}"

run_one "detail_gt_uniform" "$UNIFORM_SCRIPT" "$UNIFORM_GPUS" "$BASE_PORT" "$@" &
UNIFORM_PID="$!"
run_one "detail_gt_temporal_detail" "$DETAIL_SCRIPT" "$DETAIL_GPUS" "$((BASE_PORT + 1))" "$@" &
DETAIL_PID="$!"

STATUS=0
wait "$UNIFORM_PID" || STATUS=1
wait "$DETAIL_PID" || STATUS=1
exit "$STATUS"
