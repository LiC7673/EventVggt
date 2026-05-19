#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
BASE_PORT="${BASE_PORT:-29900}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-3}"
NUM_VIEWS="${NUM_VIEWS:-6}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
OMP_THREADS="${OMP_THREADS:-1}"
LDR_MODE="${LDR_MODE:-common}"
LDR_LIST="${LDR_LIST:-auto_detect}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

if [[ "$LDR_LIST" == "auto_detect" ]]; then
  echo "Detecting LDR levels from ${DATA_ROOT} ..."
  LDR_CSV="$(
    cd "$ROOT_DIR"
    "$PYTHON_BIN" fine_rgb/detect_ldr_levels.py \
      --root "$DATA_ROOT" \
      --num-views "$NUM_VIEWS" \
      --active-scene-count "$ACTIVE_SCENE_COUNT" \
      --mode "$LDR_MODE" \
      --format csv
  )"
else
  LDR_CSV="$LDR_LIST"
fi

IFS=',' read -r -a LDR_IDS <<< "$LDR_CSV"
if [[ "${#LDR_IDS[@]}" -eq 0 || -z "${LDR_IDS[0]}" ]]; then
  echo "No LDR levels found. Set LDR_LIST=ev_2,ev_5 or check DATA_ROOT."
  exit 1
fi

if (( ${#GPUS[@]} == 0 )); then
  echo "No GPUs configured."
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
LOG_DIR="${ROOT_DIR}/ablation_logs/fine_rgb_ldr_${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "Root: ${ROOT_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "LDR ids: ${LDR_IDS[*]}"
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
  local ldr_idx ldr_id safe_ldr log_file
  local worker_status=0

  for ((ldr_idx = worker_idx; ldr_idx < ${#LDR_IDS[@]}; ldr_idx += ${#GPU_GROUPS[@]})); do
    ldr_id="${LDR_IDS[$ldr_idx]}"
    safe_ldr="${ldr_id//\//_}"
    safe_ldr="${safe_ldr// /_}"
    log_file="${LOG_DIR}/${ldr_idx}_fine_rgb_${safe_ldr}_gpus_${gpu_group//,/}.log"

    echo "[GPUs ${gpu_group}] start RGB LDR ${ldr_id} -> ${log_file}"
    if (
      cd "$ROOT_DIR"
      CUDA_VISIBLE_DEVICES="$gpu_group" \
      HYDRA_FULL_ERROR=1 \
      OMP_NUM_THREADS="$OMP_THREADS" \
      MKL_NUM_THREADS="$OMP_THREADS" \
      "$ACCELERATE_BIN" launch \
        --multi_gpu \
        --num_processes "$GPUS_PER_JOB" \
        --num_machines 1 \
        --main_process_port "$port" \
        fine_rgb/finetune_rgb_ldr.py \
        data.root="$DATA_ROOT" \
        data.active_scene_count="$ACTIVE_SCENE_COUNT" \
        data.num_views="$NUM_VIEWS" \
        data.ldr_event_id="$ldr_id" \
        exp_name="fine_rgb_${safe_ldr}" \
        num_workers="$NUM_WORKERS" \
        pin_mem="$PIN_MEM" \
        "$@"
    ) >"$log_file" 2>&1; then
      echo "[GPUs ${gpu_group}] done  RGB LDR ${ldr_id}"
    else
      worker_status=1
      echo "[GPUs ${gpu_group}] fail  RGB LDR ${ldr_id} (see ${log_file})"
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
  echo "All pure-RGB LDR jobs finished. Logs are in ${LOG_DIR}"
else
  echo "Some pure-RGB LDR jobs failed. Check logs in ${LOG_DIR}"
fi

exit "$STATUS"
