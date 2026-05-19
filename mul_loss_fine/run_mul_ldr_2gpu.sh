#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPU_LIST="${GPU_LIST:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29800}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
OMP_THREADS="${OMP_THREADS:-1}"

# Comma-separated LDR folders. Use names as in the dataset, for example:
# LDR_TRAIN_IDS=ev_2,ev_5,ev_10
LDR_TRAIN_IDS="${LDR_TRAIN_IDS:-ev_2,ev_5,ev_10}"
EVAL_LDR_ID="${EVAL_LDR_ID:-auto}"
EXPOSURES_PER_SAMPLE="${EXPOSURES_PER_SAMPLE:-2}"
SCENES_PER_BATCH="${SCENES_PER_BATCH:-1}"
TEST_VIS_BATCHES="${TEST_VIS_BATCHES:-2}"
TEST_VIS_VIEWS="${TEST_VIS_VIEWS:-6}"
EXP_NAME="${EXP_NAME:-mul_ldr_event}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/ablation_logs/mul_ldr_${RUN_ID}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_gpus_${GPU_LIST//,/}.log"

echo "Root: ${ROOT_DIR}"
echo "GPUs: ${GPU_LIST}"
echo "Num processes: ${NUM_PROCESSES}"
echo "LDR train ids: ${LDR_TRAIN_IDS}"
echo "Eval LDR id: ${EVAL_LDR_ID}"
echo "Exposures per sample: ${EXPOSURES_PER_SAMPLE}"
echo "Scenes per batch: ${SCENES_PER_BATCH}"
echo "Test visualization batches: ${TEST_VIS_BATCHES}"
echo "Logs: ${LOG_FILE}"
echo "Extra Hydra args: $*"
echo

(
  cd "$ROOT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  HYDRA_FULL_ERROR=1 \
  OMP_NUM_THREADS="$OMP_THREADS" \
  MKL_NUM_THREADS="$OMP_THREADS" \
  "$ACCELERATE_BIN" launch \
    --multi_gpu \
    --num_processes "$NUM_PROCESSES" \
    --num_machines 1 \
    --main_process_port "$MAIN_PROCESS_PORT" \
    mul_loss_fine/finetune_mul_ldr_event.py \
    exp_name="$EXP_NAME" \
    num_workers="$NUM_WORKERS" \
    pin_mem="$PIN_MEM" \
    data.ldr_event_id=random \
    +data.eval_ldr_event_id="$EVAL_LDR_ID" \
    +data.mul_ldr_train_ids="[$LDR_TRAIN_IDS]" \
    +data.mul_ldr_exposures_per_sample="$EXPOSURES_PER_SAMPLE" \
    +data.mul_ldr_scenes_per_batch="$SCENES_PER_BATCH" \
    +vis.test_max_batches="$TEST_VIS_BATCHES" \
    +vis.test_num_views="$TEST_VIS_VIEWS" \
    "$@"
) >"$LOG_FILE" 2>&1

echo "Mul-LDR job finished. Log: ${LOG_FILE}"
