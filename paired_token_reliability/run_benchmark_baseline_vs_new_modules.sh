#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-exp_f/hardware_baseline_vs_new_modules}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-ckpt/model.pt}"
METHOD_CHECKPOINT="${METHOD_CHECKPOINT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
SCENE="${SCENE:-DH2_Socrates and Seneca_Car_Paint_Midnight}"
EXPOSURE="${EXPOSURE:-ev_2}"
NUM_VIEWS="${NUM_VIEWS:-4}"
WIDTH="${WIDTH:-518}"
HEIGHT="${HEIGHT:-392}"
EVENT_BINS="${EVENT_BINS:-5}"
WARMUP="${WARMUP:-10}"
REPEATS="${REPEATS:-50}"
AMP="${AMP:-none}"

mkdir -p "${OUTPUT_ROOT}/logs"
[[ -f "${BASELINE_CHECKPOINT}" ]] || { echo "Missing baseline checkpoint: ${BASELINE_CHECKPOINT}" >&2; exit 2; }
[[ -f "${METHOD_CHECKPOINT}" ]] || { echo "Missing method checkpoint: ${METHOD_CHECKPOINT}" >&2; exit 2; }

echo "[1/2] RGB baseline: forward latency and CUDA memory"
python -m paired_token_reliability.benchmark_rgb_hardware \
  --checkpoint "${BASELINE_CHECKPOINT}" \
  --output "${OUTPUT_ROOT}/baseline.json" \
  --root "${DATA_ROOT}" --scene "${SCENE}" --exposure "${EXPOSURE}" \
  --num-views "${NUM_VIEWS}" --resolution "${WIDTH}" "${HEIGHT}" \
  --warmup "${WARMUP}" --repeats "${REPEATS}" --amp "${AMP}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/baseline.log"

echo "[2/2] RGB + proposed modules: forward latency and CUDA memory"
python -m paired_token_reliability.benchmark_cur_event_hardware \
  --variant full \
  --checkpoint "${METHOD_CHECKPOINT}" \
  --output "${OUTPUT_ROOT}/new_modules.json" \
  --root "${DATA_ROOT}" --scene "${SCENE}" --exposure "${EXPOSURE}" \
  --num-views "${NUM_VIEWS}" --resolution "${WIDTH}" "${HEIGHT}" \
  --event-resize-bins "${EVENT_BINS}" \
  --event-resize-method voxel_linear_time \
  --warmup "${WARMUP}" --repeats "${REPEATS}" --amp "${AMP}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/new_modules.log"

python -m paired_token_reliability.collect_baseline_vs_extension_hardware \
  --baseline "${OUTPUT_ROOT}/baseline.json" \
  --extension "${OUTPUT_ROOT}/new_modules.json" \
  --output-prefix "${OUTPUT_ROOT}/hardware_comparison"

echo "Done: ${OUTPUT_ROOT}/hardware_comparison.csv"
