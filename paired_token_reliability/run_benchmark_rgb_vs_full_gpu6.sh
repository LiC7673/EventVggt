#!/usr/bin/env bash
# Fair hardware comparison: untouched RGB baseline vs full cur-event method.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-exp_f/hardware_rgb_vs_full}"
RGB_CHECKPOINT="${RGB_CHECKPOINT:-ckpt/model.pt}"
FULL_CHECKPOINT="${FULL_CHECKPOINT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
SCENE="${SCENE:-DH2_Socrates and Seneca_Car_Paint_Midnight}"
EXPOSURE="${EXPOSURE:-ev_2}"
WARMUP="${WARMUP:-10}"
REPEATS="${REPEATS:-50}"
AMP="${AMP:-none}"

mkdir -p "${OUTPUT_ROOT}/logs"
[[ -f "${RGB_CHECKPOINT}" ]] || { echo "Missing RGB checkpoint: ${RGB_CHECKPOINT}" >&2; exit 2; }
[[ -f "${FULL_CHECKPOINT}" ]] || { echo "Missing full checkpoint: ${FULL_CHECKPOINT}" >&2; exit 2; }

echo "[1/2] pure RGB baseline"
python -m paired_token_reliability.benchmark_rgb_hardware \
  --checkpoint "${RGB_CHECKPOINT}" \
  --output "${OUTPUT_ROOT}/rgb.json" \
  --root "${DATA_ROOT}" --scene "${SCENE}" --exposure "${EXPOSURE}" \
  --num-views 4 --resolution 518 392 \
  --warmup "${WARMUP}" --repeats "${REPEATS}" --amp "${AMP}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/rgb.log"

echo "[2/2] full cur-event method"
python -m paired_token_reliability.benchmark_cur_event_hardware \
  --variant full \
  --checkpoint "${FULL_CHECKPOINT}" \
  --output "${OUTPUT_ROOT}/full.json" \
  --root "${DATA_ROOT}" --scene "${SCENE}" --exposure "${EXPOSURE}" \
  --num-views 4 --resolution 518 392 \
  --event-resize-bins 5 --event-resize-method voxel_linear_time \
  --warmup "${WARMUP}" --repeats "${REPEATS}" --amp "${AMP}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/full.log"

python -m paired_token_reliability.collect_rgb_full_hardware_results \
  --rgb "${OUTPUT_ROOT}/rgb.json" \
  --full "${OUTPUT_ROOT}/full.json" \
  --output-prefix "${OUTPUT_ROOT}/hardware_summary"

echo "Saved comparison to ${OUTPUT_ROOT}/hardware_summary.csv"
