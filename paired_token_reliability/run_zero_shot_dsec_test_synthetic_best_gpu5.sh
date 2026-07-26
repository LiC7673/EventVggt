#!/usr/bin/env bash
# Zero-shot evaluation on the DSEC test split.
# Loads the synthetic-data checkpoint directly and performs no DSEC fine-tuning.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_refiner_first_1k_then_joint_gpu4/checkpoint-adapter-best.pth}"
OUTPUT="${OUTPUT:-exp_f/zero_shot_dsec_test_synthetic_best_gpu5}"
GPU="${GPU:-5}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing synthetic-data checkpoint: ${CHECKPOINT}" >&2
  echo "Specify it explicitly, for example:" >&2
  echo "  CHECKPOINT=/absolute/path/checkpoint-adapter-best.pth bash $0" >&2
  exit 2
fi

if [[ ! -d "${DSEC_ROOT}/test" ]]; then
  echo "Missing DSEC test directory: ${DSEC_ROOT}/test" >&2
  echo "Set DSEC_ROOT to the directory containing train/ and test/." >&2
  exit 2
fi

mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[DSEC zero-shot] checkpoint=${CHECKPOINT}"
echo "[DSEC zero-shot] root=${DSEC_ROOT}"
echo "[DSEC zero-shot] epochs=0; parameter updates=NONE"
echo "[DSEC zero-shot] one fixed scale from first ${SCALE_CALIBRATION_FRAMES:-20} frames"

# This entry point contains the common DSEC evaluator. With epochs=0 its
# optimization loop is skipped, and the loaded synthetic checkpoint is sent
# directly through the complete DSEC test split.
python -m paired_token_reliability.finetune_refiner_first_dsec \
  --checkpoint "${CHECKPOINT}" \
  --root "${DSEC_ROOT}" \
  --output "${OUTPUT}" \
  --epochs 0 \
  --max-train-steps 0 \
  --max-test-batches "${MAX_TEST_BATCHES:-0}" \
  --scale-calibration-frames "${SCALE_CALIBRATION_FRAMES:-20}" \
  --scale-calibration-pixels-per-frame \
    "${SCALE_CALIBRATION_PIXELS_PER_FRAME:-10000}" \
  --batch-size 1 \
  --num-workers "${NUM_WORKERS:-2}" \
  --num-views "${NUM_VIEWS:-4}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --max-visualizations "${MAX_VISUALIZATIONS:-0}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/zero_shot_dsec_test.log"

echo "[DSEC zero-shot] metrics: ${OUTPUT}/final_test_metrics.json"
echo "[DSEC zero-shot] visuals: ${OUTPUT}/final_test_visualizations"
