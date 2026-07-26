#!/usr/bin/env bash
# Zero-shot evaluation on MVSEC outdoor_day2.
# The synthetic-data checkpoint is loaded directly; no MVSEC fine-tuning occurs.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RAW_ROOT="${MVSEC_RAW_ROOT:-/data1/lzh/dataset/MVSEC_raw}"
H5_ROOT="${MVSEC_H5_ROOT:-${RAW_ROOT}/converted_hdf5}"
CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_refiner_first_1k_then_joint_gpu4/checkpoint-adapter-best.pth}"
OUTPUT="${OUTPUT:-exp_f/zero_shot_mvsec_outdoor_day2_synthetic_best_scale20_gpu5}"
GPU="${GPU:-5}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing synthetic-data checkpoint: ${CHECKPOINT}" >&2
  echo "Specify it explicitly, for example:" >&2
  echo "  CHECKPOINT=/absolute/path/checkpoint-adapter-best.pth bash $0" >&2
  exit 2
fi

# Convert only outdoor_day2 when the cached HDF5 files are unavailable.
if [[ ! -s "${H5_ROOT}/outdoor_day2_data.hdf5" || \
      ! -s "${H5_ROOT}/outdoor_day2_gt.hdf5" ]]; then
  echo "[MVSEC] converting outdoor_day2 ROS bags with rosbags -> ${H5_ROOT}"
  python -m paired_token_reliability.convert_mvsec_rosbag_to_hdf5 \
    --root "${RAW_ROOT}" \
    --output "${H5_ROOT}" \
    --sequences outdoor_day2
fi

mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[MVSEC zero-shot] checkpoint=${CHECKPOINT}"
echo "[MVSEC zero-shot] sequence=outdoor_day2; epochs=0; parameter updates=NONE"
echo "[MVSEC zero-shot] fixed depth scale from first ${SCALE_CALIBRATION_FRAMES:-20} unique frames"

# finetune_refiner_first_mvsec also provides the common MVSEC evaluator.
# epochs=0 bypasses its entire training loop and directly executes final test.
python -m paired_token_reliability.finetune_refiner_first_mvsec \
  --checkpoint "${CHECKPOINT}" \
  --root "${H5_ROOT}" \
  --output "${OUTPUT}" \
  --train-sequence outdoor_day2 \
  --test-sequences outdoor_day2 \
  --epochs 0 \
  --max-train-steps 0 \
  --max-test-batches "${MAX_TEST_BATCHES:-0}" \
  --scale-calibration-frames "${SCALE_CALIBRATION_FRAMES:-20}" \
  --scale-calibration-pixels-per-frame \
    "${SCALE_CALIBRATION_PIXELS_PER_FRAME:-10000}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --num-views "${NUM_VIEWS:-4}" \
  --event-bins "${EVENT_BINS:-5}" \
  --max-depth "${MAX_DEPTH:-80}" \
  --visualize-every "${VISUALIZE_EVERY:-10}" \
  --max-visualizations "${MAX_VISUALIZATIONS:-30}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/zero_shot_outdoor_day2.log"

echo "[MVSEC zero-shot] metrics: ${OUTPUT}/final_test_metrics.json"
echo "[MVSEC zero-shot] visuals: ${OUTPUT}/final_test_visualizations/outdoor_day2"
