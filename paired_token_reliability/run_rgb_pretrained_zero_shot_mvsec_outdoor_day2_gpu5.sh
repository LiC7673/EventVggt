#!/usr/bin/env bash
# Pure, pretrained RGB StreamVGGT zero-shot evaluation on MVSEC outdoor_day2.
# No event input, no HDR-Diff frames, and no parameter update.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-5}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
MVSEC_RAW_ROOT="${MVSEC_RAW_ROOT:-/data1/lzh/dataset/MVSEC_raw}"
MVSEC_H5_ROOT="${MVSEC_H5_ROOT:-${MVSEC_RAW_ROOT}/converted_hdf5}"
OUTPUT="${OUTPUT:-exp_f/rgb_pretrained_zero_shot_mvsec_outdoor_day2_scale20_gpu5}"

if [[ ! -f "${PRETRAINED}" ]]; then
  echo "Missing pretrained RGB checkpoint: ${PRETRAINED}" >&2
  exit 2
fi

if [[ ! -s "${MVSEC_H5_ROOT}/outdoor_day2_data.hdf5" || \
      ! -s "${MVSEC_H5_ROOT}/outdoor_day2_gt.hdf5" ]]; then
  echo "[MVSEC RGB] converting outdoor_day2 ROS bags with rosbags"
  python -m paired_token_reliability.convert_mvsec_rosbag_to_hdf5 \
    --root "${MVSEC_RAW_ROOT}" \
    --output "${MVSEC_H5_ROOT}" \
    --sequences outdoor_day2
fi

mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[MVSEC RGB zero-shot] checkpoint=${PRETRAINED}"
echo "[MVSEC RGB zero-shot] RGB=original APS frames stored in converted HDF5"
echo "[MVSEC RGB zero-shot] event=NONE; HDR-Diff=NONE; parameter updates=NONE"
echo "[MVSEC RGB zero-shot] fixed scale from first ${SCALE_CALIBRATION_FRAMES:-20} unique frames"

python -m paired_token_reliability.finetune_rgb_real_dataset \
  --dataset mvsec \
  --root "${MVSEC_H5_ROOT}" \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --train-sequence outdoor_day2 \
  --test-sequence outdoor_day2 \
  --epochs 0 \
  --max-train-steps 0 \
  --max-test-batches "${MAX_TEST_BATCHES:-0}" \
  --depth-scale "${DEPTH_SCALE:-1.0}" \
  --scale-calibration-frames "${SCALE_CALIBRATION_FRAMES:-20}" \
  --num-views "${NUM_VIEWS:-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --visualize-every "${VISUALIZE_EVERY:-10}" \
  --max-visualizations "${MAX_VISUALIZATIONS:-30}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/zero_shot_test.log"

echo "[MVSEC RGB zero-shot] JSON: ${OUTPUT}/final_test_metrics.json"
echo "[MVSEC RGB zero-shot] TXT:  ${OUTPUT}/metrics.txt"
echo "[MVSEC RGB zero-shot] VIS:  ${OUTPUT}/final_test_visualizations"
