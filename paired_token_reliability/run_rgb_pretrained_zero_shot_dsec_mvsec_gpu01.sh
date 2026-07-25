#!/usr/bin/env bash
# One-click, strictly zero-shot pure-RGB evaluation on DSEC and MVSEC.
# No event input and no parameter update are allowed.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
MVSEC_RAW_ROOT="${MVSEC_RAW_ROOT:-/data1/lzh/dataset/MVSEC_raw}"
MVSEC_H5_ROOT="${MVSEC_H5_ROOT:-${MVSEC_RAW_ROOT}/converted_hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp_f/rgb_pretrained_zero_shot_dsec_mvsec}"
DSEC_GPU="${DSEC_GPU:-1}"
MVSEC_GPU="${MVSEC_GPU:-2}"
SCALE_CALIBRATION_FRAMES="${SCALE_CALIBRATION_FRAMES:-20}"

if [[ ! -f "${PRETRAINED}" ]]; then
  echo "Missing pretrained RGB checkpoint: ${PRETRAINED}" >&2
  exit 2
fi
if [[ ! -d "${DSEC_ROOT}/test" ]]; then
  echo "Missing DSEC test split: ${DSEC_ROOT}/test" >&2
  exit 2
fi

# The MVSEC pure-RGB evaluator still needs converted image/depth/pose streams.
if [[ ! -s "${MVSEC_H5_ROOT}/outdoor_day2_data.hdf5" || \
      ! -s "${MVSEC_H5_ROOT}/outdoor_day2_gt.hdf5" ]]; then
  python -m paired_token_reliability.convert_mvsec_rosbag_to_hdf5 \
    --root "${MVSEC_RAW_ROOT}" \
    --output "${MVSEC_H5_ROOT}" \
    --sequences outdoor_day2
fi

mkdir -p "${OUTPUT_ROOT}/dsec/logs" "${OUTPUT_ROOT}/mvsec_outdoor_day2/logs"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[RGB zero-shot] DSEC GPU=${DSEC_GPU}; MVSEC GPU=${MVSEC_GPU}"
echo "[RGB zero-shot] epochs=0; event input=NONE; parameter updates=NONE"
echo "[RGB zero-shot] one fixed depth scale calibrated from first ${SCALE_CALIBRATION_FRAMES} frames"

(
  export CUDA_VISIBLE_DEVICES="${DSEC_GPU}"
  python -m paired_token_reliability.finetune_rgb_real_dataset \
    --dataset dsec \
    --root "${DSEC_ROOT}" \
    --pretrained "${PRETRAINED}" \
    --output "${OUTPUT_ROOT}/dsec" \
    --epochs 0 --max-train-steps 0 \
    --scale-calibration-frames "${SCALE_CALIBRATION_FRAMES}" \
    --max-test-batches "${DSEC_MAX_TEST_BATCHES:-0}" \
    --num-views "${NUM_VIEWS:-4}" \
    --num-workers "${NUM_WORKERS:-2}" \
    --visualize-every "${VISUALIZE_EVERY:-10}" \
    --max-visualizations "${MAX_VISUALIZATIONS:-30}" \
    2>&1 | tee "${OUTPUT_ROOT}/dsec/logs/zero_shot_test.log"
) &
dsec_pid=$!

(
  export CUDA_VISIBLE_DEVICES="${MVSEC_GPU}"
  python -m paired_token_reliability.finetune_rgb_real_dataset \
    --dataset mvsec \
    --root "${MVSEC_H5_ROOT}" \
    --pretrained "${PRETRAINED}" \
    --output "${OUTPUT_ROOT}/mvsec_outdoor_day2" \
    --train-sequence outdoor_day2 \
    --test-sequence outdoor_day2 \
    --epochs 0 --max-train-steps 0 \
    --scale-calibration-frames "${SCALE_CALIBRATION_FRAMES}" \
    --max-test-batches "${MVSEC_MAX_TEST_BATCHES:-0}" \
    --num-views "${NUM_VIEWS:-4}" \
    --num-workers "${NUM_WORKERS:-2}" \
    --visualize-every "${VISUALIZE_EVERY:-10}" \
    --max-visualizations "${MAX_VISUALIZATIONS:-30}" \
    2>&1 | tee "${OUTPUT_ROOT}/mvsec_outdoor_day2/logs/zero_shot_test.log"
) &
mvsec_pid=$!

status=0
wait "${dsec_pid}" || status=1
wait "${mvsec_pid}" || status=1

echo "[RGB zero-shot] DSEC:  ${OUTPUT_ROOT}/dsec/metrics.txt"
echo "[RGB zero-shot] MVSEC: ${OUTPUT_ROOT}/mvsec_outdoor_day2/metrics.txt"
exit "${status}"
