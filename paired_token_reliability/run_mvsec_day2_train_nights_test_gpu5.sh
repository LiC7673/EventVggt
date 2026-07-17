#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
RAW_ROOT="${MVSEC_RAW_ROOT:-/data1/lzh/dataset/MVSEC_raw}"
H5_ROOT="${MVSEC_H5_ROOT:-${RAW_ROOT}/converted_hdf5}"
OUTPUT="${OUTPUT:-exp_f/mvsec_outdoor_day2_to_night_gpu5}"
CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_refiner_first_1k_then_joint_gpu4/checkpoint-adapter-best.pth}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing best checkpoint: ${CHECKPOINT}" >&2
  echo "Run with CHECKPOINT=/absolute/path/to/your/best.pth" >&2
  exit 2
fi
if [[ ! -f "${H5_ROOT}/outdoor_day2_data.hdf5" ]]; then
  echo "Converting raw MVSEC ROS bags to ${H5_ROOT} ..."
  python -m paired_token_reliability.convert_mvsec_rosbag_to_hdf5 --root "${RAW_ROOT}" --output "${H5_ROOT}"
fi
mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU:-5}" PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python -m paired_token_reliability.finetune_refiner_first_mvsec \
  --checkpoint "${CHECKPOINT}" --root "${H5_ROOT}" --output "${OUTPUT}" \
  --train-sequence outdoor_day2 \
  --test-sequences outdoor_night1 outdoor_night2 outdoor_night3 \
  --epochs "${EPOCHS:-2}" --max-train-steps "${MAX_TRAIN_STEPS:-1500}" \
  --lr "${LR:-1e-5}" --num-workers "${NUM_WORKERS:-2}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/train_and_test.log"
