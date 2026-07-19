#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

RAW_ROOT="${MVSEC_RAW_ROOT:-/data1/lzh/dataset/MVSEC_raw}"
H5_ROOT="${MVSEC_H5_ROOT:-${RAW_ROOT}/converted_hdf5}"
OUTPUT="${OUTPUT:-exp_f/mvsec_outdoor_day1_finetune_day2_test_gpu5}"
CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_refiner_first_1k_then_joint_gpu4/checkpoint-adapter-best.pth}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing source checkpoint: ${CHECKPOINT}" >&2
  echo "Run with CHECKPOINT=/absolute/path/to/checkpoint.pth" >&2
  exit 2
fi

need_convert=0
for sequence in outdoor_day1 outdoor_day2; do
  [[ -s "${H5_ROOT}/${sequence}_data.hdf5" && -s "${H5_ROOT}/${sequence}_gt.hdf5" ]] || need_convert=1
done
if [[ "${need_convert}" == 1 ]]; then
  echo "Converting outdoor_day1/day2 ROS bags with rosbags -> ${H5_ROOT}"
  python -m paired_token_reliability.convert_mvsec_rosbag_to_hdf5 \
    --root "${RAW_ROOT}" --output "${H5_ROOT}" \
    --sequences outdoor_day1 outdoor_day2
fi

mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m paired_token_reliability.finetune_refiner_first_mvsec \
  --checkpoint "${CHECKPOINT}" \
  --root "${H5_ROOT}" \
  --output "${OUTPUT}" \
  --train-sequence outdoor_day1 \
  --test-sequences outdoor_day2 \
  --epochs "${EPOCHS:-2}" \
  --max-train-steps "${MAX_TRAIN_STEPS:-1500}" \
  --lr "${LR:-1e-5}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --visualize-every "${VISUALIZE_EVERY:-10}" \
  --max-visualizations "${MAX_VISUALIZATIONS:-30}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/train_day1_test_day2.log"
