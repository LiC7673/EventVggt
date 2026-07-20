#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
MVSEC_RAW_ROOT="${MVSEC_RAW_ROOT:-/data1/lzh/dataset/MVSEC_raw}"
MVSEC_H5_ROOT="${MVSEC_H5_ROOT:-${MVSEC_RAW_ROOT}/converted_hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp_f/pure_rgb_finetune_dsec_mvsec_gpu01}"

if [[ ! -f "${PRETRAINED}" ]]; then echo "Missing RGB pretrained checkpoint: ${PRETRAINED}" >&2; exit 2; fi
need_convert=0
for sequence in outdoor_day1 outdoor_day2; do
  [[ -s "${MVSEC_H5_ROOT}/${sequence}_data.hdf5" && -s "${MVSEC_H5_ROOT}/${sequence}_gt.hdf5" ]] || need_convert=1
done
if [[ "${need_convert}" == 1 ]]; then
  python -m paired_token_reliability.convert_mvsec_rosbag_to_hdf5 \
    --root "${MVSEC_RAW_ROOT}" --output "${MVSEC_H5_ROOT}" \
    --sequences outdoor_day1 outdoor_day2
fi

mkdir -p "${OUTPUT_ROOT}/dsec/logs" "${OUTPUT_ROOT}/mvsec_day1_to_day2/logs"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

(
  export CUDA_VISIBLE_DEVICES="${DSEC_GPU:-0}"
  python -m paired_token_reliability.finetune_rgb_real_dataset \
    --dataset dsec --root "${DSEC_ROOT}" --pretrained "${PRETRAINED}" \
    --output "${OUTPUT_ROOT}/dsec" --epochs "${EPOCHS:-2}" \
    --max-train-steps "${MAX_TRAIN_STEPS:-1500}" \
    --lr-head "${LR_HEAD:-2e-5}" --lr-backbone "${LR_BACKBONE:-2e-6}" \
    --num-workers "${NUM_WORKERS:-2}" \
    2>&1 | tee "${OUTPUT_ROOT}/dsec/logs/train_and_test.log"
) &
pid_dsec=$!

(
  export CUDA_VISIBLE_DEVICES="${MVSEC_GPU:-1}"
  python -m paired_token_reliability.finetune_rgb_real_dataset \
    --dataset mvsec --root "${MVSEC_H5_ROOT}" --pretrained "${PRETRAINED}" \
    --output "${OUTPUT_ROOT}/mvsec_day1_to_day2" \
    --train-sequence outdoor_day1 --test-sequence outdoor_day2 \
    --epochs "${EPOCHS:-2}" --max-train-steps "${MAX_TRAIN_STEPS:-1500}" \
    --lr-head "${LR_HEAD:-2e-5}" --lr-backbone "${LR_BACKBONE:-2e-6}" \
    --num-workers "${NUM_WORKERS:-2}" \
    2>&1 | tee "${OUTPUT_ROOT}/mvsec_day1_to_day2/logs/train_and_test.log"
) &
pid_mvsec=$!

status=0
wait "${pid_dsec}" || status=1
wait "${pid_mvsec}" || status=1
exit "${status}"
