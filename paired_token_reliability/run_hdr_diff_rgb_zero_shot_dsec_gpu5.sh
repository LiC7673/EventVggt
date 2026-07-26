#!/usr/bin/env bash
# Zero-shot pure-RGB evaluation of HDR-Diff restored frames on DSEC.
# No event tensor is forwarded and no model parameter is updated.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-5}"
DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
RGB_SUBDIR="${RGB_SUBDIR:-hdr_diff_images/event_aligned}"
OUTPUT="${OUTPUT:-exp_f/hdr_diff_rgb_zero_shot_dsec_gpu5}"

if [[ ! -f "${PRETRAINED}" ]]; then
  echo "Missing pretrained RGB checkpoint: ${PRETRAINED}" >&2
  exit 2
fi
if [[ ! -d "${DSEC_ROOT}/test" ]]; then
  echo "Missing DSEC test split: ${DSEC_ROOT}/test" >&2
  exit 2
fi

mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[HDR-Diff RGB zero-shot] checkpoint=${PRETRAINED}"
echo "[HDR-Diff RGB zero-shot] input=${DSEC_ROOT}/test/*/${RGB_SUBDIR}"
echo "[HDR-Diff RGB zero-shot] epochs=0; parameter updates=NONE; event input=NONE"

python -m paired_token_reliability.finetune_rgb_real_dataset \
  --dataset dsec \
  --root "${DSEC_ROOT}" \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --dsec-rgb-subdir "${RGB_SUBDIR}" \
  --epochs 0 \
  --max-train-steps 0 \
  --max-test-batches "${MAX_TEST_BATCHES:-0}" \
  --depth-scale "${DEPTH_SCALE:-1.0}" \
  --scale-calibration-frames "${SCALE_CALIBRATION_FRAMES:-0}" \
  --num-views "${NUM_VIEWS:-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --max-visualizations "${MAX_VISUALIZATIONS:-0}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/zero_shot_test.log"

echo "[HDR-Diff RGB zero-shot] JSON: ${OUTPUT}/final_test_metrics.json"
echo "[HDR-Diff RGB zero-shot] TXT:  ${OUTPUT}/metrics.txt"
echo "[HDR-Diff RGB zero-shot] VIS:  ${OUTPUT}/final_test_visualizations"
