#!/usr/bin/env bash
# Pure-RGB StreamVGGT adaptation/evaluation using HDR-Diff restored DSEC frames.
# No event tensor or event module is passed to the model.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-5}"
DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
OUTPUT="${OUTPUT:-exp_f/hdr_diff_rgb_finetune_dsec_gpu5}"
RGB_SUBDIR="${RGB_SUBDIR:-hdr_diff_images/event_aligned}"

if [[ ! -f "${PRETRAINED}" ]]; then
  echo "Missing RGB pretrained checkpoint: ${PRETRAINED}" >&2
  exit 2
fi
if [[ ! -d "${DSEC_ROOT}/test" ]]; then
  echo "Missing DSEC root/test: ${DSEC_ROOT}/test" >&2
  exit 2
fi

mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[HDR-Diff RGB] input subdirectory=${RGB_SUBDIR}"
echo "[HDR-Diff RGB] event input=NONE"

python -m paired_token_reliability.finetune_rgb_real_dataset \
  --dataset dsec \
  --root "${DSEC_ROOT}" \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --dsec-rgb-subdir "${RGB_SUBDIR}" \
  --epochs "${EPOCHS:-2}" \
  --max-train-steps "${MAX_TRAIN_STEPS:-1500}" \
  --max-test-batches "${MAX_TEST_BATCHES:-0}" \
  --lr-head "${LR_HEAD:-2e-5}" \
  --lr-backbone "${LR_BACKBONE:-2e-6}" \
  --unfreeze-last-blocks "${UNFREEZE_LAST_BLOCKS:-2}" \
  --num-views "${NUM_VIEWS:-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --max-visualizations "${MAX_VISUALIZATIONS:-0}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/train_and_test.log"

echo "[HDR-Diff RGB] JSON: ${OUTPUT}/final_test_metrics.json"
echo "[HDR-Diff RGB] TXT:  ${OUTPUT}/metrics.txt"
echo "[HDR-Diff RGB] VIS:  ${OUTPUT}/final_test_visualizations"
