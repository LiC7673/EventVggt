#!/usr/bin/env bash
# Pure-RGB DSEC zero-shot with the same leading-20-frame fixed-scale protocol
# used by the event method and HDR-Diff baseline.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-5}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
DSEC_ROOT="${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
OUTPUT="${OUTPUT:-exp_f/dsec_rgb_pretrained_no_finetune_gpu5}"

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

echo "[DSEC RGB zero-shot] normal RGB input; event input=NONE"
echo "[DSEC RGB zero-shot] one fixed scale from first ${SCALE_CALIBRATION_FRAMES:-20} test frames"

python -m paired_token_reliability.finetune_rgb_real_dataset \
  --dataset dsec \
  --root "${DSEC_ROOT}" \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --epochs 0 \
  --max-train-steps 0 \
  --max-test-batches "${MAX_TEST_BATCHES:-0}" \
  --depth-scale "${DEPTH_SCALE:-1.0}" \
  --scale-calibration-frames "${SCALE_CALIBRATION_FRAMES:-20}" \
  --num-views "${NUM_VIEWS:-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --visualize-every "${VISUALIZE_EVERY:-10}" \
  --max-visualizations "${MAX_VISUALIZATIONS:-40}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/zero_shot_test.log"

echo "[DSEC RGB zero-shot] JSON: ${OUTPUT}/final_test_metrics.json"
echo "[DSEC RGB zero-shot] TXT:  ${OUTPUT}/metrics.txt"
echo "[DSEC RGB zero-shot] VIS:  ${OUTPUT}/final_test_visualizations"
