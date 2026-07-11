#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-2}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
OUTPUT="${OUTPUT:-abl_event_exp/unified_geometry_contribution}"
EPOCHS_A="${EPOCHS_A:-5}"
EPOCHS_B="${EPOCHS_B:-10}"
EPOCHS_C="${EPOCHS_C:-0}"
PAIR_MODE="${PAIR_MODE:-anchor}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "${OUTPUT}/logs"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m paired_token_reliability.train_unified_geometry_contribution \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --epochs-a "${EPOCHS_A}" \
  --epochs-b "${EPOCHS_B}" \
  --epochs-c "${EPOCHS_C}" \
  --pair-mode "${PAIR_MODE}" \
  --num-workers "${NUM_WORKERS}" \
  --visualize-every-batches 40 \
  "$@" \
  2>&1 | tee "${OUTPUT}/logs/train.log"

if [[ "${RUN_EVAL}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python -m paired_token_reliability.evaluate_unified_geometry_contribution \
    --checkpoint "${OUTPUT}/checkpoint-best.pth" \
    --output-dir "${OUTPUT}/heldout_eval" \
    --num-workers "${NUM_WORKERS}" \
    --amp none \
    2>&1 | tee "${OUTPUT}/logs/evaluate.log"
fi
