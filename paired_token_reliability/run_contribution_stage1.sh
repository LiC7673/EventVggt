#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-2,3,4,5,6,7}"
CONFIG="${CONFIG:-config/finetune_event.yaml}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
OUTPUT="${OUTPUT:-abl_event_exp/event_contribution_stage1}"
EPOCHS_PROXY="${EPOCHS_PROXY:-5}"
EPOCHS_CONTRIBUTION="${EPOCHS_CONTRIBUTION:-15}"
EPOCHS_JOINT="${EPOCHS_JOINT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-100}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_NO_BRIDGE="${RUN_NO_BRIDGE:-0}"
NO_BRIDGE_OUTPUT="${NO_BRIDGE_OUTPUT:-${OUTPUT}_no_bridge}"
EXPOSURES="${EXPOSURES:-0,1,2,5,10}"
PAIR_MODE="${PAIR_MODE:-anchor}"
VISUALIZE_EVERY_BATCHES="${VISUALIZE_EVERY_BATCHES:-40}"

mkdir -p "${OUTPUT}/logs"

echo "[0/2] architectural/unit tests"
python -m paired_token_reliability.test_contribution_stage1 \
  2>&1 | tee "${OUTPUT}/logs/unit_tests.log"

if [[ "${RUN_NO_BRIDGE}" == "1" ]]; then
  mkdir -p "${NO_BRIDGE_OUTPUT}/logs"
  echo "[ablation] train ContributionNet without the bridge mask"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m paired_token_reliability.train_contribution_stage1 \
    --config "${CONFIG}" \
    --pretrained "${PRETRAINED}" \
    --output "${NO_BRIDGE_OUTPUT}" \
    --phase all \
    --epochs-proxy "${EPOCHS_PROXY}" \
    --epochs-contribution "${EPOCHS_CONTRIBUTION}" \
    --epochs-joint "${EPOCHS_JOINT}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-train-batches "${MAX_TRAIN_BATCHES}" \
    --max-val-batches "${MAX_VAL_BATCHES}" \
    --mixed-precision "${MIXED_PRECISION}" \
    --supervision-region event_support \
    --exposures "${EXPOSURES}" \
    --pair-mode "${PAIR_MODE}" \
    --visualize-every-batches "${VISUALIZE_EVERY_BATCHES}" \
    "$@" \
    2>&1 | tee "${NO_BRIDGE_OUTPUT}/logs/train.log"
fi

echo "[1/2] train Multi-LDR event contribution on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m paired_token_reliability.train_contribution_stage1 \
  --config "${CONFIG}" \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --phase all \
  --epochs-proxy "${EPOCHS_PROXY}" \
  --epochs-contribution "${EPOCHS_CONTRIBUTION}" \
  --epochs-joint "${EPOCHS_JOINT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --mixed-precision "${MIXED_PRECISION}" \
  --exposures "${EXPOSURES}" \
  --pair-mode "${PAIR_MODE}" \
  --visualize-every-batches "${VISUALIZE_EVERY_BATCHES}" \
  "$@" \
  2>&1 | tee "${OUTPUT}/logs/train.log"

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[2/2] held-out contribution counterfactuals"
  EVAL_EXTRA=()
  if [[ "${RUN_NO_BRIDGE}" == "1" ]]; then
    EVAL_EXTRA+=(--no-bridge-checkpoint "${NO_BRIDGE_OUTPUT}/checkpoint-best.pth")
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python -m paired_token_reliability.evaluate_contribution_stage1 \
    --checkpoint "${OUTPUT}/checkpoint-best.pth" \
    --rgb-checkpoint "${PRETRAINED}" \
    --output "${OUTPUT}/stage1_causal_eval.json" \
    --num-workers "${NUM_WORKERS}" \
    "${EVAL_EXTRA[@]}" \
    2>&1 | tee "${OUTPUT}/logs/evaluate.log"
fi
