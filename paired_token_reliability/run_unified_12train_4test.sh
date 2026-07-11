#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-2}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
OUTPUT="${OUTPUT:-abl_event_exp/unified_geometry_contribution_12train_4test}"
EPOCHS_A="${EPOCHS_A:-5}"
EPOCHS_B="${EPOCHS_B:-10}"
EPOCHS_C="${EPOCHS_C:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_VIEWS="${NUM_VIEWS:-4}"
TRAIN_INITIAL_SCENE_IDX="${TRAIN_INITIAL_SCENE_IDX:-0}"
TRAIN_SCENE_COUNT="${TRAIN_SCENE_COUNT:-12}"
TEST_INITIAL_SCENE_IDX="${TEST_INITIAL_SCENE_IDX:-12}"
TEST_SCENE_COUNT="${TEST_SCENE_COUNT:-4}"
HELDOUT_TEST_FRAME_COUNT="${HELDOUT_TEST_FRAME_COUNT:-120}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${OUTPUT}/logs"

python -m paired_token_reliability.train_unified_geometry_contribution \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --epochs-a "${EPOCHS_A}" \
  --epochs-b "${EPOCHS_B}" \
  --epochs-c "${EPOCHS_C}" \
  --num-workers "${NUM_WORKERS}" \
  --visualize-every-batches 40 \
  "data.num_views=${NUM_VIEWS}" \
  "+data.train_initial_scene_idx=${TRAIN_INITIAL_SCENE_IDX}" \
  "+data.train_scene_count=${TRAIN_SCENE_COUNT}" \
  "+data.train_holdout_frame_count=0" \
  "+data.test_initial_scene_idx=${TEST_INITIAL_SCENE_IDX}" \
  "+data.test_scene_count=${TEST_SCENE_COUNT}" \
  "+data.heldout_test_frame_count=${HELDOUT_TEST_FRAME_COUNT}" \
  "$@" \
  2>&1 | tee "${OUTPUT}/logs/train.log"

