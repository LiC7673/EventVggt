#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPUS="${GPUS:-${GPU:-2}}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC="${#GPU_ARRAY[@]}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
fi
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
EXP_NAME="${EXP_NAME:-unified_geometry_contribution_12train_4test}"
OUTPUT="${OUTPUT:-exp/${EXP_NAME}}"
EPOCHS_A="${EPOCHS_A:-2}"
EPOCHS_B="${EPOCHS_B:-10}"
EPOCHS_C="${EPOCHS_C:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_VIEWS="${NUM_VIEWS:-4}"
TRAIN_INITIAL_SCENE_IDX="${TRAIN_INITIAL_SCENE_IDX:-0}"
TRAIN_SCENE_COUNT="${TRAIN_SCENE_COUNT:-12}"
TEST_INITIAL_SCENE_IDX="${TEST_INITIAL_SCENE_IDX:-12}"
TEST_SCENE_COUNT="${TEST_SCENE_COUNT:-4}"
HELDOUT_TEST_FRAME_COUNT="${HELDOUT_TEST_FRAME_COUNT:-120}"
DECOMPOSITION_EVENT_ROOT="${DECOMPOSITION_EVENT_ROOT:-events_additive}"
DECOMPOSITION_GEO_BRANCH="${DECOMPOSITION_GEO_BRANCH:-geometry_motion}"
DECOMPOSITION_FULL_BRANCH="${DECOMPOSITION_FULL_BRANCH:-full}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT}/logs"

python -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
  -m paired_token_reliability.train_unified_geometry_contribution \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --epochs-a "${EPOCHS_A}" \
  --epochs-b "${EPOCHS_B}" \
  --epochs-c "${EPOCHS_C}" \
  --require-full-event-phase-b \
  --num-workers "${NUM_WORKERS}" \
  --visualize-every-batches "${TRAIN_VIS_EVERY:-40}" \
  --visualize-val-every-batches "${VAL_VIS_EVERY:-20}" \
  "data.num_views=${NUM_VIEWS}" \
  "model.head_frames_chunk_size=${HEAD_CHUNK:-1}" \
  "data.train_initial_scene_idx=${TRAIN_INITIAL_SCENE_IDX}" \
  "data.train_scene_count=${TRAIN_SCENE_COUNT}" \
  "data.train_holdout_frame_count=0" \
  "data.test_initial_scene_idx=${TEST_INITIAL_SCENE_IDX}" \
  "data.test_scene_count=${TEST_SCENE_COUNT}" \
  "data.heldout_test_frame_count=${HELDOUT_TEST_FRAME_COUNT}" \
  "data.event_source_mode=decomposition_full" \
  "data.decomposition_supervision=true" \
  "data.decomposition_event_root=${DECOMPOSITION_EVENT_ROOT}" \
  "data.decomposition_geo_branch=${DECOMPOSITION_GEO_BRANCH}" \
  "data.decomposition_full_branch=${DECOMPOSITION_FULL_BRANCH}" \
  "$@" \
  2>&1 | tee "${OUTPUT}/logs/train.log"

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  CHECKPOINT="${OUTPUT}/checkpoint-best.pth" OUTPUT_DIR="${OUTPUT}/test_all_exposures" \
  GPU="${GPU_ARRAY[0]}" NUM_VIEWS="${NUM_VIEWS}" \
  bash paired_token_reliability/run_unified_all_exposures_eval.sh \
    2>&1 | tee "${OUTPUT}/logs/evaluate_all_exposures.log"
fi
