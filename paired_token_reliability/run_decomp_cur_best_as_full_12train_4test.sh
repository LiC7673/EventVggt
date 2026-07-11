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
EXP_NAME="${EXP_NAME:-decomp_cur_best_as_full_12train_4test}"
OUTPUT="${OUTPUT:-exp/${EXP_NAME}}"
EPOCHS_A="${EPOCHS_A:-2}"
EPOCHS_B="${EPOCHS_B:-10}"
EPOCHS_C="${EPOCHS_C:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DECOMP_WEIGHT="${DECOMP_WEIGHT:-0.2}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT}/logs"

# Input/denominator: cur_best_event.  Only geometry_motion is additionally
# opened; events_additive/full is never probed in this mode.
python -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
  -m paired_token_reliability.train_unified_geometry_contribution \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --epochs-a "${EPOCHS_A}" --epochs-b "${EPOCHS_B}" --epochs-c "${EPOCHS_C}" \
  --num-workers "${NUM_WORKERS}" --decomposition-weight "${DECOMP_WEIGHT}" \
  --visualize-every-batches "${TRAIN_VIS_EVERY:-40}" \
  --visualize-val-every-batches "${VAL_VIS_EVERY:-20}" \
  "data.num_views=${NUM_VIEWS:-4}" \
  "model.head_frames_chunk_size=${HEAD_CHUNK:-1}" \
  "data.train_initial_scene_idx=0" \
  "data.train_scene_count=12" \
  "data.train_holdout_frame_count=0" \
  "data.test_initial_scene_idx=12" \
  "data.test_scene_count=4" \
  "data.heldout_test_frame_count=120" \
  "data.event_source_mode=cur_best" \
  "data.decomposition_supervision=true" \
  "data.decomposition_event_root=events_additive" \
  "data.decomposition_geo_branch=geometry_motion" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/train.log"

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  CHECKPOINT="${OUTPUT}/checkpoint-best.pth" OUTPUT_DIR="${OUTPUT}/test_all_exposures" \
  GPU="${GPU_ARRAY[0]}" NUM_VIEWS="${NUM_VIEWS:-4}" \
  bash paired_token_reliability/run_unified_all_exposures_eval.sh \
    2>&1 | tee "${OUTPUT}/logs/evaluate_all_exposures.log"
fi
