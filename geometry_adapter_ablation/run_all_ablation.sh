#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
EXP_ROOT="${EXP_ROOT:-exp/geometry_adapter_failure_ablation}"
EPOCHS_A="${EPOCHS_A:-2}"
EPOCHS_B="${EPOCHS_B:-6}"
EPOCHS_C="${EPOCHS_C:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_VIEWS="${NUM_VIEWS:-4}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-2}"
GPU_D="${GPU_D:-3}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${EXP_ROOT}"

run_one() {
  local name="$1"
  local module="$2"
  local gpu="$3"
  local output="${EXP_ROOT}/${name}"
  mkdir -p "${output}/logs"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m "${module}" \
    --pretrained "${PRETRAINED}" \
    --output "${output}" \
    --epochs-a "${EPOCHS_A}" \
    --epochs-b "${EPOCHS_B}" \
    --epochs-c "${EPOCHS_C}" \
    --require-full-event-phase-b \
    --num-workers "${NUM_WORKERS}" \
    --visualize-every-batches "${TRAIN_VIS_EVERY:-40}" \
    --visualize-val-every-batches "${VAL_VIS_EVERY:-20}" \
    "data.num_views=${NUM_VIEWS}" \
    "model.head_frames_chunk_size=${HEAD_CHUNK:-1}" \
    "data.event_source_mode=decomposition_full" \
    "data.decomposition_supervision=true" \
    "data.decomposition_event_root=${DECOMPOSITION_EVENT_ROOT:-events_additive}" \
    "data.decomposition_geo_branch=${DECOMPOSITION_GEO_BRANCH:-geometry_motion}" \
    "data.decomposition_full_branch=${DECOMPOSITION_FULL_BRANCH:-full}" \
    2>&1 | tee "${output}/logs/train.log"
}

run_one experiment_a geometry_adapter_ablation.experiment_a.train "${GPU_A}" & pid_a=$!
run_one experiment_b geometry_adapter_ablation.experiment_b.train "${GPU_B}" & pid_b=$!
run_one experiment_c geometry_adapter_ablation.experiment_c.train "${GPU_C}" & pid_c=$!
run_one experiment_d geometry_adapter_ablation.experiment_d.train "${GPU_D}" & pid_d=$!

status=0
for pid in "${pid_a}" "${pid_b}" "${pid_c}" "${pid_d}"; do
  wait "${pid}" || status=1
done
exit "${status}"

