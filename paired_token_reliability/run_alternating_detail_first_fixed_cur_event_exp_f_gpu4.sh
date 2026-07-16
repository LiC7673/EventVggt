#!/usr/bin/env bash
# Latest alternating model: strict cur_event main stream + geometry_motion Geo teacher.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
GPUS="${GPUS:-${GPU:-4}}"; IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"; NPROC="${#GPU_ARRAY[@]}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')"
fi
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
OUTPUT="${OUTPUT:-exp_f/alternating_geo_detail_first_dual_c_fixed_cur_event_gpu4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "${OUTPUT}/logs"

python -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
  -m paired_token_reliability.train_linear_voxel_alternating_detail_first_fixed_cur_event \
  --pretrained "${PRETRAINED}" --output "${OUTPUT}" \
  --epochs-a 1 --epochs-b "${EPOCHS_B:-6}" --epochs-c 0 \
  --num-workers "${NUM_WORKERS}" \
  --point-weight 0.0 --pair-weight 0.0 --decomposition-weight 0.0 \
  --event-normal-weight 1.0 --depth-event-normal-weight 0.5 \
  --update-weight 0.0 --no-budget \
  --visualize-every-batches "${TRAIN_VIS_EVERY:-40}" \
  --visualize-val-every-batches "${VAL_VIS_EVERY:-5}" \
  "data.num_views=${NUM_VIEWS:-4}" \
  "data.event_resize_bins=5" "data.event_resize_method=voxel_linear_time" \
  "data.train_initial_scene_idx=0" "data.train_scene_count=12" \
  "data.train_holdout_frame_count=0" \
  "data.test_initial_scene_idx=12" "data.test_scene_count=4" \
  "data.heldout_test_frame_count=120" \
  "data.event_source_mode=cur_event" \
  "data.decomposition_supervision=true" \
  "data.decomposition_event_root=events_additive" \
  "data.decomposition_geo_branch=geometry_motion" \
  "model.head_frames_chunk_size=${HEAD_CHUNK:-1}" \
  "model.pixel_refiner_delay=500" "model.pixel_refine_log_limit=0.30" \
  "model.c_delay_steps=1000" "model.c_transition_steps=1000" \
  "model.event_decay_tau=0.0015" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/train.log"

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  CHECKPOINT="${CHECKPOINT:-${OUTPUT}/checkpoint-adapter-last.pth}" \
  OUTPUT_DIR="${OUTPUT}/test_cur_event_four_scenes_all_ev" \
  GPU="${GPU_ARRAY[0]}" NUM_VIEWS="${NUM_VIEWS:-4}" \
  TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-120}" \
  bash paired_token_reliability/run_test_alternating_detail_first_fixed_cur_event_four_scenes_gpu0.sh \
    2>&1 | tee "${OUTPUT}/logs/test_cur_event_four_scenes_all_ev.log"
fi
