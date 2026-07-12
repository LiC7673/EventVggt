#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
GPUS="${GPUS:-${GPU:-2}}"; IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"; NPROC="${#GPU_ARRAY[@]}"
if [[ -z "${MASTER_PORT:-}" ]]; then MASTER_PORT="$(python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')"; fi
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"; OUTPUT="${OUTPUT:-exp/linear_voxel_multiscale_5bin_12train_4test}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPUS}"
mkdir -p "${OUTPUT}/logs"
python -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
 -m paired_token_reliability.train_linear_voxel_multiscale --pretrained "${PRETRAINED}" --output "${OUTPUT}" \
 --epochs-a "${EPOCHS_A:-2}" --epochs-b "${EPOCHS_B:-10}" --epochs-c "${EPOCHS_C:-0}" \
 --num-workers "${NUM_WORKERS:-2}" --decomposition-weight "${DECOMP_WEIGHT:-0.2}" --require-full-event-phase-b --no-budget \
 --visualize-every-batches "${TRAIN_VIS_EVERY:-40}" --visualize-val-every-batches "${VAL_VIS_EVERY:-20}" \
 "data.num_views=${NUM_VIEWS:-4}" "data.event_resize_bins=5" "data.event_resize_method=voxel_linear_time" \
 "model.head_frames_chunk_size=${HEAD_CHUNK:-1}" "model.event_decay_tau=${EVENT_DECAY_TAU:-0.003}" \
 "data.train_initial_scene_idx=0" "data.train_scene_count=12" "data.train_holdout_frame_count=0" \
 "data.test_initial_scene_idx=12" "data.test_scene_count=4" "data.heldout_test_frame_count=120" \
 "data.event_source_mode=decomposition_full" "data.decomposition_supervision=true" \
 "data.decomposition_event_root=events_additive" "data.decomposition_geo_branch=geometry_motion" \
 "data.decomposition_full_branch=full" "$@" 2>&1 | tee "${OUTPUT}/logs/train.log"
if [[ "${RUN_EVAL:-1}" == 1 ]]; then
 CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" python -m paired_token_reliability.evaluate_linear_voxel_multiscale \
  --checkpoint "${OUTPUT}/checkpoint-best.pth" --output-dir "${OUTPUT}/test_all_exposures" \
  --initial-scene-idx 12 --active-scene-count 4 --test-frame-count 120 --num-views "${NUM_VIEWS:-4}" \
  --event-resize-method voxel_linear_time --event-resize-bins 5 --num-workers "${NUM_WORKERS:-2}" \
  2>&1 | tee "${OUTPUT}/logs/evaluate_all_exposures.log"
fi
