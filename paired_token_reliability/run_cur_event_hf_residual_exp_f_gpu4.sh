#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
export GPU="${GPU:-4}"
export OUTPUT="${OUTPUT:-exp_f/cur_event_clean_hf_residual_v2_gpu4}"

# Reuse the strict cur_event launcher protocol, changing only the Python model
# and objective entry. The explicit module remains the authoritative source.
TRAIN_MODULE="paired_token_reliability.train_linear_voxel_cur_event_hf_residual"
GPUS="${GPUS:-${GPU}}"; IFS=',' read -r -a ga <<< "${GPUS}"; n="${#ga[@]}"
port="${MASTER_PORT:-$(python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPUS}"
mkdir -p "${OUTPUT}/logs"
python -m torch.distributed.run --nproc_per_node "${n}" --master_port "${port}" \
 -m "${TRAIN_MODULE}" --pretrained "${PRETRAINED:-ckpt/model.pt}" --output "${OUTPUT}" \
 --epochs-a 1 --epochs-b "${EPOCHS_B:-6}" --epochs-c 0 --num-workers "${NUM_WORKERS:-2}" \
 --point-weight 0 --pair-weight 0 --decomposition-weight 0 --event-normal-weight 1 \
 --depth-event-normal-weight .5 --update-weight 0 --no-budget \
 --visualize-every-batches "${TRAIN_VIS_EVERY:-40}" --visualize-val-every-batches "${VAL_VIS_EVERY:-5}" \
 "data.num_views=${NUM_VIEWS:-4}" "data.event_resize_bins=5" "data.event_resize_method=voxel_linear_time" \
 "data.train_initial_scene_idx=0" "data.train_scene_count=12" "data.train_holdout_frame_count=0" \
 "data.test_initial_scene_idx=12" "data.test_scene_count=4" "data.heldout_test_frame_count=120" \
 "data.event_source_mode=cur_event" "data.decomposition_supervision=true" \
 "data.decomposition_event_root=events_additive" "data.decomposition_geo_branch=geometry_motion" \
 "model.head_frames_chunk_size=${HEAD_CHUNK:-1}" "model.pixel_refiner_delay=500" \
 "model.pixel_refine_log_limit=0.30" "model.c_delay_steps=1000" \
 "model.c_transition_steps=1000" "model.event_decay_tau=0.0015" "$@" \
 2>&1 | tee "${OUTPUT}/logs/train.log"
