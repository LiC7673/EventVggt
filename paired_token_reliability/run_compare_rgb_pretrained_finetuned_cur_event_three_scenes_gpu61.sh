#!/usr/bin/env bash
# Evaluate three frozen checkpoints with exactly the same scenes/exposures.
# Order: RGB pretrained (GPU 6) -> RGB finetuned (GPU 1) -> cur-event (GPU 6).
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp_f/compare_rgb_pretrained_finetuned_cur_event_three_scenes}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
FINETUNED_TEMPLATE="${FINETUNED_TEMPLATE:-checkpoints/fine_rgb_{ldr_event_id}/checkpoint-last.pth}"
EVENT_CHECKPOINT="${EVENT_CHECKPOINT:-exp_f/cur_event_refiner_first_1k_then_joint_gpu4/checkpoint-adapter-best.pth}"
EXPOSURES="${EXPOSURES:-0,1,2,5,10}"
DEPTH_SCALE="${DEPTH_SCALE:-2.0}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-120}"
NUM_VIEWS="${NUM_VIEWS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VISUALIZE_EVERY="${VISUALIZE_EVERY:-1}"

SCENES=(
  "DH2_Socrates and Seneca_Car_Paint_Midnight"
  "Dragon_1_Car_Paint_Midnight"
  "NAPOLEON_fix_Anodized_Red"
)

mkdir -p "${OUTPUT_ROOT}/logs"

for scene in "${SCENES[@]}"; do
  if [[ ! -d "${DATA_ROOT}/${scene}" ]]; then
    echo "Scene directory not found: ${DATA_ROOT}/${scene}" >&2
    echo "Available close matches:" >&2
    find "${DATA_ROOT}" -maxdepth 1 -mindepth 1 -type d \
      -iname "*$(printf '%s' "${scene}" | cut -c1-12)*" -printf '  %f\n' >&2 || true
    exit 2
  fi
done

if [[ ! -f "${PRETRAINED}" ]]; then
  echo "RGB pretrained checkpoint not found: ${PRETRAINED}" >&2
  exit 2
fi
if [[ ! -f "${EVENT_CHECKPOINT}" ]]; then
  echo "cur-event checkpoint not found: ${EVENT_CHECKPOINT}" >&2
  echo "Override it with EVENT_CHECKPOINT=/path/to/checkpoint-adapter-{best,last}.pth" >&2
  exit 2
fi

echo "[1/3] RGB pretrained/no finetuning on physical GPU 6"
CUDA_VISIBLE_DEVICES=6 python -m fine_rgb.evaluate_rgb_four_scenes_streaming \
  --pretrained "${PRETRAINED}" \
  --skip-finetuned \
  --output-dir "${OUTPUT_ROOT}/rgb_pretrained" \
  --data-root "${DATA_ROOT}" \
  --scenes "${SCENES[@]}" \
  --ldr-event-ids "${EXPOSURES}" \
  --num-views "${NUM_VIEWS}" \
  --test-frame-count "${TEST_FRAME_COUNT}" \
  --num-workers "${NUM_WORKERS}" \
  --depth-scale "${DEPTH_SCALE}" \
  --visualize-every "${VISUALIZE_EVERY}" \
  --max-visuals-per-condition 0 \
  "$@" 2>&1 | tee "${OUTPUT_ROOT}/logs/01_rgb_pretrained.log"

echo "[2/3] RGB finetuned checkpoints on physical GPU 1"
CUDA_VISIBLE_DEVICES=1 python -m fine_rgb.evaluate_rgb_four_scenes_streaming \
  --pretrained "${PRETRAINED}" \
  --finetuned-template "${FINETUNED_TEMPLATE}" \
  --skip-pretrained \
  --output-dir "${OUTPUT_ROOT}/rgb_finetuned" \
  --data-root "${DATA_ROOT}" \
  --scenes "${SCENES[@]}" \
  --ldr-event-ids "${EXPOSURES}" \
  --num-views "${NUM_VIEWS}" \
  --test-frame-count "${TEST_FRAME_COUNT}" \
  --num-workers "${NUM_WORKERS}" \
  --depth-scale "${DEPTH_SCALE}" \
  --visualize-every "${VISUALIZE_EVERY}" \
  --max-visuals-per-condition 0 \
  "$@" 2>&1 | tee "${OUTPUT_ROOT}/logs/02_rgb_finetuned.log"

echo "[3/3] cur_event_refiner_first_1k_then_joint checkpoint on physical GPU 6"
CUDA_VISIBLE_DEVICES=6 python -m \
  paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes \
  --checkpoint "${EVENT_CHECKPOINT}" \
  --output-dir "${OUTPUT_ROOT}/cur_event_refiner_first" \
  --root "${DATA_ROOT}" \
  --event-source-mode cur_event \
  --scene-names "${SCENES[@]}" \
  --exposures "${EXPOSURES}" \
  --num-views "${NUM_VIEWS}" \
  --test-frame-count "${TEST_FRAME_COUNT}" \
  --batch-size 1 \
  --num-workers "${NUM_WORKERS}" \
  --event-resize-method voxel_linear_time \
  --event-resize-bins 5 \
  --depth-scale "${DEPTH_SCALE}" \
  --visualize-every "${VISUALIZE_EVERY}" \
  --max-visuals-per-condition 0 \
  "$@" 2>&1 | tee "${OUTPUT_ROOT}/logs/03_cur_event_refiner_first.log"

echo "All three evaluations completed: ${OUTPUT_ROOT}"
