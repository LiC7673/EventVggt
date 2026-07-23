#!/usr/bin/env bash
# Sequential evaluation of the three latest ablations on seven scenes/all EVs.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

ABLATION_ROOT="${ABLATION_ROOT:-exp_f/latest_three_strategy_ablation_3epoch_v2_rgb_routes}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
EXPOSURES="${EXPOSURES:-0,1,2,5,10}"
DEPTH_SCALE="${DEPTH_SCALE:-2.05}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-120}"
NUM_VIEWS="${NUM_VIEWS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VISUALIZE_EVERY="${VISUALIZE_EVERY:-1}"
MAX_VISUALS="${MAX_VISUALS:-0}"

SCENES=(
  "Centaur_Anodized_Red"
  "Child_with_goose_Industrial_Plastic_Grey"
  "Colchester Sphinx_Old_Copper"
  "Cupid as Shepherd_100MB_Old_Copper"
  "DH2_Socrates and Seneca_Car_Paint_Midnight"
  "Dragon_1_Car_Paint_Midnight"
  "NAPOLEON_fix_Anodized_Red"
)

VARIANTS=(
  "noisy_event_only"
  "multi_ldr_only"
  "without_refiner_normal"
)

mkdir -p "${ABLATION_ROOT}/logs"

for scene in "${SCENES[@]}"; do
  if [[ ! -d "${DATA_ROOT}/${scene}" ]]; then
    echo "Scene directory not found: ${DATA_ROOT}/${scene}" >&2
    exit 2
  fi
done

for variant in "${VARIANTS[@]}"; do
  checkpoint="${ABLATION_ROOT}/${variant}/checkpoint-adapter-last.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing ablation checkpoint: ${checkpoint}" >&2
    exit 2
  fi
done

index=0
for variant in "${VARIANTS[@]}"; do
  index=$((index + 1))
  checkpoint="${ABLATION_ROOT}/${variant}/checkpoint-adapter-last.pth"
  output="${ABLATION_ROOT}/${variant}/test_seven_scenes_all_ev_scale_2p05"
  log="${ABLATION_ROOT}/logs/test_${variant}_seven_scenes_scale_2p05.log"

  echo "[${index}/3] variant=${variant} GPU=${CUDA_VISIBLE_DEVICES}"
  echo "checkpoint=${checkpoint}"
  echo "output=${output}"

  python -m paired_token_reliability.evaluate_latest_strategy_ablation \
    --variant "${variant}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${output}" \
    --root "${DATA_ROOT}" \
    --scene-names "${SCENES[@]}" \
    --event-source-mode cur_event \
    --exposures "${EXPOSURES}" \
    --test-frame-count "${TEST_FRAME_COUNT}" \
    --num-views "${NUM_VIEWS}" \
    --window-stride 1 \
    --batch-size 1 \
    --num-workers "${NUM_WORKERS}" \
    --event-resize-method voxel_linear_time \
    --event-resize-bins 5 \
    --depth-scale "${DEPTH_SCALE}" \
    --visualize-every "${VISUALIZE_EVERY}" \
    --max-visuals-per-condition "${MAX_VISUALS}" \
    "$@" 2>&1 | tee "${log}"
done

echo "All ablation evaluations completed under ${ABLATION_ROOT}"
