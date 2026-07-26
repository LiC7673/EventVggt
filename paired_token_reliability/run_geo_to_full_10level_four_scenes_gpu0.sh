#!/usr/bin/env bash
# Inference-only nested material/noise injection on Actaeon/all EVs.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_refiner_first_1k_then_joint_gpu4/checkpoint-adapter-best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-exp_f/cur_event_refiner_first_1k_then_joint_gpu4/test_geo_plus_random_material_noise_actaeon}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT explicitly; this launcher never silently falls back to another weight." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/logs"
echo "[controlled injection] inference only; GPU=${GPU:-0}; checkpoint=${CHECKPOINT}"
echo "[controlled injection] C=1; geometry always kept; nested material/noise sampling"

CUDA_VISIBLE_DEVICES="${GPU:-0}" python -m \
  paired_token_reliability.evaluate_geo_to_full_10level_four_scenes \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --scene-names "Actaeon_Anodized_Red" \
  --exposures "${EXPOSURES:-0,1,2,5,10}" \
  --ratios "${RATIOS:-0,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.80,1.00}" \
  --geo-depth-scale 2.3 \
  --full-depth-scale 2.2 \
  --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --num-views "${NUM_VIEWS:-4}" \
  --batch-size 1 \
  --num-workers "${NUM_WORKERS:-0}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --save-every-view \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/logs/test.log"
