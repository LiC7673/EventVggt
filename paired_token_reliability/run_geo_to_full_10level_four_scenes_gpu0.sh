#!/usr/bin/env bash
# Inference-only ten-level E_geo -> E_full continuum on four scenes/all EVs.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_refiner_first_1k_then_joint_gpu4/checkpoint-adapter-best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-exp_f/cur_event_refiner_first_1k_then_joint_gpu4/test_geo_to_full_10level}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT explicitly; this launcher never silently falls back to another weight." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/logs"
echo "[geo->full] inference only; GPU=${GPU:-0}; checkpoint=${CHECKPOINT}"
echo "[geo->full] 10 levels; scale 2.3 (geo) -> 2.2 (full)"

CUDA_VISIBLE_DEVICES="${GPU:-0}" python -m \
  paired_token_reliability.evaluate_geo_to_full_10level_four_scenes \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --scene-names \
    "Centaur_Anodized_Red" \
    "Child_with_goose_Industrial_Plastic_Grey" \
    "Colchester Sphinx_Old_Copper" \
    "Cupid as Shepherd_100MB_Old_Copper" \
  --exposures "${EXPOSURES:-0,1,2,5,10}" \
  --levels 10 \
  --geo-depth-scale 2.3 \
  --full-depth-scale 2.2 \
  --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --num-views "${NUM_VIEWS:-4}" \
  --batch-size 1 \
  --num-workers "${NUM_WORKERS:-0}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --save-every-view \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/logs/test.log"
