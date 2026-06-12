#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-7}"
HELDOUT_INITIAL_SCENE_IDX="${HELDOUT_INITIAL_SCENE_IDX:-12}"
HELDOUT_ACTIVE_SCENE_COUNT="${HELDOUT_ACTIVE_SCENE_COUNT:-6}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/ablation/results/scene12_heldout6}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/ablation/eag3r_eval_manifest_scene12.json}"

cd "${ROOT_DIR}"

GPU="${GPU}" \
HELDOUT_INITIAL_SCENE_IDX="${HELDOUT_INITIAL_SCENE_IDX}" \
HELDOUT_ACTIVE_SCENE_COUNT="${HELDOUT_ACTIVE_SCENE_COUNT}" \
SPLIT=all \
OUT_DIR="${OUT_DIR}" \
MANIFEST="${MANIFEST}" \
bash ablation/run_eag3r_metrics_heldout_scenes.sh "$@"
