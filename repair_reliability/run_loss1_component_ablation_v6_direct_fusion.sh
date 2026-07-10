#!/usr/bin/env bash
set -Eeuo pipefail

# Intended architecture: Stage-1 reliability weights each event voxel before
# an explicit polarity/temporal encoder; event and RGB patch tokens then share
# the camera/depth/point heads. No coarse-depth residual is used.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OUT_ROOT="${OUT_ROOT:-abl_event_exp/loss1_factor_ablation_v6_direct_fusion}"
export LABEL_DIR="${LABEL_DIR:-${OUT_ROOT}/labels_schema2_raw_quantile}"
export LABEL_SCHEMA_VERSION=2
export TOKEN_SOURCE=raw
export TOKEN_AGREEMENT_MODE=robust_quantile
export STAGE2_TAG="direct_fusion_v6"
export STAGE2_MODULE="repair_reliability.finetune_stage2_direct_fusion"
export EVAL_MODULE="repair_reliability.evaluate_stage2_direct_fusion"
export NORMAL_WEIGHT="${NORMAL_WEIGHT:-0.30}"
export POSE_WEIGHT="${POSE_WEIGHT:-1.0}"
export DETAIL_NORMAL_WEIGHT=0.0
export DETAIL_HF_WEIGHT=0.0
export DETAIL_GRAD_WEIGHT=0.0

exec bash "${ROOT_DIR}/repair_reliability/run_loss1_component_ablation.sh" "$@"
