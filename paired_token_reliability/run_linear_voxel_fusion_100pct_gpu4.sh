#!/usr/bin/env bash
# Full fusion pipeline: Contribution + event normal + depth consistency,
# with a ±100% pixel-depth correction range. Runs only on physical GPU 4.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
export GPUS=4
export OUTPUT="${OUTPUT:-exp/linear_voxel_fusion_100pct_gpu4}"
bash paired_token_reliability/run_linear_voxel_multiscale_12train_4test.sh \
  --point-weight 0.0 --decomposition-weight 1.0 --pair-weight 0.10 \
  "model.depth_update_scale=1.00" "$@"
