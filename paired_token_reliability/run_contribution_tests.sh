#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python -m paired_token_reliability.test_contribution_stage1

if [[ $# -gt 0 ]]; then
  python -m paired_token_reliability.evaluate_contribution_stage1 \
    --checkpoint "$1" \
    "${@:2}"
fi

