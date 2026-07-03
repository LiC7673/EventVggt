"""Evaluate clean main-table checkpoints with the existing EAG3R metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ablation.eag3r_metrics_eval as evaluator  # noqa: E402
from paper_main_ablation.common import build_model  # noqa: E402


def _build_main_table_model(_family, cfg, ckpt, device):
    model = build_model(cfg)
    state = evaluator.strip_module_prefix(evaluator.fe.unwrap_state_dict(ckpt))
    message = model.load_state_dict(state, strict=False)
    print(
        f"[load main-table] variant={cfg.main_table_variant}, "
        f"missing={len(message.missing_keys)}, unexpected={len(message.unexpected_keys)}"
    )
    model.to(device).eval()
    return model


if __name__ == "__main__":
    evaluator.build_model = _build_main_table_model
    evaluator.main()
