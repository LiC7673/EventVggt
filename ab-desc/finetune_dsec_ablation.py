"""Train DSEC RGB/event ablations and collect their held-out metrics."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import finetune_event as fe  # noqa: E402
from dsec_exp.common import build_dsec_loader  # noqa: E402
from dsec_exp.finetune_dsec import _prepare_common, _run_event, _run_rgb  # noqa: E402


def _set_output_paths(cfg) -> None:
    output = Path(str(cfg.save_dir)).resolve() / str(cfg.exp_name)
    cfg.output_dir = str(output)
    cfg.logdir = str(output / "logs")


def _configure_plain_event_params(model, cfg) -> None:
    """Train the raw event path and the same prediction heads as RGB fine-tuning."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "event_encoder" in name or "event_patch_embed" in name:
            parameter.requires_grad = True
    for module_name in ("depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True


def _run_plain_event(cfg) -> None:
    cfg.approach = "event_plain"
    cfg.model.variant = "base"
    cfg.model.event_hidden_dim = int(getattr(cfg.model, "event_hidden_dim", 16))
    cfg.train.unfreeze_heads = True
    cfg.train.unfreeze_aggregator_blocks = False
    fe.build_event_loader = lambda local_cfg, split="train": build_dsec_loader(
        local_cfg,
        "train" if split == "train" else "val",
        rgb_only=False,
    )
    fe.configure_trainable_params = _configure_plain_event_params
    print(
        "DSEC ablation=event_plain: raw event token fusion; "
        "no reliability and no extra GT-detail loss."
    )
    fe.train(cfg)


@hydra.main(
    version_base=None,
    config_path=str(ROOT / "config"),
    config_name="finetune_dsec_event.yaml",
)
def train(cfg: OmegaConf) -> None:
    cfg = _prepare_common(cfg)
    OmegaConf.set_struct(cfg, False)
    variant = str(getattr(cfg, "ablation_variant", "full_img_reliability")).lower()
    _set_output_paths(cfg)

    if variant == "rgb_finetune":
        cfg.approach = "rgb"
        _run_rgb(cfg)
        return
    if variant == "event_plain":
        _run_plain_event(cfg)
        return
    if variant == "full_img_reliability":
        cfg.approach = "full_img_reliability"
        _run_event(cfg)
        return
    raise ValueError(
        f"Unknown ablation_variant={variant}; expected rgb_finetune, "
        "event_plain, or full_img_reliability"
    )


def collect_results(run_root: Path) -> None:
    summary_rows = []
    scene_rows = []
    for variant_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        metrics_path = variant_dir / "heldout_test" / "metrics.json"
        if not metrics_path.is_file():
            continue
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        variant = variant_dir.name
        summary_rows.append({"variant": variant, **report.get("metrics_mean", {})})
        for row in report.get("scenes", []):
            scene_rows.append({"variant": variant, **row})

    if not summary_rows:
        raise RuntimeError(f"No heldout_test/metrics.json files found below {run_root}")

    def write_csv(path: Path, rows) -> None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(run_root / "summary_metrics.csv", summary_rows)
    write_csv(run_root / "per_scene_metrics.csv", scene_rows)
    print(f"Collected {len(summary_rows)} variants into {run_root / 'summary_metrics.csv'}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--collect":
        collect_results(Path(sys.argv[2]).resolve())
    else:
        train()
