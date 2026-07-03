"""Train the full paired-Multi-LDR reliability model on a 60/12 scene split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import finetune_event as fe  # noqa: E402
from paper_main_ablation.common import build_model, configure_trainable_params  # noqa: E402
from paper_main_ablation.finetune_main_table import (  # noqa: E402
    UnevenBatchAccelerator,
    _make_paired_multildr_loss,
    _prepare_cfg,
    _safe_code_snapshot,
)
from paper_scale_training.scene_split_loader import (  # noqa: E402
    build_scene_disjoint_loader,
    load_scene_split,
)


@hydra.main(
    version_base=None,
    config_path=str(ROOT / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg):
    OmegaConf.set_struct(cfg, False)
    cfg.main_table_output_root = str(
        getattr(cfg, "paper_scale_output_root", ROOT / "abl_event_exp/paper_scale_60_12_12")
    )
    cfg = _prepare_cfg(cfg, "m4_full_reliability")

    manifest_path = Path(str(cfg.data.scene_split_manifest)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Scene split manifest missing: {manifest_path}")
    manifest = load_scene_split(manifest_path)
    counts = {name: len(values) for name, values in manifest["splits"].items()}
    if counts != {"train": 60, "val": 12, "test": 12}:
        raise ValueError(f"Expected scene counts 60/12/12, got {counts}")

    cfg.data.scene_split_manifest = str(manifest_path)
    cfg.epochs = int(getattr(cfg, "paper_scale_epochs", 30))
    cfg.validate_each_epoch = True
    cfg.validation_monitor = str(getattr(cfg, "paper_scale_validation_monitor", "loss"))
    cfg.validation_min_delta = float(getattr(cfg, "paper_scale_validation_min_delta", 1.0e-4))
    cfg.early_stopping_patience = int(getattr(cfg, "paper_scale_early_stopping_patience", 5))
    cfg.eval_every_steps = 0
    cfg.skip_final_eval = True
    cfg.vis.test_max_batches = 0
    cfg.vis.test_num_views = 1

    output_root = Path(str(cfg.main_table_output_root))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    cfg.exp_name = "full_model_train60_val12"
    cfg.output_dir = str(output_root / cfg.exp_name)
    cfg.logdir = str(Path(cfg.output_dir) / "logs")
    cfg.save_dir = str(output_root)
    cfg.ablation_contract.update(
        {
            "scene_split_manifest": str(manifest_path),
            "scene_counts": counts,
            "validation_each_epoch": True,
            "early_stopping_patience": cfg.early_stopping_patience,
            "test_used_during_training": False,
            "multi_ldr_mode": "same_window_paired_exposure_consistency",
        }
    )

    reliability_checkpoint = Path(str(cfg.model.reliability_checkpoint))
    if not reliability_checkpoint.is_file():
        raise FileNotFoundError(f"Frozen ReliabilityNet missing: {reliability_checkpoint}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(cfg.output_dir) / "training_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(OmegaConf.to_container(cfg.ablation_contract, resolve=True), handle, indent=2)
    with (Path(cfg.output_dir) / "scene_split.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    fe.Accelerator = UnevenBatchAccelerator
    fe.save_current_code = _safe_code_snapshot
    fe.build_event_loader = build_scene_disjoint_loader
    fe.build_event_model = build_model
    fe.configure_trainable_params = configure_trainable_params
    fe.EventSupervisedLoss = _make_paired_multildr_loss(cfg)

    print(
        "[paper scale] full model, scene-disjoint 60 train / 12 val / 12 test; "
        f"epochs={cfg.epochs}, patience={cfg.early_stopping_patience}, output={cfg.output_dir}"
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
