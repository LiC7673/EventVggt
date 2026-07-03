"""Train source-aware Multi-LDR VGGT with a frozen source decomposer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import finetune_event as fe  # noqa: E402
from eventvggt.models.streamvggt_source_aware_detail import StreamVGGT  # noqa: E402
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


def _build_model(cfg):
    return StreamVGGT(
        img_size=int(cfg.model.img_size),
        patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim),
        head_frames_chunk_size=int(cfg.model.head_frames_chunk_size),
        event_num_bins=int(cfg.model.event_num_bins),
        event_hidden_dim=int(cfg.model.main_event_hidden_dim),
        event_count_cmax=float(cfg.model.event_count_cmax),
        residual_scale=float(cfg.model.refiner_residual_scale),
        residual_highpass_kernel=int(cfg.model.event_delta_highpass_kernel),
        residual_patch_zero_mean=bool(cfg.model.event_delta_patch_zero_mean),
        residual_patch_size=int(cfg.model.event_delta_patch_size),
        residual_abs_limit=float(cfg.model.event_delta_abs_limit),
        refine_points=True,
        use_checkpoint=bool(cfg.model.refiner_use_checkpoint),
        support_threshold=float(cfg.model.causal_support_threshold),
        support_dilate_kernel=int(cfg.model.causal_support_dilate_kernel),
        support_blur_kernel=int(cfg.model.causal_support_blur_kernel),
        forward_batch_chunk=int(cfg.model.exposure_forward_batch_chunk),
        source_checkpoint=str(cfg.model.source_checkpoint),
        source_hidden_dim=int(cfg.model.source_hidden_dim),
        source_gate_floor=float(cfg.model.source_gate_floor),
        source_frame_chunk_size=int(cfg.model.source_frame_chunk_size),
        source_ablation_mode=str(cfg.model.source_ablation_mode),
    )


def _configure_trainable(model, _cfg):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in ("camera_head", "depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if name.startswith("event_detail_refiner."):
            parameter.requires_grad = True
    model.event_source_decomposer.requires_grad_(False)
    model.event_source_decomposer.eval()


def _make_source_aware_loss(cfg):
    paired_loss = _make_paired_multildr_loss(cfg)

    class SourceAwareStage2Loss(paired_loss):
        def forward(self, model_output, views):
            total, details, aux = super().forward(model_output, views)
            if model_output.ress and all(
                "pred_event_source_probability" in result for result in model_output.ress
            ):
                source = torch.stack(
                    [result["pred_event_source_probability"] for result in model_output.ress],
                    dim=1,
                )
                aux["event_reliability"] = source[:, :, 0].detach()
                details.update(
                    {
                        "source_geometry_mean": float(source[:, :, 0].mean().detach()),
                        "source_material_mean": float(source[:, :, 1].mean().detach()),
                        "source_noise_mean": float(source[:, :, 2].mean().detach()),
                    }
                )
            return total, details, aux

    return SourceAwareStage2Loss


@hydra.main(
    version_base=None,
    config_path=str(ROOT / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg):
    OmegaConf.set_struct(cfg, False)
    output_root = Path(
        str(getattr(cfg, "source_aware_output_root", ROOT / "abl_event_exp/source_aware_60_12_12"))
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    cfg.main_table_output_root = str(output_root)
    cfg = _prepare_cfg(cfg, "m3_event_detail_multildr")

    manifest_path = Path(str(cfg.data.scene_split_manifest)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = load_scene_split(manifest_path)
    counts = {name: len(values) for name, values in manifest["splits"].items()}
    if counts != {"train": 60, "val": 12, "test": 12}:
        raise ValueError(f"Expected source-aware split 60/12/12, got {counts}")

    source_checkpoint = Path(str(cfg.model.source_checkpoint)).expanduser()
    if not source_checkpoint.is_absolute():
        source_checkpoint = ROOT / source_checkpoint
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Source decomposer checkpoint missing: {source_checkpoint}")
    cfg.model.source_checkpoint = str(source_checkpoint)
    cfg.model.source_hidden_dim = int(getattr(cfg.model, "source_hidden_dim", 24))
    cfg.model.source_gate_floor = float(getattr(cfg.model, "source_gate_floor", 0.05))
    cfg.model.source_frame_chunk_size = int(
        getattr(cfg.model, "source_frame_chunk_size", 1)
    )
    cfg.model.source_ablation_mode = str(
        getattr(cfg.model, "source_ablation_mode", "learned")
    )
    cfg.data.scene_split_manifest = str(manifest_path)
    cfg.epochs = int(getattr(cfg, "source_aware_epochs", 30))
    cfg.validate_each_epoch = True
    cfg.validation_monitor = "loss"
    cfg.validation_min_delta = 1.0e-4
    cfg.early_stopping_patience = int(getattr(cfg, "source_aware_patience", 5))
    cfg.eval_every_steps = 0
    cfg.skip_final_eval = True
    cfg.vis.test_max_batches = 0
    cfg.exp_name = f"source_{cfg.model.source_ablation_mode}_full_train60_val12"
    cfg.output_dir = str(output_root / cfg.exp_name)
    cfg.logdir = str(Path(cfg.output_dir) / "logs")
    cfg.save_dir = str(output_root)
    cfg.ablation_contract.update(
        {
            "event_guidance": "frozen_geometry_material_noise_source_decomposition",
            "source_checkpoint": str(source_checkpoint),
            "source_gate_floor": cfg.model.source_gate_floor,
            "source_ablation_mode": cfg.model.source_ablation_mode,
            "source_training_test_scenes": False,
            "test_used_during_training": False,
            "scene_counts": counts,
        }
    )
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "training_contract.json").write_text(
        json.dumps(OmegaConf.to_container(cfg.ablation_contract, resolve=True), indent=2),
        encoding="utf-8",
    )

    fe.Accelerator = UnevenBatchAccelerator
    fe.save_current_code = _safe_code_snapshot
    fe.build_event_loader = build_scene_disjoint_loader
    fe.build_event_model = _build_model
    fe.configure_trainable_params = _configure_trainable
    fe.EventSupervisedLoss = _make_source_aware_loss(cfg)
    print(
        "[source-aware] explicit geometry/material/noise pretraining; "
        f"source={source_checkpoint}; output={cfg.output_dir}"
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
