"""Stage 2: train full-img reliability with frozen Stage-1 geometry events."""

import datetime
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from omegaconf import OmegaConf

import finetune_event as fe
from event_filter_two_stage.data import build_stage2_full_event_loader
from event_filter_two_stage.loss import make_two_stage_reliability_loss
from eventvggt.models.streamvggt_frozen_additive_geometry_detail import StreamVGGT
from event_branch_ablation.common import FULL_RELIABILITY_WEIGHTS
from mul_loss_fine.launcher import configure_mul_loss_cfg


def _safe_snapshot(outdir: str):
    destination = Path(outdir) / "code" / datetime.datetime.now().strftime("%m_%d-%H-%M-%S")
    shutil.copytree(
        ROOT_DIR,
        destination,
        ignore=shutil.ignore_patterns(
            "checkpoints*", "abl_event_exp*", "*__pycache__*", "*.git*", "*.png", "*.jpg"
        ),
        dirs_exist_ok=True,
    )
    return os.fspath(destination)


def _build_model(cfg):
    return StreamVGGT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        event_hidden_dim=16,
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        event_num_bins=int(cfg.model.event_num_bins),
        event_count_cmax=float(getattr(cfg.model, "event_count_cmax", 3.0)),
        residual_scale=0.035,
        residual_highpass_kernel=9,
        residual_patch_zero_mean=True,
        residual_patch_size=14,
        residual_abs_limit=0.025,
        reliability_gate_enabled=True,
        reliability_gate_floor=0.20,
        reliability_init_bias=0.0,
        refine_points=True,
        use_checkpoint=True,
        decomposition_hidden_dim=int(cfg.model.decomposition_hidden_dim),
        decomposition_checkpoint=str(cfg.model.decomposition_checkpoint),
        geometry_event_floor=float(cfg.model.geometry_event_floor),
    )


def _configure_trainable(model, cfg):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if name.startswith("event_detail_refiner."):
            parameter.requires_grad = True


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    for branch_name in ("model", "train", "loss", "data"):
        OmegaConf.set_struct(getattr(cfg, branch_name), False)
    exp_name = str(getattr(cfg, "exp_name", "") or "")
    if exp_name in {"", "event_finetune_LDR5"}:
        exp_name = "two_stage_frozen_geometry_full_img_reliability"
    output_root = ROOT_DIR / "abl_event_exp"
    cfg.exp_name = exp_name
    cfg.save_dir = str(output_root)
    cfg.output_dir = str(output_root / exp_name)
    cfg.logdir = str(output_root / exp_name / "logs")
    reference = (
        ROOT_DIR
        / "checkpoints"
        / "ablation_full_img_reliability_scene12"
        / "checkpoint-last.pth"
    )
    if str(getattr(cfg, "pretrained", "") or "") in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        if not reference.is_file():
            raise FileNotFoundError(
                "Stage 2 requires the trained full_img_reliability checkpoint: "
                f"{reference}. Override pretrained if stored elsewhere."
            )
        cfg.pretrained = str(reference)

    cfg.model.variant = "frozen_additive_geometry_full_img_reliability"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.decomposition_hidden_dim = int(getattr(cfg.model, "decomposition_hidden_dim", 24))
    cfg.model.decomposition_checkpoint = str(
        getattr(
            cfg.model,
            "decomposition_checkpoint",
            output_root / "additive_decomposer_stage1_v2_scene12" / "checkpoint-best.pth",
        )
    )
    cfg.model.geometry_event_floor = float(getattr(cfg.model, "geometry_event_floor", 0.0))
    cfg.data.random_train_ldr = True
    cfg.data.eval_ldr_event_id = str(
        getattr(cfg.data, "eval_ldr_event_id", getattr(cfg.data, "ldr_event_id", "ev_5"))
    )
    cfg.data.ldr_event_id = "random"
    cfg.data.additive_event_root = str(getattr(cfg.data, "additive_event_root", "events_additive"))
    cfg.data.return_normal_gt = True
    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    cfg.epochs = max(int(getattr(cfg, "epochs", 10)), 20)
    cfg.lr = min(float(getattr(cfg, "lr", 1e-4)), 4e-5)
    cfg = configure_mul_loss_cfg(cfg, weights=FULL_RELIABILITY_WEIGHTS, exp_name=exp_name)

    fe.build_event_loader = build_stage2_full_event_loader
    fe.build_event_model = _build_model
    fe.configure_trainable_params = _configure_trainable
    fe.EventSupervisedLoss = make_two_stage_reliability_loss(cfg)
    fe.save_current_code = _safe_snapshot
    print(
        f"[two-stage-2] frozen={cfg.model.decomposition_checkpoint}, "
        f"event_floor={cfg.model.geometry_event_floor}, output={cfg.output_dir}"
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
