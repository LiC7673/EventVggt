from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
import torch
from omegaconf import OmegaConf

import finetune_event as fe
from eventvggt.models.streamvggt_two_stage import StreamVGGT as TwoStageStreamVGGT


def _set_exp_paths(cfg, exp_name: str) -> None:
    cfg.exp_name = exp_name
    cfg.logdir = f"{cfg.save_dir}/{exp_name}/logs"
    cfg.output_dir = f"{cfg.save_dir}/{exp_name}"


def _stack_res_field(model_output, key: str) -> Optional[torch.Tensor]:
    if not getattr(model_output, "ress", None):
        return None
    if not all(key in res for res in model_output.ress):
        return None
    value = torch.stack([res[key] for res in model_output.ress], dim=1)
    if value.ndim == 5 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    return value


class TwoStageResidualEventSupervisedLoss(fe.EventSupervisedLoss):
    def __init__(
        self,
        *args,
        residual_depth_weight: float = 1.0,
        coarse_depth_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.residual_depth_weight = float(residual_depth_weight)
        self.coarse_depth_weight = float(coarse_depth_weight)

    def forward(
        self,
        model_output,
        views: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        total_loss, details, aux = super().forward(model_output, views)

        depth_pred = torch.stack([res["depth"] for res in model_output.ress], dim=1).squeeze(-1)
        valid_mask = aux["valid_mask"].to(device=depth_pred.device)
        depth_gt_aligned = aux["depth_gt_aligned"].to(device=depth_pred.device, dtype=depth_pred.dtype)

        depth_coarse = _stack_res_field(model_output, "depth_coarse")
        depth_residual = _stack_res_field(model_output, "depth_residual")

        if depth_coarse is None or depth_residual is None:
            zero = depth_pred.new_tensor(0.0)
            details["coarse_depth_loss"] = 0.0
            details["residual_depth_loss"] = 0.0
            details["depth_residual_abs"] = 0.0
            return total_loss + zero, details, aux

        depth_coarse = depth_coarse.to(device=depth_pred.device, dtype=depth_pred.dtype)
        depth_residual = depth_residual.to(device=depth_pred.device, dtype=depth_pred.dtype)
        residual_target = depth_gt_aligned - depth_coarse.detach()

        coarse_depth_loss = fe.masked_l1(depth_coarse, depth_gt_aligned, valid_mask)
        residual_depth_loss = fe.masked_l1(depth_residual, residual_target, valid_mask)
        residual_abs = fe.masked_l1(depth_residual, torch.zeros_like(depth_residual), valid_mask)

        total_loss = (
            total_loss
            + self.coarse_depth_weight * coarse_depth_loss
            + self.residual_depth_weight * residual_depth_loss
        )

        details["coarse_depth_loss"] = float(coarse_depth_loss.detach())
        details["residual_depth_loss"] = float(residual_depth_loss.detach())
        details["depth_residual_abs"] = float(residual_abs.detach())

        aux["depth_coarse"] = depth_coarse.detach()
        aux["depth_residual"] = depth_residual.detach()
        aux["depth_residual_target"] = residual_target.detach()

        event_motion_density = getattr(model_output, "event_motion_density", None)
        if event_motion_density is not None:
            aux["event_motion_density"] = event_motion_density.detach()

        return total_loss, details, aux


def configure_two_stage_trainable_params(model, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    enabled_prefixes = ["event_encoder", "event_residual_refiner"]
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in enabled_prefixes):
            param.requires_grad = True

    if cfg.train.unfreeze_heads:
        for module in (model.camera_head, model.depth_head, model.point_head, model.track_head):
            for param in module.parameters():
                param.requires_grad = True

    if cfg.train.unfreeze_aggregator_blocks:
        for param in model.aggregator.frame_blocks.parameters():
            param.requires_grad = True
        for param in model.aggregator.global_blocks.parameters():
            param.requires_grad = True

    last_blocks = int(getattr(cfg.train, "unfreeze_aggregator_last_blocks", 0))
    if last_blocks > 0:
        for block in model.aggregator.frame_blocks[-last_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for block in model.aggregator.global_blocks[-last_blocks:]:
            for param in block.parameters():
                param.requires_grad = True


def build_two_stage_optimizer_params(model, cfg):
    base_lr = float(cfg.lr)
    vggt_lr = float(getattr(cfg.train, "vggt_last_blocks_lr", base_lr))
    groups = {"event": [], "heads": [], "vggt": [], "other": []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("event_encoder", "event_residual_refiner")):
            groups["event"].append(param)
        elif name.startswith("aggregator"):
            groups["vggt"].append(param)
        elif name.startswith(("camera_head", "depth_head", "point_head", "track_head")):
            groups["heads"].append(param)
        else:
            groups["other"].append(param)

    param_groups = []
    for key in ("event", "heads", "other"):
        if groups[key]:
            param_groups.append({"params": groups[key], "lr_scale": 1.0})
    if groups["vggt"]:
        param_groups.append({"params": groups["vggt"], "lr_scale": vggt_lr / max(base_lr, 1e-12)})
    return param_groups


def launch(
    cfg,
    *,
    exp_name: Optional[str] = None,
    residual_input_mode: Optional[str] = None,
) -> None:
    if residual_input_mode is not None:
        cfg.model.residual_input_mode = residual_input_mode
    if exp_name is not None:
        _set_exp_paths(cfg, exp_name)

    OmegaConf.resolve(cfg)

    class ConfiguredTwoStageStreamVGGT(TwoStageStreamVGGT):
        def __init__(self, *args, **kwargs):
            kwargs.pop("event_hidden_dim", None)
            kwargs.pop("head_frames_chunk_size", None)
            super().__init__(
                *args,
                event_hidden_dim=int(getattr(cfg.model, "event_hidden_dim", 32)),
                event_num_bins=int(getattr(cfg.model, "event_num_bins", 8)),
                event_encode_downsample=int(getattr(cfg.model, "event_encode_downsample", getattr(cfg.model, "event_downsample", 4))),
                head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 8)),
                residual_hidden_dim=int(getattr(cfg.model, "residual_hidden_dim", 96)),
                event_downsample=int(getattr(cfg.model, "event_downsample", 4)),
                residual_scale=float(getattr(cfg.model, "residual_scale", 0.1)),
                residual_input_mode=str(getattr(cfg.model, "residual_input_mode", "current_event")),
                **kwargs,
            )

    class ConfiguredTwoStageLoss(TwoStageResidualEventSupervisedLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                residual_depth_weight=float(getattr(cfg.loss, "residual_depth_weight", 1.0)),
                coarse_depth_weight=float(getattr(cfg.loss, "coarse_depth_weight", 0.0)),
                **kwargs,
            )

    fe.EventStreamVGGT = ConfiguredTwoStageStreamVGGT
    fe.EventSupervisedLoss = ConfiguredTwoStageLoss
    fe.configure_trainable_params = configure_two_stage_trainable_params
    fe.build_optimizer_params = build_two_stage_optimizer_params
    fe.train(cfg)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event_two_stage_residual.yaml",
)
def run(cfg: OmegaConf):
    launch(cfg)


if __name__ == "__main__":
    run()
