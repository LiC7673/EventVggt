from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import finetune_event as fe
from eventvggt.models.streamvggt_global_local import StreamVGGT as GlobalLocalStreamVGGT


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


def _gradient_magnitude(depth: torch.Tensor) -> torch.Tensor:
    dx = F.pad(depth[..., :, 1:] - depth[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(depth[..., 1:, :] - depth[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def _normalize_map_per_frame(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.to(device=value.device)
    if mask.dtype != value.dtype:
        mask = mask.to(dtype=value.dtype)
    masked_value = value * mask
    max_value = masked_value.flatten(2).amax(dim=-1).clamp_min(eps).view(*value.shape[:2], 1, 1)
    return value / max_value


class GlobalLocalEventSupervisedLoss(fe.EventSupervisedLoss):
    def __init__(
        self,
        *args,
        event_edge_weight: float = 0.0,
        detail_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.event_edge_weight = float(event_edge_weight)
        self.detail_weight = float(detail_weight)

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
        points_coarse = _stack_res_field(model_output, "pts3d_coarse")
        points_residual = _stack_res_field(model_output, "pts3d_residual")

        if depth_coarse is not None:
            depth_coarse = depth_coarse.to(device=depth_pred.device, dtype=depth_pred.dtype)
            coarse_depth_loss = fe.masked_l1(depth_coarse, depth_gt_aligned, valid_mask)
            total_loss = total_loss + 0.0 * coarse_depth_loss
            details["coarse_depth_loss"] = float(coarse_depth_loss.detach())
            aux["depth_coarse"] = depth_coarse.detach()

        if depth_residual is not None:
            depth_residual = depth_residual.to(device=depth_pred.device, dtype=depth_pred.dtype)
            residual_abs = (depth_residual.abs() * valid_mask.to(depth_residual.dtype)).sum()
            residual_abs = residual_abs / valid_mask.to(depth_residual.dtype).sum().clamp_min(1.0)
            details["depth_residual_abs"] = float(residual_abs.detach())
            aux["depth_residual"] = depth_residual.detach()

        if points_coarse is not None:
            aux["points_coarse"] = points_coarse.detach()
        if points_residual is not None:
            aux["points_residual"] = points_residual.detach()

        if self.detail_weight > 0:
            detail_loss = fe.masked_l1(
                _gradient_magnitude(depth_pred),
                _gradient_magnitude(depth_gt_aligned),
                valid_mask,
            )
            total_loss = total_loss + self.detail_weight * detail_loss
        else:
            detail_loss = depth_pred.new_tensor(0.0)
        details["detail_grad_loss"] = float(detail_loss.detach())

        event_motion_density = getattr(model_output, "event_motion_density", None)
        if self.event_edge_weight > 0 and event_motion_density is not None:
            if event_motion_density.shape[-2:] != depth_pred.shape[-2:]:
                density_flat = event_motion_density.reshape(-1, 1, *event_motion_density.shape[-2:])
                density_flat = F.interpolate(
                    density_flat,
                    size=depth_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                event_motion_density = density_flat.reshape(*depth_pred.shape)

            depth_edge = _normalize_map_per_frame(_gradient_magnitude(depth_pred), valid_mask)
            event_edge = _normalize_map_per_frame(event_motion_density.detach().to(depth_edge.dtype), valid_mask)
            event_edge_loss = fe.masked_l1(depth_edge, event_edge, valid_mask)
            total_loss = total_loss + self.event_edge_weight * event_edge_loss
            aux["event_motion_density"] = event_motion_density.detach()
        else:
            event_edge_loss = depth_pred.new_tensor(0.0)
        details["event_edge_loss"] = float(event_edge_loss.detach())

        global_memory = getattr(model_output, "global_memory", None)
        if global_memory is not None:
            details["global_memory_abs"] = float(global_memory.detach().abs().mean())

        return total_loss, details, aux


def configure_global_local_trainable_params(model, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    enabled_prefixes = []
    if getattr(model, "use_local_branch", False):
        enabled_prefixes.extend(["event_encoder", "event_detail_refiner"])
    if getattr(model, "use_global_branch", False):
        enabled_prefixes.append("global_memory")

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


def build_global_local_optimizer_params(model, cfg):
    base_lr = float(cfg.lr)
    vggt_lr = float(getattr(cfg.train, "vggt_last_blocks_lr", base_lr))
    groups = {"event": [], "global": [], "heads": [], "vggt": [], "other": []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("event_encoder", "event_detail_refiner")):
            groups["event"].append(param)
        elif name.startswith("global_memory"):
            groups["global"].append(param)
        elif name.startswith("aggregator"):
            groups["vggt"].append(param)
        elif name.startswith(("camera_head", "depth_head", "point_head", "track_head")):
            groups["heads"].append(param)
        else:
            groups["other"].append(param)

    param_groups = []
    for key in ("event", "global", "heads", "other"):
        if groups[key]:
            param_groups.append({"params": groups[key], "lr_scale": 1.0})
    if groups["vggt"]:
        param_groups.append({"params": groups["vggt"], "lr_scale": vggt_lr / max(base_lr, 1e-12)})
    return param_groups


def launch(
    cfg,
    *,
    exp_name: Optional[str] = None,
    branch_mode: Optional[str] = None,
    num_global_tokens: Optional[int] = None,
    event_downsample: Optional[int] = None,
) -> None:
    if branch_mode is not None:
        cfg.model.branch_mode = branch_mode
    if num_global_tokens is not None:
        cfg.model.num_global_tokens = int(num_global_tokens)
    if event_downsample is not None:
        cfg.model.event_downsample = int(event_downsample)
    if exp_name is not None:
        _set_exp_paths(cfg, exp_name)

    OmegaConf.resolve(cfg)

    class ConfiguredGlobalLocalStreamVGGT(GlobalLocalStreamVGGT):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                branch_mode=str(getattr(cfg.model, "branch_mode", "global_local")),
                num_global_tokens=int(getattr(cfg.model, "num_global_tokens", 16)),
                event_downsample=int(getattr(cfg.model, "event_downsample", 4)),
                global_num_heads=int(getattr(cfg.model, "global_num_heads", 8)),
                global_inject_layers=list(getattr(cfg.model, "global_inject_layers", [23])),
                detail_hidden_dim=int(getattr(cfg.model, "detail_hidden_dim", 128)),
                residual_scale=float(getattr(cfg.model, "residual_scale", 0.1)),
                **kwargs,
            )

    class ConfiguredGlobalLocalLoss(GlobalLocalEventSupervisedLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                event_edge_weight=float(getattr(cfg.loss, "event_edge_weight", 0.0)),
                detail_weight=float(getattr(cfg.loss, "detail_weight", 0.0)),
                **kwargs,
            )

    fe.EventStreamVGGT = ConfiguredGlobalLocalStreamVGGT
    fe.EventSupervisedLoss = ConfiguredGlobalLocalLoss
    fe.configure_trainable_params = configure_global_local_trainable_params
    fe.build_optimizer_params = build_global_local_optimizer_params
    fe.train(cfg)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event_global_local.yaml",
)
def run(cfg: OmegaConf):
    launch(cfg)


if __name__ == "__main__":
    run()
