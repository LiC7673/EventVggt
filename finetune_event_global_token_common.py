from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import finetune_event as fe
from eventvggt.models.streamvggt_global_token import StreamVGGT as GlobalTokenStreamVGGT


def normalized_token_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred.float(), dim=-1, eps=1e-6)
    target = F.normalize(target.float(), dim=-1, eps=1e-6)
    return F.smooth_l1_loss(pred, target)


def make_token_supervised_loss(event_token_weight: float, global_token_weight: float):
    class TokenSupervisedLoss(fe.EventSupervisedLoss):
        def forward(self, model_output, views):
            total_loss, details, aux = super().forward(model_output, views)

            if event_token_weight > 0.0:
                event_token_loss = normalized_token_loss(
                    model_output.global_token_event,
                    model_output.global_token_rgb.detach(),
                )
                total_loss = total_loss + event_token_weight * event_token_loss
                details["event_token_loss"] = float(event_token_loss.detach())

            if global_token_weight > 0.0:
                global_token_loss = normalized_token_loss(
                    model_output.global_token_refined,
                    model_output.global_token_target.detach(),
                )
                total_loss = total_loss + global_token_weight * global_token_loss
                details["global_token_loss"] = float(global_token_loss.detach())

            return total_loss, details, aux

    return TokenSupervisedLoss


def configure_global_token_trainable_params(model, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    enabled_prefixes = [
        "event_encoder",
        "event_patch_embed",
        "event_token_proj",
        "global_fusion",
    ]
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


def run_global_token_variant(cfg, event_token_weight: float, global_token_weight: float):
    OmegaConf.resolve(cfg)
    fe.EventStreamVGGT = GlobalTokenStreamVGGT
    fe.EventSupervisedLoss = make_token_supervised_loss(event_token_weight, global_token_weight)
    fe.configure_trainable_params = configure_global_token_trainable_params
    fe.train(cfg)


def make_hydra_main(event_token_weight: float, global_token_weight: float):
    @hydra.main(
        version_base=None,
        config_path=str(Path(__file__).resolve().parent / "config"),
        config_name="finetune_event.yaml",
    )
    def run(cfg: OmegaConf):
        loss_cfg = getattr(cfg, "loss", {})
        event_weight = float(getattr(loss_cfg, "event_token_weight", event_token_weight))
        global_weight = float(getattr(loss_cfg, "global_token_weight", global_token_weight))
        run_global_token_variant(cfg, event_weight, global_weight)

    return run
