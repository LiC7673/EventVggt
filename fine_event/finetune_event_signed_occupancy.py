from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
import torch
from omegaconf import OmegaConf

import finetune_event as fe
from fine_event.event_geometry_losses import (
    build_event_polarity_maps,
    flatten_pairs,
    signed_occupancy_event_loss,
    soft_occupancy_from_depth,
    stack_occupancy_from_views,
)


def _set_exp_paths(cfg, exp_name: str) -> None:
    cfg.exp_name = exp_name
    cfg.logdir = f"{cfg.save_dir}/{exp_name}/logs"
    cfg.output_dir = f"{cfg.save_dir}/{exp_name}"


class SignedOccupancyEventSupervisedLoss(fe.EventSupervisedLoss):
    def __init__(
        self,
        *args,
        signed_occupancy_weight: float = 0.1,
        event_count_cmax: float = 3.0,
        contrast_sign: float = 1.0,
        occupancy_source: str = "depth",
        occupancy_depth_threshold: float = 0.02,
        occupancy_temperature: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.signed_occupancy_weight = float(signed_occupancy_weight)
        self.event_count_cmax = float(event_count_cmax)
        self.contrast_sign = float(contrast_sign)
        self.occupancy_source = str(occupancy_source).lower()
        self.occupancy_depth_threshold = float(occupancy_depth_threshold)
        self.occupancy_temperature = float(occupancy_temperature)

    def _occupancy(
        self,
        depth_pred: torch.Tensor,
        views: List[Dict[str, torch.Tensor]],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.occupancy_source == "mask":
            return stack_occupancy_from_views(views, valid_mask).detach()
        if self.occupancy_source != "depth":
            raise ValueError("occupancy_source must be 'depth' or 'mask'")
        return soft_occupancy_from_depth(
            depth_pred,
            threshold=self.occupancy_depth_threshold,
            temperature=self.occupancy_temperature,
        )

    def forward(
        self,
        model_output,
        views: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        total_loss, details, aux = super().forward(model_output, views)
        depth_pred = torch.stack([res["depth"] for res in model_output.ress], dim=1).squeeze(-1)

        if depth_pred.shape[1] <= 1 or self.signed_occupancy_weight <= 0:
            signed_loss = depth_pred.new_tensor(0.0)
            details["signed_occupancy_event_loss"] = 0.0
            return total_loss, details, aux

        valid_mask = aux["valid_mask"].to(device=depth_pred.device)
        height, width = depth_pred.shape[-2:]
        event_pos, event_neg = build_event_polarity_maps(
            views,
            height=height,
            width=width,
            device=depth_pred.device,
            dtype=depth_pred.dtype,
            count_cmax=self.event_count_cmax,
        )
        occupancy = self._occupancy(depth_pred, views, valid_mask)
        pair_valid = (valid_mask[:, :-1] | valid_mask[:, 1:]).unsqueeze(2)

        signed_loss = signed_occupancy_event_loss(
            event_pos=flatten_pairs(event_pos, 1, None),
            event_neg=flatten_pairs(event_neg, 1, None),
            occ0=flatten_pairs(occupancy, 0, -1),
            occ1=flatten_pairs(occupancy, 1, None),
            contrast_sign=self.contrast_sign,
            valid_mask=flatten_pairs(pair_valid, 0, None),
        )

        total_loss = total_loss + self.signed_occupancy_weight * signed_loss
        details["signed_occupancy_event_loss"] = float(signed_loss.detach())
        details["signed_occupancy_weight"] = self.signed_occupancy_weight
        aux["event_pos"] = event_pos.detach()
        aux["event_neg"] = event_neg.detach()
        aux["occupancy_pred"] = occupancy.detach()
        return total_loss, details, aux


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    _set_exp_paths(cfg, "event_signed_occupancy")
    cfg.loss.signed_occupancy_weight = float(getattr(cfg.loss, "signed_occupancy_weight", 0.1))
    cfg.loss.event_count_cmax = float(getattr(cfg.loss, "event_count_cmax", 3.0))
    cfg.loss.contrast_sign = float(getattr(cfg.loss, "contrast_sign", 1.0))
    cfg.loss.occupancy_source = str(getattr(cfg.loss, "occupancy_source", "depth"))
    cfg.loss.occupancy_depth_threshold = float(getattr(cfg.loss, "occupancy_depth_threshold", 0.02))
    cfg.loss.occupancy_temperature = float(getattr(cfg.loss, "occupancy_temperature", 0.02))
    OmegaConf.resolve(cfg)

    class ConfiguredSignedOccupancyLoss(SignedOccupancyEventSupervisedLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                signed_occupancy_weight=float(getattr(cfg.loss, "signed_occupancy_weight", 0.1)),
                event_count_cmax=float(getattr(cfg.loss, "event_count_cmax", 3.0)),
                contrast_sign=float(getattr(cfg.loss, "contrast_sign", 1.0)),
                occupancy_source=str(getattr(cfg.loss, "occupancy_source", "depth")),
                occupancy_depth_threshold=float(getattr(cfg.loss, "occupancy_depth_threshold", 0.02)),
                occupancy_temperature=float(getattr(cfg.loss, "occupancy_temperature", 0.02)),
                **kwargs,
            )

    fe.EventSupervisedLoss = ConfiguredSignedOccupancyLoss
    fe.train(cfg)


if __name__ == "__main__":
    run()
