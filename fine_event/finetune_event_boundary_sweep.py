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
    boundary_sweep_loss,
    build_event_polarity_maps,
    flatten_pairs,
    predicted_normals_chw,
    stack_occupancy_from_views,
)


def _set_exp_paths(cfg, exp_name: str) -> None:
    cfg.exp_name = exp_name
    cfg.logdir = f"{cfg.save_dir}/{exp_name}/logs"
    cfg.output_dir = f"{cfg.save_dir}/{exp_name}"


class BoundarySweepEventSupervisedLoss(fe.EventSupervisedLoss):
    def __init__(
        self,
        *args,
        boundary_sweep_weight: float = 0.1,
        event_count_cmax: float = 3.0,
        use_occupancy_boundary: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.boundary_sweep_weight = float(boundary_sweep_weight)
        self.event_count_cmax = float(event_count_cmax)
        self.use_occupancy_boundary = bool(use_occupancy_boundary)

    def forward(
        self,
        model_output,
        views: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        total_loss, details, aux = super().forward(model_output, views)
        depth_pred = torch.stack([res["depth"] for res in model_output.ress], dim=1).squeeze(-1)

        if depth_pred.shape[1] <= 1 or self.boundary_sweep_weight <= 0:
            boundary_loss = depth_pred.new_tensor(0.0)
            details["boundary_sweep_loss"] = 0.0
            return total_loss, details, aux

        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(
            device=depth_pred.device,
            dtype=depth_pred.dtype,
        )
        valid_mask = aux["valid_mask"].to(device=depth_pred.device)
        height, width = depth_pred.shape[-2:]

        normals = predicted_normals_chw(depth_pred, intrinsics)
        event_pos, event_neg = build_event_polarity_maps(
            views,
            height=height,
            width=width,
            device=depth_pred.device,
            dtype=depth_pred.dtype,
            count_cmax=self.event_count_cmax,
        )

        occupancy = None
        if self.use_occupancy_boundary:
            occupancy = stack_occupancy_from_views(views, valid_mask)

        pair_valid = (valid_mask[:, :-1] | valid_mask[:, 1:]).unsqueeze(2)
        occ0 = flatten_pairs(occupancy, 0, -1) if occupancy is not None else None
        occ1 = flatten_pairs(occupancy, 1, None) if occupancy is not None else None
        boundary_loss = boundary_sweep_loss(
            depth0=flatten_pairs(depth_pred.unsqueeze(2), 0, -1),
            depth1=flatten_pairs(depth_pred.unsqueeze(2), 1, None),
            normal0=flatten_pairs(normals, 0, -1),
            normal1=flatten_pairs(normals, 1, None),
            event_pos=flatten_pairs(event_pos, 1, None),
            event_neg=flatten_pairs(event_neg, 1, None),
            occ0=occ0,
            occ1=occ1,
            valid_mask=flatten_pairs(pair_valid, 0, None),
        )

        total_loss = total_loss + self.boundary_sweep_weight * boundary_loss
        details["boundary_sweep_loss"] = float(boundary_loss.detach())
        details["boundary_sweep_weight"] = self.boundary_sweep_weight
        aux["event_pos"] = event_pos.detach()
        aux["event_neg"] = event_neg.detach()
        return total_loss, details, aux


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    _set_exp_paths(cfg, "event_boundary_sweep")
    cfg.loss.boundary_sweep_weight = float(getattr(cfg.loss, "boundary_sweep_weight", 0.1))
    cfg.loss.event_count_cmax = float(getattr(cfg.loss, "event_count_cmax", 3.0))
    cfg.loss.use_occupancy_boundary = bool(getattr(cfg.loss, "use_occupancy_boundary", True))
    OmegaConf.resolve(cfg)

    class ConfiguredBoundarySweepLoss(BoundarySweepEventSupervisedLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                boundary_sweep_weight=float(getattr(cfg.loss, "boundary_sweep_weight", 0.1)),
                event_count_cmax=float(getattr(cfg.loss, "event_count_cmax", 3.0)),
                use_occupancy_boundary=bool(getattr(cfg.loss, "use_occupancy_boundary", True)),
                **kwargs,
            )

    fe.EventSupervisedLoss = ConfiguredBoundarySweepLoss
    fe.train(cfg)


if __name__ == "__main__":
    run()
