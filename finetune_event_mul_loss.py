from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import finetune_event as fe


def _ensure_homogeneous_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-2:] == (3, 4):
        bottom_row = torch.tensor([0, 0, 0, 1], device=pose.device, dtype=pose.dtype)
        bottom_row = bottom_row.view(*([1] * (pose.ndim - 2)), 1, 4).expand(*pose.shape[:-2], 1, 4)
        pose = torch.cat([pose, bottom_row], dim=-2)
    if pose.shape[-2:] != (4, 4):
        raise ValueError(f"Expected pose shape [...,4,4] or [...,3,4], got {tuple(pose.shape)}")
    return pose


def _normalize_grid(coord: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return torch.zeros_like(coord)
    return 2.0 * coord / (size - 1) - 1.0


def _sample_pixel_map(pixel_map: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _, height, width = pixel_map.shape
    grid = torch.stack(
        [
            _normalize_grid(u, width),
            _normalize_grid(v, height),
        ],
        dim=-1,
    ).view(pixel_map.shape[0], height, width, 2)
    return F.grid_sample(
        pixel_map.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1)


def _ordered_view_pairs(num_views: int, device: torch.device) -> torch.Tensor:
    pairs = [(src, tgt) for src in range(num_views) for tgt in range(num_views) if src != tgt]
    return torch.tensor(pairs, device=device, dtype=torch.long)


def random_pixel_mapping_loss(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_pairs: int,
    depth_min: float = 1e-6,
    detach_target: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if depth.ndim != 4:
        raise ValueError(f"Expected depth shape [B,S,H,W], got {tuple(depth.shape)}")
    if depth.shape[1] < 2 or num_pairs <= 0:
        zero = depth.new_tensor(0.0)
        return zero, {
            "mapping_pairs": depth.new_tensor(0.0),
            "mapping_valid_ratio": depth.new_tensor(0.0),
        }

    batch, seq, height, width = depth.shape

    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    xs = xs.view(1, height, width)
    ys = ys.view(1, height, width)

    camera_pose = _ensure_homogeneous_pose(camera_pose)
    world_to_camera = torch.linalg.inv(camera_pose)

    all_pairs = _ordered_view_pairs(seq, depth.device)
    pair_count = min(int(num_pairs), int(all_pairs.shape[0]))
    selected = all_pairs[torch.randperm(all_pairs.shape[0], device=depth.device)[:pair_count]]

    pair_losses = []
    valid_ratios = []

    for src_idx, tgt_idx in selected.tolist():
        src_depth = depth[:, src_idx]
        tgt_depth = depth[:, tgt_idx]
        if detach_target:
            tgt_depth = tgt_depth.detach()

        src_valid = valid_mask[:, src_idx].to(device=depth.device)
        tgt_valid = valid_mask[:, tgt_idx].to(device=depth.device, dtype=depth.dtype)

        src_k = intrinsics[:, src_idx]
        tgt_k = intrinsics[:, tgt_idx]
        src_c2w = camera_pose[:, src_idx]
        tgt_w2c = world_to_camera[:, tgt_idx]

        fx_src = src_k[:, 0, 0].view(batch, 1, 1).clamp_min(1e-6)
        fy_src = src_k[:, 1, 1].view(batch, 1, 1).clamp_min(1e-6)
        cx_src = src_k[:, 0, 2].view(batch, 1, 1)
        cy_src = src_k[:, 1, 2].view(batch, 1, 1)

        x_src = (xs - cx_src) * src_depth / fx_src
        y_src = (ys - cy_src) * src_depth / fy_src
        z_src = src_depth
        src_points = torch.stack([x_src, y_src, z_src], dim=-1)

        src_rot = src_c2w[:, :3, :3]
        src_trans = src_c2w[:, :3, 3]
        world_points = torch.einsum("bij,bhwj->bhwi", src_rot, src_points) + src_trans.view(batch, 1, 1, 3)

        tgt_rot = tgt_w2c[:, :3, :3]
        tgt_trans = tgt_w2c[:, :3, 3]
        tgt_points = torch.einsum("bij,bhwj->bhwi", tgt_rot, world_points) + tgt_trans.view(batch, 1, 1, 3)
        z_tgt_from_src = tgt_points[..., 2]

        fx_tgt = tgt_k[:, 0, 0].view(batch, 1, 1)
        fy_tgt = tgt_k[:, 1, 1].view(batch, 1, 1)
        cx_tgt = tgt_k[:, 0, 2].view(batch, 1, 1)
        cy_tgt = tgt_k[:, 1, 2].view(batch, 1, 1)

        z_safe = z_tgt_from_src.clamp_min(1e-6)
        u_tgt = fx_tgt * tgt_points[..., 0] / z_safe + cx_tgt
        v_tgt = fy_tgt * tgt_points[..., 1] / z_safe + cy_tgt

        u_sample = u_tgt.clamp(0, width - 1)
        v_sample = v_tgt.clamp(0, height - 1)
        sampled_tgt_depth = _sample_pixel_map(tgt_depth, u_sample, v_sample)
        sampled_tgt_valid = _sample_pixel_map(tgt_valid, u_sample.detach(), v_sample.detach()) > 0.5

        inside = (u_tgt >= 0) & (u_tgt <= width - 1) & (v_tgt >= 0) & (v_tgt <= height - 1)
        pair_valid = (
            src_valid
            & sampled_tgt_valid
            & inside
            & torch.isfinite(z_tgt_from_src)
            & torch.isfinite(sampled_tgt_depth)
            & (src_depth > depth_min)
            & (sampled_tgt_depth > depth_min)
            & (z_tgt_from_src > depth_min)
        )
        valid_float = pair_valid.to(dtype=depth.dtype)
        valid_count = valid_float.sum()
        valid_ratios.append(valid_float.mean())

        if valid_count <= 0:
            continue

        pair_loss = F.smooth_l1_loss(
            z_tgt_from_src[pair_valid],
            sampled_tgt_depth[pair_valid],
            reduction="mean",
        )
        pair_losses.append(pair_loss)

    if not pair_losses:
        zero = depth.new_tensor(0.0)
        return zero, {
            "mapping_pairs": depth.new_tensor(float(pair_count)),
            "mapping_valid_ratio": torch.stack(valid_ratios).mean() if valid_ratios else zero,
        }

    return torch.stack(pair_losses).mean(), {
        "mapping_pairs": depth.new_tensor(float(pair_count)),
        "mapping_valid_ratio": torch.stack(valid_ratios).mean(),
    }


class MultiViewMappingEventSupervisedLoss(fe.EventSupervisedLoss):
    mapping_weight_default = 0.2
    mapping_num_pairs_default = 4
    mapping_detach_target_default = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mapping_weight = float(self.mapping_weight_default)
        self.mapping_num_pairs = int(self.mapping_num_pairs_default)
        self.mapping_detach_target = bool(self.mapping_detach_target_default)

    def forward(
        self,
        model_output,
        views: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        total_loss, details, aux = super().forward(model_output, views)

        if self.mapping_weight <= 0.0 or self.mapping_num_pairs <= 0:
            details["mapping_loss"] = 0.0
            details["mapping_weighted_loss"] = 0.0
            details["mapping_pairs"] = 0.0
            details["mapping_valid_ratio"] = 0.0
            return total_loss, details, aux

        depth_pred = torch.stack([res["depth"] for res in model_output.ress], dim=1).squeeze(-1)
        depth_pred = depth_pred.to(device=total_loss.device)
        valid_mask = aux["valid_mask"].to(device=depth_pred.device)
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(device=depth_pred.device, dtype=depth_pred.dtype)
        camera_pose = fe.stack_view_field(views, "camera_pose").to(device=depth_pred.device, dtype=depth_pred.dtype)

        mapping_loss, mapping_stats = random_pixel_mapping_loss(
            depth_pred,
            intrinsics,
            camera_pose,
            valid_mask,
            num_pairs=self.mapping_num_pairs,
            depth_min=self.depth_min,
            detach_target=self.mapping_detach_target,
        )

        total_loss = total_loss + self.mapping_weight * mapping_loss
        details["mapping_loss"] = float(mapping_loss.detach())
        details["mapping_weighted_loss"] = float((self.mapping_weight * mapping_loss).detach())
        details["mapping_pairs"] = float(mapping_stats["mapping_pairs"].detach())
        details["mapping_valid_ratio"] = float(mapping_stats["mapping_valid_ratio"].detach())
        return total_loss, details, aux


def _configure_mapping_loss_from_cfg(cfg) -> None:
    loss_cfg = getattr(cfg, "loss", {})

    MultiViewMappingEventSupervisedLoss.mapping_weight_default = float(
        getattr(loss_cfg, "mapping_weight", 0.2)
    )
    MultiViewMappingEventSupervisedLoss.mapping_num_pairs_default = int(
        getattr(loss_cfg, "mapping_num_pairs", getattr(loss_cfg, "nums_views", 4))
    )
    MultiViewMappingEventSupervisedLoss.mapping_detach_target_default = bool(
        getattr(loss_cfg, "mapping_detach_target", False)
    )


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "config"),
    config_name="finetune_event_mul_loss.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    _configure_mapping_loss_from_cfg(cfg)
    fe.printer.info(
        "Pixel projection mapping loss: weight=%.4f, pairs=%d, detach_target=%s",
        MultiViewMappingEventSupervisedLoss.mapping_weight_default,
        MultiViewMappingEventSupervisedLoss.mapping_num_pairs_default,
        MultiViewMappingEventSupervisedLoss.mapping_detach_target_default,
    )
    fe.EventSupervisedLoss = MultiViewMappingEventSupervisedLoss
    fe.train(cfg)


if __name__ == "__main__":
    run()
