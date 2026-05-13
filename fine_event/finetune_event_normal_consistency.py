from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import finetune_event as fe


def stack_loaded_normals(
    views: List[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if all("normal_gt" in view for view in views):
        normals = fe.stack_view_field(views, "normal_gt")
    elif all("normal" in view for view in views):
        normals = fe.stack_view_field(views, "normal")
    else:
        return None, None

    normals = normals.to(device=device, dtype=dtype)
    if normals.ndim != 5:
        fe.printer.warning(
            "Skip loaded normals with unexpected ndim: expected 5D [B,S,H,W,3] or [B,S,3,H,W], got %s",
            tuple(normals.shape),
        )
        return None, None

    if normals.shape[2] == 3:
        normals = normals.permute(0, 1, 3, 4, 2)
    elif normals.shape[-1] == 3:
        pass
    else:
        fe.printer.warning(
            "Skip loaded normals with unexpected shape: expected [B,S,H,W,3] or [B,S,3,H,W], got %s",
            tuple(normals.shape),
        )
        return None, None

    raw_valid = torch.isfinite(normals).all(dim=-1) & (normals.abs().sum(dim=-1) > 1e-6)

    finite_normals = torch.where(torch.isfinite(normals), normals, torch.zeros_like(normals))
    normal_max = finite_normals.detach().abs().amax()
    normal_min = finite_normals.detach().amin()
    if normal_max > 2.0:
        finite_normals = finite_normals / 127.5 - 1.0
    elif normal_min >= 0.0:
        finite_normals = finite_normals * 2.0 - 1.0

    finite_normals = F.normalize(finite_normals, dim=-1, eps=1e-6)
    return finite_normals, raw_valid


class NormalConsistencyEventSupervisedLoss(fe.EventSupervisedLoss):
    def forward(
        self,
        model_output,
        views: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        pred = model_output.ress

        depth_pred = torch.stack([res["depth"] for res in pred], dim=1).squeeze(-1)
        points_pred = torch.stack([res["pts3d_in_other_view"] for res in pred], dim=1)
        pose_pred = torch.stack([res["camera_pose"] for res in pred], dim=1)

        depth_gt = fe.stack_view_field(views, "depthmap").to(device=depth_pred.device, dtype=depth_pred.dtype)
        intrinsics_gt = fe.stack_view_field(views, "camera_intrinsics").to(
            device=depth_pred.device,
            dtype=depth_pred.dtype,
        )
        pose_matrix_gt = fe.stack_view_field(views, "camera_pose").to(device=depth_pred.device, dtype=depth_pred.dtype)
        valid_mask = fe.build_valid_mask(views, depth_gt, depth_min=self.depth_min, depth_max=self.depth_max)

        if self.align_depth_scale_enabled:
            depth_gt_aligned, depth_scales = fe.align_depth_scale(depth_pred.detach(), depth_gt, valid_mask)
        else:
            depth_gt_aligned = depth_gt
            depth_scales = depth_gt.new_ones(depth_gt.shape[:2])

        points_gt = fe.depth_to_world_points(depth_gt_aligned, intrinsics_gt, pose_matrix_gt)
        points_mask = valid_mask.unsqueeze(-1).expand_as(points_gt)

        height, width = depth_gt.shape[-2:]
        pose_gt = fe.camera_pose_to_pose_encoding(
            pose_matrix_gt,
            intrinsics_gt,
            image_size_hw=(height, width),
        ).to(device=pose_pred.device, dtype=pose_pred.dtype)

        pose_first_pred = pose_pred[:, 0:1, :]
        pose_first_gt = pose_gt[:, 0:1, :]
        pose_alignment = pose_first_gt - pose_first_pred

        pose_pred_aligned = pose_pred.clone()
        pose_pred_aligned[:, 1:, :] = pose_pred[:, 1:, :] + pose_alignment
        if self.align_depth_scale_enabled:
            pose_gt[..., :3] = pose_gt[..., :3] * depth_scales.unsqueeze(-1)

        if pose_pred_aligned.shape[1] > 1:
            pose_loss = F.smooth_l1_loss(pose_pred_aligned[:, 1:, :], pose_gt[:, 1:, :])
        else:
            pose_loss = pose_pred.new_tensor(0.0, requires_grad=True)

        depth_loss = fe.masked_l1(depth_pred, depth_gt_aligned, valid_mask)
        if self.points_loss_type == "l1":
            points_loss = fe.masked_l1(points_pred, points_gt, points_mask)
        else:
            points_loss = fe.masked_chamfer_distance(points_pred, points_gt, valid_mask)

        normal_mask = valid_mask.clone()
        normal_mask[..., 0, :] = False
        normal_mask[..., -1, :] = False
        normal_mask[..., :, 0] = False
        normal_mask[..., :, -1] = False

        loaded_normals, loaded_normal_mask = stack_loaded_normals(views, depth_pred.device, depth_pred.dtype)
        if loaded_normals is None:
            loaded_normals = fe.depth_to_normals(depth_gt_aligned, intrinsics_gt)
            loaded_normal_mask = normal_mask
        else:
            loaded_normal_mask = loaded_normal_mask.to(device=normal_mask.device) & normal_mask

        pred_depth_normals = fe.depth_to_normals(depth_pred, intrinsics_gt)
        gt_depth_normals = fe.depth_to_normals(depth_gt_aligned, intrinsics_gt)

        normal_loss = fe.masked_cosine_loss(pred_depth_normals, loaded_normals, loaded_normal_mask)
        gt_depth_normal_consistency = fe.masked_cosine_loss(
            gt_depth_normals.detach(),
            loaded_normals.detach(),
            loaded_normal_mask,
        )

        total_loss = (
            self.pose_weight * pose_loss
            + self.depth_weight * depth_loss
            + self.points_weight * points_loss
            + self.normal_weight * normal_loss
        )

        details = {
            "pose_loss": float(pose_loss.detach()),
            "depth_loss": float(depth_loss.detach()),
            "points_loss": float(points_loss.detach()),
            "normal_loss": float(normal_loss.detach()),
            "gt_depth_normal_consistency": float(gt_depth_normal_consistency.detach()),
            "depth_scale": float(depth_scales.mean().detach()),
        }
        aux = {
            "depth_pred": depth_pred.detach(),
            "depth_gt": depth_gt.detach(),
            "depth_gt_aligned": depth_gt_aligned.detach(),
            "points_pred": points_pred.detach(),
            "points_gt": points_gt.detach(),
            "valid_mask": valid_mask.detach(),
            "pose_pred": pose_pred_aligned.detach(),
            "pose_gt": pose_gt.detach(),
            "normal_from_depth_pred": pred_depth_normals.detach(),
            "normal_from_depth_gt": gt_depth_normals.detach(),
            "normal_loaded": loaded_normals.detach(),
            "normal_loaded_mask": loaded_normal_mask.detach(),
        }
        return total_loss, details, aux


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event_normal_consistency.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    fe.EventSupervisedLoss = NormalConsistencyEventSupervisedLoss
    fe.train(cfg)


if __name__ == "__main__":
    run()
