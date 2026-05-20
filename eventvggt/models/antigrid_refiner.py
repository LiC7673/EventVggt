from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        groups = _group_count(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class AntiGridDepthPointRefiner(nn.Module):
    """Image-guided output refiner for patch-grid depth ripples.

    The branch predicts a bounded residual in log-depth.  Its final layer is
    zero-initialized, so a fresh model exactly matches the base StreamVGGT output
    before finetuning.  When point maps are refined, points are scaled by the
    same depth ratio to keep depth and point supervision consistent.
    """

    def __init__(
        self,
        image_channels: int = 3,
        hidden_dim: int = 48,
        num_blocks: int = 4,
        residual_scale: float = 0.05,
        refine_points: bool = True,
        min_depth: float = 1e-6,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.refine_points = bool(refine_points)
        self.min_depth = float(min_depth)

        input_channels = 1 + int(image_channels) + 4
        groups = _group_count(hidden_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )

        dilations = [1, 2, 3, 1]
        blocks = []
        for idx in range(max(1, int(num_blocks))):
            blocks.append(_ResidualBlock(hidden_dim, dilation=dilations[idx % len(dilations)]))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        depth: torch.Tensor,
        points: Optional[torch.Tensor],
        image: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if depth.ndim != 4 or depth.shape[-1] != 1:
            return depth, points

        depth_chw = depth.permute(0, 3, 1, 2).contiguous()
        image = self._prepare_image(image, depth_chw)
        log_depth = torch.log(depth_chw.clamp_min(self.min_depth))

        mean = log_depth.mean(dim=(-2, -1), keepdim=True)
        std = log_depth.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        depth_feat = (log_depth - mean) / std

        depth_gx = F.pad(depth_feat[..., :, 1:] - depth_feat[..., :, :-1], (1, 0, 0, 0))
        depth_gy = F.pad(depth_feat[..., 1:, :] - depth_feat[..., :-1, :], (0, 0, 1, 0))
        depth_high = depth_feat - F.avg_pool2d(depth_feat, kernel_size=3, stride=1, padding=1)

        gray = image.mean(dim=1, keepdim=True)
        img_gx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (1, 0, 0, 0))
        img_gy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 1, 0))
        img_grad = torch.sqrt(img_gx.square() + img_gy.square() + 1e-6)

        features = torch.cat([depth_feat, image, img_grad, depth_gx, depth_gy, depth_high], dim=1)
        delta_log_depth = torch.tanh(self.out(self.blocks(self.stem(features)))) * self.residual_scale
        refined_depth_chw = torch.exp(log_depth + delta_log_depth).clamp_min(self.min_depth)
        refined_depth = refined_depth_chw.permute(0, 2, 3, 1).contiguous()

        refined_points = points
        if self.refine_points and points is not None and points.ndim == 4 and points.shape[-1] == 3:
            ratio = (refined_depth / depth.clamp_min(self.min_depth)).to(dtype=points.dtype)
            refined_points = points * ratio

        return refined_depth.to(dtype=depth.dtype), refined_points

    def _prepare_image(self, image: Optional[torch.Tensor], depth_chw: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = depth_chw.shape
        if image is None:
            return depth_chw.new_zeros(batch, 3, height, width)

        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device=depth_chw.device, dtype=depth_chw.dtype)
        if image.shape[-2:] != (height, width):
            image = F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)
        return image


def refine_stream_output(
    output,
    views: Optional[List[dict]],
    refiner: AntiGridDepthPointRefiner,
):
    if output is None or getattr(output, "ress", None) is None:
        return output
    if views is None:
        views = getattr(output, "views", None)

    refined_ress = []
    for view_idx, res in enumerate(output.ress):
        new_res = dict(res)
        depth = new_res.get("depth")
        points = new_res.get("pts3d_in_other_view")
        image = None
        if views is not None and view_idx < len(views):
            image = views[view_idx].get("img")
        if depth is not None:
            depth, points = refiner(depth, points, image)
            new_res["depth"] = depth
            if points is not None:
                new_res["pts3d_in_other_view"] = points
        refined_ress.append(new_res)

    output.ress = refined_ress
    return output
