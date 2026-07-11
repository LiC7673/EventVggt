"""Stage-1 multi-LDR event-contribution learning.

This module intentionally does *not* implement the legacy reliability-label
distillation path.  A frozen RGB model produces coarse geometry from the bad
exposure, ``ContributionNet`` selects the useful part of the event voxel, and
``SelectedEventRefiner`` is allowed to see only that selected voxel.

Tensor conventions used here are:

* RGB: ``[batch, views, 3, height, width]`` in ``[0, 1]``.
* event voxel: ``[batch, views, 2 * bins, height, width]``.
* depth/contribution/masks: ``[batch, views, height, width]``.
* normals: ``[batch, views, height, width, 3]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_event_voxel(voxel: torch.Tensor, count_cmax: float = 3.0) -> torch.Tensor:
    """Log-compress separate non-negative polarity/bin count channels."""
    ceiling = max(float(count_cmax), 1.0)
    denominator = math.log1p(ceiling)
    return torch.log1p(voxel.float().clamp(0.0, ceiling)) / denominator


def _saturated_pixels(rgb: torch.Tensor, threshold: float, mode: str) -> torch.Tensor:
    normalized = str(mode).strip().lower()
    if normalized == "any_channel":
        return rgb.float().amax(dim=2) >= float(threshold)
    if normalized == "all_channels":
        return rgb.float().amin(dim=2) >= float(threshold)
    if normalized == "luminance":
        coefficients = rgb.new_tensor((0.299, 0.587, 0.114)).view(1, 1, 3, 1, 1)
        return (rgb.float() * coefficients).sum(dim=2) >= float(threshold)
    raise ValueError(f"Unknown saturation mode: {mode!r}")


def saturation_ratio(
    rgb: torch.Tensor, threshold: float = 0.98, mode: str = "any_channel"
) -> torch.Tensor:
    """Return the saturated-pixel ratio for every item in a batch."""
    if rgb.ndim != 5 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB [B,S,3,H,W], got {tuple(rgb.shape)}")
    saturated = _saturated_pixels(rgb, threshold, mode)
    return saturated.flatten(1).float().mean(dim=1)


def _central_gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central spatial differences with a zero-valued one-pixel border."""
    grad_x = torch.zeros_like(value)
    grad_y = torch.zeros_like(value)
    grad_x[..., 1:-1] = 0.5 * (value[..., 2:] - value[..., :-2])
    grad_y[..., 1:-1, :] = 0.5 * (value[..., 2:, :] - value[..., :-2, :])
    return grad_x, grad_y


def image_gradient_magnitude(rgb: torch.Tensor) -> torch.Tensor:
    """Luminance-gradient magnitude for RGB ``[B,S,3,H,W]``."""
    coefficients = rgb.new_tensor((0.299, 0.587, 0.114)).view(1, 1, 3, 1, 1)
    gray = (rgb.float() * coefficients).sum(dim=2)
    grad_x, grad_y = _central_gradient(gray)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-12)


def orient_exposure_pair(
    rgb_a: torch.Tensor,
    rgb_b: torch.Tensor,
    *,
    saturation_threshold: float = 0.98,
    saturation_mode: str = "any_channel",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orient a pair as ``reference -> bad`` using measured saturation.

    Returns ``(rgb_ref, rgb_bad, ref_is_a, sat_ref, sat_bad)``.  This avoids
    assuming that the numerical LDR id always has the same photometric order.
    """
    if rgb_a.shape != rgb_b.shape:
        raise ValueError(f"Paired RGB shapes differ: {rgb_a.shape} versus {rgb_b.shape}")
    sat_a = saturation_ratio(rgb_a, saturation_threshold, saturation_mode)
    sat_b = saturation_ratio(rgb_b, saturation_threshold, saturation_mode)
    ref_is_a = sat_a <= sat_b
    selector = ref_is_a.view(-1, 1, 1, 1, 1)
    rgb_ref = torch.where(selector, rgb_a, rgb_b)
    rgb_bad = torch.where(selector, rgb_b, rgb_a)
    sat_ref = torch.minimum(sat_a, sat_b)
    sat_bad = torch.maximum(sat_a, sat_b)
    return rgb_ref, rgb_bad, ref_is_a, sat_ref, sat_bad


@dataclass
class BridgeMasks:
    saturated_bad: torch.Tensor
    visible_reference: torch.Tensor
    event_support: torch.Tensor
    bridge: torch.Tensor
    area: torch.Tensor


def build_bridge_masks(
    rgb_reference: torch.Tensor,
    rgb_bad: torch.Tensor,
    event_voxel: torch.Tensor,
    *,
    saturation_threshold: float = 0.98,
    reference_gradient_threshold: float = 0.02,
    require_reference_gradient: bool = True,
    event_support_dilate_kernel: int = 1,
    saturation_mode: str = "any_channel",
) -> BridgeMasks:
    """Construct the exposure bridge where event usefulness is identifiable."""
    if rgb_reference.shape != rgb_bad.shape:
        raise ValueError("Reference and bad RGB tensors must have the same shape.")
    expected = (*rgb_bad.shape[:2], rgb_bad.shape[-2], rgb_bad.shape[-1])
    if event_voxel.shape[:2] != rgb_bad.shape[:2] or event_voxel.shape[-2:] != rgb_bad.shape[-2:]:
        raise ValueError(
            f"Event/RGB shape mismatch: event={tuple(event_voxel.shape)}, rgb={tuple(rgb_bad.shape)}"
        )
    saturated_bad = _saturated_pixels(rgb_bad, saturation_threshold, saturation_mode)
    visible_reference = ~_saturated_pixels(
        rgb_reference, saturation_threshold, saturation_mode
    )
    if require_reference_gradient:
        visible_reference = visible_reference & (
            image_gradient_magnitude(rgb_reference) > float(reference_gradient_threshold)
        )
    event_support = event_voxel.float().abs().sum(dim=2) > 0.0
    kernel = max(int(event_support_dilate_kernel), 1)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        batch, views, height, width = event_support.shape
        event_support = F.max_pool2d(
            event_support.float().reshape(batch * views, 1, height, width),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).reshape(batch, views, height, width).bool()
    bridge = saturated_bad & visible_reference & event_support
    if bridge.shape != expected:
        raise RuntimeError(f"Unexpected bridge shape {bridge.shape}; expected {expected}")
    area = bridge.flatten(1).float().mean(dim=1)
    return BridgeMasks(saturated_bad, visible_reference, event_support, bridge, area)


def _gradient_magnitude(value: torch.Tensor) -> torch.Tensor:
    """Spatial gradient norm for scalar or channel-last vector maps."""
    if value.ndim == 5 and value.shape[-1] in (2, 3, 4):
        grad_x, grad_y = _central_gradient(value.movedim(-1, -3))
        return torch.sqrt(
            grad_x.square().sum(dim=-3) + grad_y.square().sum(dim=-3) + 1.0e-12
        )
    grad_x, grad_y = _central_gradient(value)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-12)


def robust_unit_map(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    quantile: float = 0.95,
) -> torch.Tensor:
    """Per-frame robust normalization that excludes invalid pixels."""
    if value.shape != valid_mask.shape:
        raise ValueError(f"value/mask shape mismatch: {value.shape} versus {valid_mask.shape}")
    batch, views = value.shape[:2]
    output = torch.zeros_like(value, dtype=torch.float32)
    for batch_index in range(batch):
        for view_index in range(views):
            current = value[batch_index, view_index].float()
            selected = current[valid_mask[batch_index, view_index].bool()]
            if selected.numel() == 0:
                continue
            scale = torch.quantile(selected, float(quantile)).clamp_min(1.0e-6)
            output[batch_index, view_index] = (current / scale).clamp(0.0, 1.0)
    return output


def geometry_emphasis_weight(
    depth_gt: torch.Tensor,
    normal_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float = 2.0,
    depth_gradient_weight: float = 0.5,
    quantile: float = 0.95,
) -> torch.Tensor:
    """Soft geometry emphasis; it is a weight, never a binary reliability label."""
    log_depth = depth_gt.float().clamp_min(1.0e-6).log()
    detail = _gradient_magnitude(normal_gt.float())
    detail = detail + float(depth_gradient_weight) * _gradient_magnitude(log_depth)
    normalized = robust_unit_map(detail, valid_mask.bool(), quantile=quantile)
    return (1.0 + float(alpha) * normalized).detach()


def weighted_mean(value: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(float(eps))


def contribution_budget(
    contribution: torch.Tensor,
    event_voxel: torch.Tensor,
    *,
    target_ratio: float = 0.5,
    sample_mask: Optional[torch.Tensor] = None,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Event-mass weighted selection budget, computed per sample."""
    event_mass = event_voxel.float().abs().sum(dim=2)
    numerator = (event_mass * contribution.float()).flatten(1).sum(dim=1)
    denominator = event_mass.flatten(1).sum(dim=1).clamp_min(float(eps))
    mean_contribution = numerator / denominator
    loss_per_sample = (mean_contribution - float(target_ratio)).abs()
    if sample_mask is not None:
        selected = sample_mask.bool()
        if selected.any():
            return loss_per_sample[selected].mean(), mean_contribution
        return loss_per_sample.sum() * 0.0, mean_contribution
    return loss_per_sample.mean(), mean_contribution


@dataclass
class ContributionLossOutput:
    loss: torch.Tensor
    depth_loss: torch.Tensor
    normal_loss: torch.Tensor
    budget_loss: torch.Tensor
    contribution_mean: torch.Tensor
    active_samples: torch.Tensor
    active_pixels: torch.Tensor


def stage1_contribution_loss(
    refined_depth: torch.Tensor,
    refined_normals: torch.Tensor,
    depth_gt: torch.Tensor,
    normal_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    bridge_mask: torch.Tensor,
    geometry_weight: torch.Tensor,
    contribution: torch.Tensor,
    event_voxel: torch.Tensor,
    *,
    minimum_bridge_area: float = 0.002,
    minimum_saturation_gap: float = 0.0,
    saturation_gap: Optional[torch.Tensor] = None,
    normal_weight: float = 0.25,
    budget_weight: float = 0.05,
    budget_ratio: float = 0.5,
) -> ContributionLossOutput:
    """Fixed-GT Stage-1 objective evaluated only in a valid bridge region."""
    bridge_area = bridge_mask.flatten(1).float().mean(dim=1)
    active_samples = bridge_area >= float(minimum_bridge_area)
    if saturation_gap is not None:
        active_samples = active_samples & (saturation_gap >= float(minimum_saturation_gap))
    active_selector = active_samples.view(-1, *([1] * (bridge_mask.ndim - 1)))
    active_pixels = bridge_mask.bool() & valid_mask.bool() & active_selector
    pixel_weight = geometry_weight.float() * active_pixels.float()

    log_error = (
        refined_depth.float().clamp_min(1.0e-6).log()
        - depth_gt.float().clamp_min(1.0e-6).log()
    ).abs()
    depth_loss = weighted_mean(log_error, pixel_weight)

    predicted = F.normalize(refined_normals.float(), dim=-1, eps=1.0e-6)
    target = F.normalize(normal_gt.float(), dim=-1, eps=1.0e-6)
    normal_valid = (refined_normals.float().norm(dim=-1) > 0.5) & (normal_gt.float().norm(dim=-1) > 0.5)
    cosine_error = 1.0 - (predicted * target).sum(dim=-1).clamp(-1.0, 1.0)
    normal_loss = weighted_mean(cosine_error, pixel_weight * normal_valid.float())

    budget_loss, mean_contribution = contribution_budget(
        contribution,
        event_voxel,
        target_ratio=budget_ratio,
        sample_mask=active_samples,
    )
    total = depth_loss + float(normal_weight) * normal_loss + float(budget_weight) * budget_loss
    return ContributionLossOutput(
        total,
        depth_loss,
        normal_loss,
        budget_loss,
        mean_contribution,
        active_samples,
        active_pixels,
    )


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1),
            nn.GroupNorm(_group_count(channels_out), channels_out),
            nn.GELU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1),
            nn.GroupNorm(_group_count(channels_out), channels_out),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class ContributionNet(nn.Module):
    """Predict dense event contribution from events, bad RGB, and frozen geometry."""

    def __init__(
        self,
        num_bins: int = 10,
        base_channels: int = 32,
        coarse_feature_dim: int = 0,
        count_cmax: float = 3.0,
        initial_contribution: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.count_cmax = float(count_cmax)
        self.coarse_feature_dim = int(coarse_feature_dim)
        local_channels = 2 * self.num_bins + 3 + 1 + 3
        self.input_block = ConvBlock(local_channels, base_channels)
        self.down_1 = nn.Sequential(nn.Conv2d(base_channels, 2 * base_channels, 3, stride=2, padding=1), nn.GELU())
        self.encoder_1 = ConvBlock(2 * base_channels, 2 * base_channels)
        self.down_2 = nn.Sequential(nn.Conv2d(2 * base_channels, 4 * base_channels, 3, stride=2, padding=1), nn.GELU())
        self.bottleneck = ConvBlock(4 * base_channels, 4 * base_channels)
        self.up_1 = ConvBlock(6 * base_channels, 2 * base_channels)
        self.up_2 = ConvBlock(3 * base_channels, base_channels)
        if self.coarse_feature_dim > 0:
            self.feature_projection = nn.Conv2d(self.coarse_feature_dim, base_channels, kernel_size=1)
            output_channels = 2 * base_channels
        else:
            self.feature_projection = None
            output_channels = base_channels
        self.output = nn.Conv2d(output_channels, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        probability = min(max(float(initial_contribution), 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(self.output.bias, math.log(probability / (1.0 - probability)))

    @staticmethod
    def _coarse_depth_input(depth: torch.Tensor) -> torch.Tensor:
        log_depth = depth.float().clamp_min(1.0e-6).log()
        center = log_depth.flatten(2).median(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        return (log_depth - center).clamp(-4.0, 4.0)

    def forward(
        self,
        event_voxel: torch.Tensor,
        bad_rgb: torch.Tensor,
        coarse_depth: torch.Tensor,
        coarse_normals: torch.Tensor,
        coarse_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, views, channels, height, width = event_voxel.shape
        if channels != 2 * self.num_bins:
            raise ValueError(f"Expected {2 * self.num_bins} event channels, got {channels}")
        event = normalize_event_voxel(event_voxel, self.count_cmax)
        normals = coarse_normals.detach().movedim(-1, -3).float()
        local = torch.cat(
            (
                event,
                bad_rgb.detach().float(),
                self._coarse_depth_input(coarse_depth.detach()).unsqueeze(2),
                normals,
            ),
            dim=2,
        ).reshape(batch * views, -1, height, width)
        skip_0 = self.input_block(local)
        skip_1 = self.encoder_1(self.down_1(skip_0))
        encoded = self.bottleneck(self.down_2(skip_1))
        decoded_1 = F.interpolate(encoded, size=skip_1.shape[-2:], mode="bilinear", align_corners=False)
        decoded_1 = self.up_1(torch.cat((decoded_1, skip_1), dim=1))
        decoded_0 = F.interpolate(decoded_1, size=skip_0.shape[-2:], mode="bilinear", align_corners=False)
        decoded_0 = self.up_2(torch.cat((decoded_0, skip_0), dim=1))

        if self.feature_projection is not None:
            if coarse_features is None:
                raise ValueError("coarse_features are required by this ContributionNet checkpoint")
            features = coarse_features.detach().reshape(
                batch * views, coarse_features.shape[2], coarse_features.shape[3], coarse_features.shape[4]
            )
            features = self.feature_projection(features.float())
            features = F.interpolate(features, size=(height, width), mode="bilinear", align_corners=False)
            decoded_0 = torch.cat((decoded_0, features), dim=1)
        logits = self.output(decoded_0)
        return torch.sigmoid(logits).reshape(batch, views, height, width)


class SelectedEventRefiner(nn.Module):
    """Refine frozen RGB geometry from the selected voxel and nothing else."""

    def __init__(
        self,
        num_bins: int = 10,
        hidden_channels: int = 32,
        count_cmax: float = 3.0,
        max_log_depth_delta: float = 0.20,
        max_normal_delta: float = 0.50,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.count_cmax = float(count_cmax)
        self.max_log_depth_delta = float(max_log_depth_delta)
        self.max_normal_delta = float(max_normal_delta)
        polarity_channels = max(hidden_channels // 2, 8)
        self.positive_encoder = nn.Sequential(
            nn.Conv3d(2, polarity_channels, (3, 5, 5), padding=(1, 2, 2)),
            nn.GroupNorm(_group_count(polarity_channels), polarity_channels),
            nn.GELU(),
        )
        self.negative_encoder = nn.Sequential(
            nn.Conv3d(2, polarity_channels, (3, 5, 5), padding=(1, 2, 2)),
            nn.GroupNorm(_group_count(polarity_channels), polarity_channels),
            nn.GELU(),
        )
        self.temporal_fusion = nn.Sequential(
            nn.Conv3d(2 * polarity_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.temporal_attention = nn.Conv3d(hidden_channels, 1, 1)
        self.geometry_refiner = ConvBlock(hidden_channels + 1 + 3, hidden_channels)
        self.output = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _coarse_depth_input(depth: torch.Tensor) -> torch.Tensor:
        log_depth = depth.float().clamp_min(1.0e-6).log()
        center = log_depth.flatten(2).median(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        return (log_depth - center).clamp(-4.0, 4.0)

    def _encode_selected_events(self, selected_event: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = selected_event.shape
        bins = channels // 2
        if bins != self.num_bins:
            raise ValueError(f"Expected {2 * self.num_bins} selected-event channels, got {channels}")
        normalized = normalize_event_voxel(selected_event, self.count_cmax)
        positive = normalized[:, :, :bins]
        negative = normalized[:, :, bins:]
        time = torch.linspace(-1.0, 1.0, bins, device=selected_event.device, dtype=normalized.dtype)
        time = time.view(1, 1, bins, 1, 1)
        positive = torch.stack((positive, positive * time), dim=2).reshape(
            batch * views, 2, bins, height, width
        )
        negative = torch.stack((negative, negative * time), dim=2).reshape(
            batch * views, 2, bins, height, width
        )
        encoded = torch.cat((self.positive_encoder(positive), self.negative_encoder(negative)), dim=1)
        encoded = self.temporal_fusion(encoded)
        attention = torch.softmax(self.temporal_attention(encoded), dim=2)
        return (encoded * attention).sum(dim=2)

    def forward(
        self,
        selected_event: torch.Tensor,
        coarse_depth: torch.Tensor,
        coarse_normals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, views, _, height, width = selected_event.shape
        event_features = self._encode_selected_events(selected_event)
        depth_input = self._coarse_depth_input(coarse_depth.detach()).reshape(batch * views, 1, height, width)
        normal_input = coarse_normals.detach().movedim(-1, -3).reshape(batch * views, 3, height, width)
        raw_delta = self.output(self.geometry_refiner(torch.cat((event_features, depth_input, normal_input), dim=1)))

        # This continuous architectural gate prevents the refiner from becoming
        # a second RGB-only decoder. Zero selected events return coarse RGB
        # exactly, and a near-zero C cannot unlock a full bias-only correction.
        selected_strength = selected_event.abs().amax(dim=2).clamp(0.0, 1.0)
        strength = selected_strength.reshape(batch * views, 1, height, width)
        delta_log_depth = self.max_log_depth_delta * torch.tanh(raw_delta[:, :1]) * strength
        delta_normal = self.max_normal_delta * torch.tanh(raw_delta[:, 1:]) * strength
        delta_log_depth = delta_log_depth.reshape(batch, views, height, width)
        delta_normal = delta_normal.reshape(batch, views, 3, height, width).movedim(-3, -1)
        refined_depth = coarse_depth.detach().float() * torch.exp(delta_log_depth.float())
        refined_normals = F.normalize(coarse_normals.detach().float() + delta_normal.float(), dim=-1, eps=1.0e-6)
        return {
            "depth": refined_depth,
            "normals": refined_normals,
            "delta_log_depth": delta_log_depth,
            "delta_normal": delta_normal,
            "selected_strength": selected_strength,
        }


class MultiLdrEventContributionModel(nn.Module):
    """ContributionNet and proxy refiner trained by the isolated A/B/C schedule."""

    checkpoint_schema = "multi_ldr_event_contribution_v1"

    def __init__(
        self,
        num_bins: int = 10,
        contribution_channels: int = 32,
        refiner_channels: int = 32,
        coarse_feature_dim: int = 0,
        count_cmax: float = 3.0,
        initial_contribution: float = 0.5,
        max_log_depth_delta: float = 0.20,
        max_normal_delta: float = 0.50,
    ) -> None:
        super().__init__()
        self.architecture = {
            "num_bins": int(num_bins),
            "contribution_channels": int(contribution_channels),
            "refiner_channels": int(refiner_channels),
            "coarse_feature_dim": int(coarse_feature_dim),
            "count_cmax": float(count_cmax),
            "initial_contribution": float(initial_contribution),
            "max_log_depth_delta": float(max_log_depth_delta),
            "max_normal_delta": float(max_normal_delta),
        }
        self.contribution_net = ContributionNet(
            num_bins=num_bins,
            base_channels=contribution_channels,
            coarse_feature_dim=coarse_feature_dim,
            count_cmax=count_cmax,
            initial_contribution=initial_contribution,
        )
        self.event_refiner = SelectedEventRefiner(
            num_bins=num_bins,
            hidden_channels=refiner_channels,
            count_cmax=count_cmax,
            max_log_depth_delta=max_log_depth_delta,
            max_normal_delta=max_normal_delta,
        )

    def forward(
        self,
        event_voxel: torch.Tensor,
        bad_rgb: torch.Tensor,
        coarse_depth: torch.Tensor,
        coarse_normals: torch.Tensor,
        coarse_features: Optional[torch.Tensor] = None,
        *,
        contribution_override: Optional[torch.Tensor] = None,
        bypass_contribution_net: bool = False,
    ) -> Dict[str, torch.Tensor]:
        expected_shape = event_voxel.shape[:2] + event_voxel.shape[-2:]
        if bypass_contribution_net:
            if contribution_override is None:
                raise ValueError("bypass_contribution_net requires contribution_override")
            predicted_contribution = contribution_override.detach()
        else:
            predicted_contribution = self.contribution_net(
                event_voxel,
                bad_rgb,
                coarse_depth,
                coarse_normals,
                coarse_features,
            )
        contribution = predicted_contribution if contribution_override is None else contribution_override
        if contribution.shape != expected_shape:
            raise ValueError(
                f"Contribution shape {contribution.shape} does not match expected {expected_shape}"
            )
        contribution = contribution.to(device=event_voxel.device, dtype=event_voxel.dtype).clamp(0.0, 1.0)
        selected_event = contribution.unsqueeze(2) * event_voxel
        refined = self.event_refiner(selected_event, coarse_depth, coarse_normals)
        return {
            **refined,
            "contribution": contribution,
            "predicted_contribution": predicted_contribution,
            "selected_event": selected_event,
        }


def permute_active_contribution(
    contribution: torch.Tensor,
    event_voxel: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomize contribution locations while preserving active-event values exactly."""
    output = contribution.clone()
    active = event_voxel.abs().sum(dim=2) > 0
    for batch_index in range(contribution.shape[0]):
        for view_index in range(contribution.shape[1]):
            mask = active[batch_index, view_index]
            values = contribution[batch_index, view_index][mask]
            if values.numel() <= 1:
                continue
            order = torch.randperm(values.numel(), generator=generator, device=values.device)
            output[batch_index, view_index][mask] = values[order]
    return output


def drop_ranked_events(
    contribution: torch.Tensor,
    event_voxel: torch.Tensor,
    *,
    fraction: float = 0.2,
    highest: bool,
) -> torch.Tensor:
    """Zero the same number of highest- or lowest-scored active event pixels."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    output = contribution.clone()
    active = event_voxel.abs().sum(dim=2) > 0
    for batch_index in range(contribution.shape[0]):
        for view_index in range(contribution.shape[1]):
            mask = active[batch_index, view_index]
            indices = mask.flatten().nonzero(as_tuple=False).squeeze(1)
            count = int(round(fraction * indices.numel()))
            if count <= 0:
                continue
            values = contribution[batch_index, view_index].flatten()[indices]
            selected = torch.topk(values, k=min(count, values.numel()), largest=bool(highest)).indices
            flattened = output[batch_index, view_index].flatten()
            flattened[indices[selected]] = 0.0
    return output


def contribution_condition(
    name: str,
    learned: torch.Tensor,
    event_voxel: torch.Tensor,
    *,
    drop_fraction: float = 0.2,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build the cheap Stage-1 causal/counterfactual test conditions."""
    normalized = str(name).lower()
    if normalized in {"learned", "bridge", "full_method"}:
        return learned
    if normalized in {"coarse_rgb", "zero_event", "zero"}:
        return torch.zeros_like(learned)
    if normalized in {"full_event", "full", "c=1"}:
        return torch.ones_like(learned)
    if normalized in {"random", "random_same_mean"}:
        return permute_active_contribution(learned, event_voxel, generator=generator)
    if normalized in {"drop_high", "remove_high"}:
        return drop_ranked_events(learned, event_voxel, fraction=drop_fraction, highest=True)
    if normalized in {"drop_low", "remove_low"}:
        return drop_ranked_events(learned, event_voxel, fraction=drop_fraction, highest=False)
    raise ValueError(f"Unknown contribution condition: {name!r}")


def build_model_from_checkpoint(checkpoint: Dict[str, object]) -> MultiLdrEventContributionModel:
    schema = checkpoint.get("schema")
    if schema != MultiLdrEventContributionModel.checkpoint_schema:
        raise ValueError(
            f"Expected Stage-1 schema {MultiLdrEventContributionModel.checkpoint_schema!r}, got {schema!r}. "
            "Legacy ReliabilityUNet checkpoints are intentionally incompatible."
        )
    architecture = dict(checkpoint["architecture"])
    model = MultiLdrEventContributionModel(**architecture)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("Stage-1 checkpoint does not contain a model state dictionary.")
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "BridgeMasks",
    "ContributionLossOutput",
    "ContributionNet",
    "MultiLdrEventContributionModel",
    "SelectedEventRefiner",
    "build_bridge_masks",
    "build_model_from_checkpoint",
    "contribution_budget",
    "contribution_condition",
    "drop_ranked_events",
    "geometry_emphasis_weight",
    "image_gradient_magnitude",
    "normalize_event_voxel",
    "orient_exposure_pair",
    "permute_active_contribution",
    "saturation_ratio",
    "stage1_contribution_loss",
]
