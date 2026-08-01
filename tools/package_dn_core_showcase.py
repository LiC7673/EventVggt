"""Build a privacy-safe, non-runnable showcase of the DN pipeline.

The generated archive contains only compact algorithmic excerpts.  It never
reads training configs, checkpoints, logs, dataset paths, scene names, user
names, host names, GPU IDs, or experiment directories from the repository.
"""
from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path


CORE_SOURCE = r'''"""Core ideas of an event-guided differential-normal depth refiner.

This educational excerpt intentionally omits model loading, datasets, training
schedules, logging, visualization, and project-specific infrastructure.
Shapes used below are illustrative rather than an executable API contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def signed_temporal_voxel(split_voxel, bin_count=5, count_ceiling=3.0,
                          bin_age=None, decay_time=0.0015):
    """Preserve time, polarity, and event mass in 2B channels.

    Channel layout is [positive bins, negative bins].  Negative-event mass is
    explicitly negative; it is not thresholded to a binary mask and is not
    collapsed with positive mass before encoding.
    """
    if split_voxel.shape[-3] != 2 * bin_count:
        raise ValueError("expected separate positive/negative temporal bins")
    scale = torch.log1p(torch.as_tensor(count_ceiling, device=split_voxel.device))
    mass = torch.log1p(split_voxel.abs().clamp_max(count_ceiling)) / scale
    sign = mass.new_ones(2 * bin_count)
    sign[bin_count:] = -1
    representation = mass * sign.view(*([1] * (mass.ndim - 3)), -1, 1, 1)
    if bin_age is not None:
        temporal_weight = torch.exp(-bin_age / decay_time).clamp(0, 1)
        temporal_weight = torch.cat((temporal_weight, temporal_weight), dim=-1)
        representation = representation * temporal_weight[..., None, None]
    return representation


class EventNormalDerivativeHead(nn.Module):
    """Predict pixelwise [dN/dx, dN/dy], not an absolute event normal."""

    def __init__(self, channels, derivative_limit=0.25):
        super().__init__()
        self.limit = float(derivative_limit)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 6, 1),
        )

    def forward(self, event_feature):
        raw = self.decoder(event_feature)
        bounded = self.limit * torch.tanh(raw / self.limit)
        return bounded.reshape(raw.shape[0], 2, 3, *raw.shape[-2:])


def forward_normal_difference(normal):
    """First-order image-plane normal differences [dN/dx, dN/dy]."""
    dx = torch.zeros_like(normal)
    dy = torch.zeros_like(normal)
    dx[..., :, :-1, :] = normal[..., :, 1:, :] - normal[..., :, :-1, :]
    dy[..., :-1, :, :] = normal[..., 1:, :, :] - normal[..., :-1, :, :]
    return torch.cat((dx, dy), dim=-1)


def detail_balanced_loss(error, target_magnitude, valid, quantile=0.70,
                         weak_weight=0.10, floor=1.0e-4):
    """Prevent numerous flat pixels from overwhelming sparse geometry detail."""
    values = target_magnitude[valid]
    threshold = (torch.quantile(values.detach(), quantile) if values.numel()
                 else target_magnitude.new_tensor(float("inf")))
    strong = valid & (target_magnitude >= threshold) & (target_magnitude > floor)
    weak = valid & ~strong
    strong_term = (error * strong).sum() / strong.sum().clamp_min(1)
    weak_term = (error * weak).sum() / weak.sum().clamp_min(1)
    return strong_term + weak_weight * weak_term


def normalized_local_mean(value, valid, kernel_size=9):
    """Masked low-pass operator that avoids leaking invalid background zeros."""
    padding = kernel_size // 2
    weight = F.avg_pool2d(valid.float(), kernel_size, 1, padding)
    total = F.avg_pool2d(value * valid.float(), kernel_size, 1, padding)
    return total / weight.clamp_min(1.0e-6)


def high_frequency_log_depth_target(base_depth, gt_depth, valid, kernel_size=9):
    """Target the detail missing from a stable coarse/HDR-like base."""
    residual = torch.log(gt_depth.clamp_min(1.0e-6)) - torch.log(
        base_depth.clamp_min(1.0e-6)
    )
    low_frequency = normalized_local_mean(residual, valid, kernel_size)
    return (residual - low_frequency).detach()


class EventConditionedHDRAdapter(nn.Module):
    """Add event-conditioned missing geometry to an LDR geometry token."""

    def __init__(self, token_dim, bottleneck):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(token_dim)
        self.event_norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.rgb_context = nn.Linear(token_dim, bottleneck)
        self.event_modulation = nn.Linear(token_dim, bottleneck, bias=False)
        self.output = nn.Linear(bottleneck, token_dim)

    def forward(self, ldr_token, selected_event_token):
        rgb = F.gelu(self.rgb_context(self.rgb_norm(ldr_token)))
        event = torch.tanh(self.event_modulation(
            self.event_norm(selected_event_token)
        ))
        token_residual = self.output(rgb * event)
        return ldr_token + token_residual, token_residual


class PixelGeometryRefiner(nn.Module):
    """Predict a bounded dense log-depth residual from event and coarse geometry."""

    def __init__(self, event_channels, hidden=64, update_limit=0.30):
        super().__init__()
        # event feature + current/base log depth + current/target normals + C
        self.network = nn.Sequential(
            nn.Conv2d(event_channels + 9, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.limit = float(update_limit)

    def forward(self, event_feature, base_depth, current_normal,
                event_normal_target, confidence):
        log_base = torch.log(base_depth.clamp_min(1.0e-6))
        refine_input = torch.cat((event_feature, log_base, log_base,
                                  current_normal, event_normal_target,
                                  confidence), dim=1)
        neutral_input = torch.cat((event_feature, log_base, log_base,
                                   current_normal, current_normal,
                                   confidence), dim=1)
        # Removing the neutral response prevents a constant bias from creating
        # geometry when the event cue proposes no normal change.
        raw = self.network(refine_input) - self.network(neutral_input)
        log_update = self.limit * torch.tanh(raw / self.limit)
        final_depth = torch.exp(log_base + log_update)
        return final_depth, log_update


def differential_geometry_losses(event_derivative, final_depth, gt_depth,
                                 intrinsics, valid, event_support,
                                 depth_to_normals):
    """Supervise event dN and propagate it to final depth geometry."""
    gt_normal = F.normalize(depth_to_normals(gt_depth, intrinsics), dim=-1)
    final_normal = F.normalize(depth_to_normals(final_depth, intrinsics), dim=-1)
    target = forward_normal_difference(gt_normal).detach()
    prediction = event_derivative.movedim(-3, -1).flatten(-2)
    final_derivative = forward_normal_difference(final_normal)
    magnitude = target.norm(dim=-1)
    live = valid & event_support

    event_error = F.smooth_l1_loss(
        prediction, target, beta=0.01, reduction="none"
    ).mean(-1)
    depth_error = F.smooth_l1_loss(
        final_derivative, target, beta=0.01, reduction="none"
    ).mean(-1)
    event_dn_loss = detail_balanced_loss(event_error, magnitude, live)
    depth_dn_loss = detail_balanced_loss(depth_error, magnitude, live)
    return event_dn_loss, depth_dn_loss


def project_patch_centers(depth_patch, source_intrinsics, target_intrinsics,
                          source_to_world, world_to_target, patch_size):
    """Map source patch centers into an adjacent target view using coarse pose."""
    height, width = depth_patch.shape[-2:]
    y = (torch.arange(height, device=depth_patch.device) + 0.5) * patch_size - 0.5
    x = (torch.arange(width, device=depth_patch.device) + 0.5) * patch_size - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    z = depth_patch
    xyz1 = torch.stack((
        (xx - source_intrinsics[0, 2]) * z / source_intrinsics[0, 0],
        (yy - source_intrinsics[1, 2]) * z / source_intrinsics[1, 1],
        z, torch.ones_like(z),
    ), dim=-1)
    world = torch.einsum("ij,hwj->hwi", source_to_world, xyz1)
    target = torch.einsum("ij,hwj->hwi", world_to_target, world)[..., :3]
    u = target_intrinsics[0, 0] * target[..., 0] / target[..., 2] + target_intrinsics[0, 2]
    v = target_intrinsics[1, 1] * target[..., 1] / target[..., 2] + target_intrinsics[1, 2]
    return u, v, target[..., 2]


def cross_view_derivative_consistency(source_magnitude, warped_target_magnitude,
                                      overlap, source_confidence,
                                      target_confidence):
    """Pose-warped adjacent-view consistency on rotation-invariant |dN|."""
    active = overlap & (source_magnitude > 1.0e-5) & (
        warped_target_magnitude > 1.0e-5
    )
    if active.any():
        calibration = torch.median(
            warped_target_magnitude[active].detach()
            / source_magnitude[active].detach().clamp_min(1.0e-6)
        ).clamp(0.25, 4.0)
    else:
        calibration = source_magnitude.new_tensor(1.0)
    error = F.smooth_l1_loss(
        source_magnitude * calibration,
        warped_target_magnitude,
        beta=0.01,
        reduction="none",
    )
    weight = overlap.float() * torch.sqrt(
        (source_confidence * target_confidence).clamp_min(0)
    )
    return (error * weight).sum() / weight.sum().clamp_min(1.0e-6)


def compose_core_objective(base_geometry_loss, hdr_alignment_loss,
                           event_dn_loss, depth_dn_loss, hf_refiner_loss,
                           cross_view_dn_loss, weights):
    """Expose the conceptual objective without project-specific bookkeeping."""
    return (
        base_geometry_loss
        + weights.hdr * hdr_alignment_loss
        + weights.event_dn * event_dn_loss
        + weights.depth_dn * depth_dn_loss
        + weights.high_frequency * hf_refiner_loss
        + weights.cross_view * cross_view_dn_loss
    )
'''


DATA_SOURCE = r'''"""Generic multi-view RGB/event geometry data pipeline.

Expected public-facing layout (adapt the field names to a released dataset):

root/sequence_id/
  rgb/000001.png
  depth/000001.npy
  events.h5                 # x, y, t, polarity
  calibration.json          # intrinsics and camera poses

No project-specific directory or scene identifier is embedded here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrameRecord:
    sequence: Path
    frame_id: int
    image_path: Path
    depth_path: Path
    timestamp: float


def linear_polarity_voxel(x, y, timestamp, polarity, height, width,
                          bin_count, start_time, end_time):
    """Create V(x,y,b,p) using linear interpolation between temporal bins."""
    voxel = torch.zeros(2 * bin_count, height, width, dtype=torch.float32)
    if len(timestamp) == 0 or end_time <= start_time:
        return voxel
    x = torch.as_tensor(x, dtype=torch.long)
    y = torch.as_tensor(y, dtype=torch.long)
    t = torch.as_tensor(timestamp, dtype=torch.float32)
    p = torch.as_tensor(polarity)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]
    normalized = (t - start_time) / max(end_time - start_time, 1.0e-9)
    coordinate = normalized.clamp(0, 1) * (bin_count - 1)
    left = coordinate.floor().long()
    right = (left + 1).clamp_max(bin_count - 1)
    right_weight = coordinate - left.float()
    left_weight = 1.0 - right_weight
    polarity_offset = torch.where(p > 0, 0, bin_count)
    flat = voxel.view(2 * bin_count, -1)
    pixel = y * width + x
    flat.index_put_((left + polarity_offset, pixel), left_weight, accumulate=True)
    flat.index_put_((right + polarity_offset, pixel), right_weight, accumulate=True)
    return voxel


class EventWindowReader:
    """Read only the event interval associated with two adjacent RGB frames."""

    def __init__(self, event_file):
        self.event_file = Path(event_file)

    def read(self, start_time, end_time):
        with h5py.File(self.event_file, "r") as handle:
            timestamps = np.asarray(handle["t"])
            begin = int(np.searchsorted(timestamps, start_time, side="right"))
            finish = int(np.searchsorted(timestamps, end_time, side="right"))
            return {
                "x": np.asarray(handle["x"][begin:finish]),
                "y": np.asarray(handle["y"][begin:finish]),
                "t": timestamps[begin:finish],
                "p": np.asarray(handle["polarity"][begin:finish]),
            }


class MultiViewEventGeometryDataset(Dataset):
    """Return ordered adjacent views with synchronized RGB, event and geometry."""

    def __init__(self, root, num_views=4, bin_count=5, image_size=None):
        self.root = Path(root)
        self.num_views = int(num_views)
        self.bin_count = int(bin_count)
        self.image_size = image_size
        self.sequences = self._discover_sequences()
        self.windows = self._build_adjacent_windows()

    def _discover_sequences(self):
        sequences = []
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            calibration = directory / "calibration.json"
            event_file = directory / "events.h5"
            if calibration.is_file() and event_file.is_file():
                sequences.append(directory)
        if not sequences:
            raise RuntimeError("no sequences match the documented public layout")
        return sequences

    def _records(self, sequence):
        metadata = json.loads((sequence / "calibration.json").read_text())
        records = []
        for item in metadata["frames"]:
            frame_id = int(item["id"])
            image = sequence / "rgb" / f"{frame_id:06d}.png"
            depth = sequence / "depth" / f"{frame_id:06d}.npy"
            if image.is_file() and depth.is_file():
                records.append(FrameRecord(sequence, frame_id, image, depth,
                                           float(item["timestamp"])))
        return records

    def _build_adjacent_windows(self):
        windows = []
        for sequence in self.sequences:
            records = self._records(sequence)
            for start in range(1, len(records) - self.num_views + 1):
                selected = records[start:start + self.num_views]
                if all(b.frame_id == a.frame_id + 1
                       for a, b in zip(selected[:-1], selected[1:])):
                    windows.append((records[start - 1], selected))
        return windows

    @staticmethod
    def _camera_metadata(sequence):
        return json.loads((sequence / "calibration.json").read_text())

    def _load_view(self, previous, current, metadata):
        rgb = np.asarray(Image.open(current.image_path).convert("RGB"))
        rgb = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(np.load(current.depth_path)).float()
        height, width = depth.shape
        events = EventWindowReader(current.sequence / "events.h5").read(
            previous.timestamp, current.timestamp
        )
        voxel = linear_polarity_voxel(
            events["x"], events["y"], events["t"], events["p"],
            height, width, self.bin_count, previous.timestamp, current.timestamp,
        )
        camera = metadata["camera_by_frame"][str(current.frame_id)]
        return {
            "image": rgb,
            "depth": depth,
            "valid_depth": torch.isfinite(depth) & (depth > 0),
            "event_voxel": voxel,
            "event_time_range": torch.tensor(
                [previous.timestamp, current.timestamp], dtype=torch.float64
            ),
            "intrinsics": torch.tensor(camera["intrinsics"], dtype=torch.float32),
            "camera_to_world": torch.tensor(camera["camera_to_world"],
                                            dtype=torch.float32),
            "frame_id": current.frame_id,
        }

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        previous, records = self.windows[index]
        metadata = self._camera_metadata(records[0].sequence)
        views = []
        for record in records:
            views.append(self._load_view(previous, record, metadata))
            previous = record
        return {"views": views}
'''


MODULE_SOURCE = r'''"""Compact neural modules for the DN pipeline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dn_core_showcase import (
    EventConditionedHDRAdapter,
    EventNormalDerivativeHead,
    PixelGeometryRefiner,
    signed_temporal_voxel,
)


class EventPyramidEncoder(nn.Module):
    """Dense pixel encoder; it avoids patch-only decoding of sparse details."""

    def __init__(self, input_channels=10, hidden=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )

    def forward(self, voxel):
        return self.network(voxel)


class GeometryRelevanceField(nn.Module):
    """Inference-time event relevance from full events, RGB and coarse geometry."""

    def __init__(self, event_channels, hidden=32):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(event_channels + 5, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1), nn.Sigmoid(),
        )

    def forward(self, event_feature, rgb, coarse_depth, coarse_normal):
        luminance = rgb.mean(1, keepdim=True)
        inputs = torch.cat((event_feature, luminance, coarse_depth,
                            coarse_normal), dim=1)
        return self.predictor(inputs)


class PatchEventProjection(nn.Module):
    def __init__(self, event_channels, token_channels):
        super().__init__()
        self.projection = nn.Conv2d(event_channels, token_channels, 1, bias=False)

    def forward(self, feature, relevance, grid_size):
        selected = feature * relevance
        pooled = F.adaptive_avg_pool2d(selected, grid_size)
        return self.projection(pooled).flatten(2).transpose(1, 2)


class DerivativeToNormalCue(nn.Module):
    """Convert local dN evidence into a bounded correction around coarse N."""

    def __init__(self, hidden=32, correction_limit=0.10):
        super().__init__()
        self.limit = float(correction_limit)
        self.integrator = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )

    def forward(self, derivative, base_normal, relevance):
        flat = derivative.flatten(1, 2)
        delta = self.limit * torch.tanh(self.integrator(flat) / self.limit)
        return F.normalize(base_normal + relevance * delta, dim=1, eps=1.0e-6)


class EventGuidedDNModel(nn.Module):
    """Minimal composition around an arbitrary frozen RGB geometry backbone."""

    def __init__(self, rgb_geometry_backbone, depth_head, normal_from_depth,
                 event_bins=5, event_channels=32, token_channels=256):
        super().__init__()
        self.rgb_backbone = rgb_geometry_backbone
        self.depth_head = depth_head
        self.normal_from_depth = normal_from_depth
        self.event_bins = event_bins
        self.event_encoder = EventPyramidEncoder(2 * event_bins, event_channels)
        self.relevance = GeometryRelevanceField(event_channels)
        self.event_dn_head = EventNormalDerivativeHead(event_channels)
        self.derivative_to_normal = DerivativeToNormalCue()
        self.event_projection = PatchEventProjection(event_channels, token_channels)
        self.hdr_adapter = EventConditionedHDRAdapter(token_channels, token_channels // 4)
        self.refiner = PixelGeometryRefiner(event_channels)

    def forward(self, image, event_voxel, intrinsics, bin_age=None):
        with torch.no_grad():
            ldr_tokens = self.rgb_backbone(image)
            coarse_depth = self.depth_head(ldr_tokens)
            coarse_normal = self.normal_from_depth(coarse_depth, intrinsics)
        representation = signed_temporal_voxel(
            event_voxel, self.event_bins, bin_age=bin_age
        )
        event_feature = self.event_encoder(representation)
        relevance = self.relevance(
            event_feature, image, coarse_depth.detach(), coarse_normal.detach()
        )
        event_derivative = self.event_dn_head(event_feature)
        grid_size = ldr_tokens.shape[-3:-1]
        event_token = self.event_projection(event_feature, relevance, grid_size)
        hdr_tokens, token_residual = self.hdr_adapter(ldr_tokens, event_token)
        hdr_depth = self.depth_head(hdr_tokens)
        hdr_normal = self.normal_from_depth(hdr_depth, intrinsics)

        local_normal_target = self.derivative_to_normal(
            event_derivative, hdr_normal, relevance
        )
        final_depth, log_update = self.refiner(
            event_feature * relevance, hdr_depth, hdr_normal,
            local_normal_target, relevance,
        )
        return {
            "coarse_depth": coarse_depth,
            "hdr_depth": hdr_depth,
            "final_depth": final_depth,
            "event_derivative": event_derivative,
            "event_relevance": relevance,
            "log_depth_update": log_update,
            "token_residual": token_residual,
        }
'''


TRAIN_SOURCE = r'''"""Illustrative loss assembly and one optimization step."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dn_core_showcase import (
    compose_core_objective,
    differential_geometry_losses,
    high_frequency_log_depth_target,
)


@dataclass
class LossWeights:
    hdr: float = 1.0
    event_dn: float = 1.0
    depth_dn: float = 1.0
    high_frequency: float = 2.0
    cross_view: float = 0.2


def stack_views(batch, key):
    return torch.stack([view[key] for view in batch["views"]], dim=1)


def compute_losses(model, batch, depth_to_normals,
                   cross_view_loss_function, weights):
    images = stack_views(batch, "image")
    events = stack_views(batch, "event_voxel")
    intrinsics = stack_views(batch, "intrinsics")
    gt_depth = stack_views(batch, "depth")
    valid = stack_views(batch, "valid_depth")
    outputs = []
    for view in range(images.shape[1]):
        outputs.append(model(images[:, view], events[:, view], intrinsics[:, view]))

    final_depth = torch.stack([item["final_depth"] for item in outputs], 1)
    hdr_depth = torch.stack([item["hdr_depth"] for item in outputs], 1)
    event_dn = torch.stack([item["event_derivative"] for item in outputs], 1)
    support = events.abs().sum(2) > 0

    base_geometry = F.smooth_l1_loss(final_depth[valid], gt_depth[valid])
    # During Multi-LDR training, hdr_teacher_tokens are stop-gradient tokens
    # extracted from a better-exposure observation.
    hdr_alignment = final_depth.new_zeros(())
    event_dn_loss, depth_dn_loss = differential_geometry_losses(
        event_dn, final_depth, gt_depth, intrinsics, valid, support,
        depth_to_normals,
    )
    hf_target = high_frequency_log_depth_target(
        hdr_depth.flatten(0, 1), gt_depth.flatten(0, 1),
        valid.flatten(0, 1),
    ).reshape_as(final_depth)
    predicted_hf = torch.stack([item["log_depth_update"] for item in outputs], 1)
    hf_loss = F.smooth_l1_loss(predicted_hf[valid], hf_target[valid])
    cross_view_loss = cross_view_loss_function(outputs, batch["views"])

    total = compose_core_objective(
        base_geometry, hdr_alignment, event_dn_loss, depth_dn_loss,
        hf_loss, cross_view_loss, weights,
    )
    return total, {
        "geometry": base_geometry.detach(),
        "event_dn": event_dn_loss.detach(),
        "depth_dn": depth_dn_loss.detach(),
        "high_frequency": hf_loss.detach(),
        "cross_view_dn": cross_view_loss.detach(),
    }


def train_one_step(model, batch, optimizer, depth_to_normals,
                   cross_view_loss_function, weights=LossWeights()):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, diagnostics = compute_losses(
        model, batch, depth_to_normals, cross_view_loss_function, weights
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_norm=1.0,
    )
    optimizer.step()
    return float(loss.detach()), diagnostics
'''


CONFIG_SOURCE = r'''# Anonymous illustrative configuration; no private paths.
data:
  root: <PUBLIC_DATASET_ROOT>
  num_views: 4
  temporal_bins: 5
  require_adjacent_frames: true

model:
  event_channels: 32
  preserve_polarity_channels: true
  temporal_decay: true
  predict_normal_derivative: true
  dense_pixel_refiner: true

loss:
  hdr_alignment: 1.0
  event_normal_derivative: 1.0
  final_depth_normal_derivative: 1.0
  high_frequency_depth_residual: 2.0
  cross_view_derivative_consistency: 0.2
'''


README = r'''# Event-guided DN refinement: privacy-safe core excerpt

This archive is a compact implementation-oriented illustration of the method.
It includes a generic data loader, event voxelization, neural modules, loss
assembly, and an optimization step, while omitting identifying metadata.

## Included ideas

1. **Signed temporal event representation**: retain temporal bins, separate
   polarities, accumulated event mass, and optional temporal decay.
2. **Event differential-normal prediction**: predict pixelwise
   `(dN/dx, dN/dy)` instead of an absolute normal map.
3. **Detail-balanced supervision**: separately normalize strong derivative
   pixels so flat regions cannot make the zero solution optimal.
4. **Event-conditioned HDR token residual**: use selected event features to
   complement an LDR geometry token through an additive residual.
5. **Pixel geometry refiner**: predict a bounded high-frequency log-depth
   residual around a stable coarse/HDR-like depth.
6. **Depth-normal coupling**: require the normal derivative induced by final
   depth to agree with the ground-truth differential geometry.
7. **Cross-view DN consistency**: use detached coarse depth, intrinsics, and
   pose to map adjacent-view patches; compare calibrated derivative magnitude
   only in reliable overlapping patches.

## Explicitly excluded

- dataset and scene names;
- local or server paths;
- user, host, or organization identifiers;
- trained-weight filenames and experiment directories;
- GPU assignments, ports, logs, and command-line launch settings;
- full training schedules and private engineering infrastructure.

The source is intended for method communication, review discussion, or a
supplementary-code sketch. It is not intended to reproduce trained results.

## Files

- `data_pipeline.py`: sequence discovery, adjacent-view sampling, synchronized
  RGB/depth/event loading, and linear polarity voxelization;
- `network_modules.py`: event encoder, relevance field, HDR adapter,
  differential-normal head, and dense depth refiner;
- `dn_core_showcase.py`: geometry operators and core losses;
- `training_step.py`: explicit loss assembly and optimization step;
- `example_config.yaml`: anonymous architecture/loss configuration.
'''


FORBIDDEN_PATTERNS = (
    r"[A-Za-z]:[\\/]",          # Windows absolute path
    r"/(?:home|data\d*|Users)/", # common personal/server roots
    r"checkpoint",               # experiment artifact naming
    r"CUDA_VISIBLE_DEVICES",
    r"MASTER_PORT",
    r"scene_names?",
)


def sanitize_esim_source(source: str) -> str:
    """Remove machine-specific values and select a stronger public noise preset."""
    # Replace the four private roots with generic, caller-provided inputs.
    replacements = {
        "OBJ_DIR": 'os.environ.get("EVENT_MODEL_ROOT", "")',
        "HDRI_DIR": 'os.environ.get("EVENT_ENVIRONMENT_ROOT", "")',
        "OUTPUT_DIR": 'os.environ.get("EVENT_OUTPUT_ROOT", "")',
        "LOG_FILE": 'os.environ.get("EVENT_LOG_FILE", "event_generation.log")',
    }
    for name, value in replacements.items():
        source = re.sub(
            rf"(?m)^{name}\s*=\s*[^\r\n]+$", f"{name} = {value}", source
        )

    # Defense in depth: scrub any remaining quoted absolute filesystem value.
    source = re.sub(
        r'''(?i)r?["'][a-z]:[\\/][^"']*["']''',
        '"<ABSOLUTE_PATH_REMOVED>"',
        source,
    )
    source = re.sub(
        r'''["']/(?:home|data\d*|users|mnt)/[^"']*["']''',
        '"<ABSOLUTE_PATH_REMOVED>"',
        source,
        flags=re.IGNORECASE,
    )

    # Stronger than the private working preset, intentionally exposed as a
    # centralized high-noise robustness configuration.
    noise_values = {
        "BACKGROUND_ACTIVITY_RATE": "0.10",
        "BRIGHT_NOISE_RATE": "0.15",
        "HOT_PIXEL_RATIO": "0.00025",
        "HOT_PIXEL_RATE": "4.0",
        "THRESHOLD_MISMATCH_STD": "0.35",
        "JITTER_ROT_DEG_SIN": "0.15",
        "JITTER_ROT_DEG_RANDOM": "0.03",
    }
    for name, value in noise_values.items():
        source = re.sub(
            rf"(?m)^{name}\s*=\s*[^\r\n]+$", f"{name} = {value}", source
        )

    header = '''"""Anonymized ESIM-style event generator (high-noise preset).

All filesystem inputs are supplied externally. No dataset, user, machine,
scene, or experiment path is embedded in this public showcase copy.
"""\n'''
    return header + source.lstrip("\ufeff\r\n")


def privacy_audit(text: str):
    hits = [pattern for pattern in FORBIDDEN_PATTERNS
            if re.search(pattern, text, flags=re.IGNORECASE)]
    if hits:
        raise RuntimeError(f"privacy audit rejected generated content: {hits}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="release_artifacts/dn_core_showcase",
        help="generated folder (must be outside the private training tree)",
    )
    parser.add_argument(
        "--archive", default="release_artifacts/dn_core_showcase.zip",
        help="output ZIP path",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--esim-source", default=None,
        help="optional private ESIM script to sanitize and include",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_dir)
    archive = Path(args.archive)
    generated = {
        "dn_core_showcase.py": CORE_SOURCE,
        "data_pipeline.py": DATA_SOURCE,
        "network_modules.py": MODULE_SOURCE,
        "training_step.py": TRAIN_SOURCE,
        "example_config.yaml": CONFIG_SOURCE,
        "README.md": README,
    }
    if args.esim_source:
        source_path = Path(args.esim_source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        esim = sanitize_esim_source(
            source_path.read_text(encoding="utf-8", errors="replace")
        )
        generated["esim_event_generator_anonymized_high_noise.py"] = esim
        generated["README.md"] = README.replace(
            "- `example_config.yaml`: anonymous architecture/loss configuration.\n",
            "- `example_config.yaml`: anonymous architecture/loss configuration;\n"
            "- `esim_event_generator_anonymized_high_noise.py`: anonymized event "
            "generation source. All machine-specific paths are replaced by "
            "environment-variable inputs, and the public example uses an "
            "intentionally stronger sensor-noise configuration.\n",
        )
    for content in generated.values():
        privacy_audit(content)

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for name, content in generated.items():
        (output / name).write_text(content, encoding="utf-8")

    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if not args.overwrite:
            raise FileExistsError(f"{archive} exists; pass --overwrite to replace it")
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.iterdir()):
            handle.write(path, arcname=f"dn_core_showcase/{path.name}")
    print(f"Privacy audit passed. Folder: {output}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
