"""Event-source attribution and saturation-localized geometry residual recovery."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_dual_alignment_hdr_model import (
    DualAlignmentHDRLinearVoxelModel,
)


class IdentityEventAlignment(nn.Module):
    """No full->geo token imitation; geo events supervise C only."""

    def __init__(self):
        super().__init__()
        # Compatibility handle for code that audits the removed old head.
        self.reliability = nn.Identity()

    def forward(self, full_feature):
        correction = torch.zeros_like(full_feature)
        reliability = full_feature[:, :1] * 0.0
        return full_feature, correction, reliability


class ForcedFullContribution(nn.Module):
    """Attribution ablation: deploy every full event while retaining DDP use."""

    def __init__(self, learned):
        super().__init__()
        self.learned = learned
        self.coarse_feature_dim = getattr(learned, "coarse_feature_dim", 0)

    def forward(self, *args, **kwargs):
        predicted = self.learned(*args, **kwargs)
        return torch.ones_like(predicted) + predicted * 0.0


class MissingGeometryResidualAdapter(nn.Module):
    """Predict missing geometry in a compact projection, then lift to RGB tokens."""

    def __init__(self, token_dim, geometry_dim=256):
        super().__init__()
        token_dim, geometry_dim = int(token_dim), int(geometry_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.event_norm = nn.LayerNorm(token_dim)
        self.geometry_projector = nn.Linear(token_dim, geometry_dim, bias=False)
        self.event_projector = nn.Linear(token_dim, geometry_dim, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(2 * geometry_dim, geometry_dim), nn.GELU(),
            nn.Linear(geometry_dim, geometry_dim),
        )
        self.geometry_lift = nn.Linear(geometry_dim, token_dim, bias=False)
        nn.init.xavier_uniform_(self.geometry_projector.weight)
        nn.init.xavier_uniform_(self.event_projector.weight)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.xavier_uniform_(self.geometry_lift.weight)
        self.records = []

    def reset_records(self):
        self.records = []

    def forward(self, ldr_token, selected_event_token):
        z_ldr = self.geometry_projector(self.token_norm(ldr_token))
        z_event = self.event_projector(self.event_norm(selected_event_token))
        delta_geometry = self.residual(torch.cat((z_ldr, z_event), dim=-1))
        token_delta = self.geometry_lift(delta_geometry)
        self.records.append((z_ldr, delta_geometry, token_delta))
        return ldr_token + token_delta, token_delta


class AttributionResidualLinearVoxelModel(DualAlignmentHDRLinearVoxelModel):
    checkpoint_schema = "linear_voxel_attribution_missing_geometry_residual_v1"

    def __init__(self, *args, geometry_projection_dim=256,
                 saturation_threshold=.98, ablate_event_attribution=False,
                 ablate_missing_residual=False, **kwargs):
        super().__init__(*args, **kwargs)
        token_dim = int(kwargs.get("embed_dim"))
        self.full_geo_aligner = IdentityEventAlignment()
        self.geometry_residual_adapter = MissingGeometryResidualAdapter(
            token_dim, geometry_projection_dim
        )
        # The parent calls this name from _fuse_tokens.
        self.ldr_event_hdr_aligner = self.geometry_residual_adapter
        self.saturation_threshold = float(saturation_threshold)
        self.ablate_event_attribution = bool(ablate_event_attribution)
        self.ablate_missing_residual = bool(ablate_missing_residual)
        if self.ablate_event_attribution:
            self.contribution_net = ForcedFullContribution(self.contribution_net)

    @staticmethod
    def _patch_grid(patch_count, image_h, image_w, patch_size):
        grid_h = max(1, image_h // int(patch_size))
        grid_w = max(1, image_w // int(patch_size))
        if grid_h * grid_w != patch_count:
            grid_h = max(1, round(math.sqrt(patch_count)))
            while grid_h > 1 and patch_count % grid_h:
                grid_h -= 1
            grid_w = patch_count // grid_h
        return grid_h, grid_w

    @staticmethod
    def _stack_optional(views, key, like):
        fields = [view.get(key) for view in views]
        if not all(torch.is_tensor(value) for value in fields):
            return None
        return torch.stack(fields, dim=1).to(like)

    def forward(self, views, *args, **kwargs):
        self.geometry_residual_adapter.reset_records()
        output = super().forward(views, *args, **kwargs)
        if not output.ress:
            return output

        images, _, _ = self._stack_inputs(views)
        b, v, _, image_h, image_w = images.shape
        contribution = torch.stack(
            [item["event_contribution"] for item in output.ress], dim=1
        ).float()
        full_repr = torch.stack(
            [item["signed_event"] for item in output.ress], dim=1
        ).float()

        # E_geo supplies only source-attribution supervision.  It is never
        # encoded as a token target in this route.
        c_target = self._stack_optional(views, "contribution_target", contribution)
        c_target_available = c_target is not None
        if c_target is None:
            c_target = torch.zeros_like(contribution)
        c_target = c_target.float().clamp(0, 1)

        selected_feature = self.event_encoder(full_repr * contribution.unsqueeze(2))
        rejected_feature = self.event_encoder(full_repr * (1.0 - contribution).unsqueeze(2))
        selected_normal = self._decode_event_normal(selected_feature)
        rejected_normal = self._decode_event_normal(rejected_feature)

        if not self.geometry_residual_adapter.records:
            raise RuntimeError("missing-geometry adapter was not called")
        z_ldr, predicted_delta, _ = self.geometry_residual_adapter.records[-1]
        patch_count = predicted_delta.shape[2]
        grid_h, grid_w = self._patch_grid(
            patch_count, image_h, image_w, self.patch_size
        )
        saturation = images.amax(dim=2).ge(self.saturation_threshold).float()
        saturation_patch = F.adaptive_avg_pool2d(
            saturation.reshape(b * v, 1, image_h, image_w), (grid_h, grid_w)
        ).flatten(2).transpose(1, 2).reshape(b, v, patch_count)

        hdr_fields = [view.get("hdr_img") for view in views]
        have_reference = all(torch.is_tensor(value) for value in hdr_fields)
        if have_reference:
            reference = torch.stack(hdr_fields, dim=1).to(images)
            with torch.no_grad():
                reference_tokens, reference_start = self.aggregator(reference)
            reference_patch = reference_tokens[-1][:, :, reference_start:].detach()
            # Stop-gradient includes both sides of Delta Z_gt as requested.
            target_delta = (
                self.geometry_residual_adapter.geometry_projector(
                    self.geometry_residual_adapter.token_norm(reference_patch)
                ) - z_ldr
            ).detach()
        else:
            target_delta = torch.zeros_like(predicted_delta)

        for index, item in enumerate(output.ress):
            item["event_contribution_target"] = c_target[:, index]
            item["event_contribution_target_available"] = item["depth"].new_tensor(
                float(c_target_available)
            )
            item["event_normal_selected"] = selected_normal[:, index]
            item["event_normal_rejected"] = rejected_normal[:, index]
            item["event_normal"] = selected_normal[:, index]
            item["predicted_missing_geometry_residual"] = predicted_delta[:, index]
            item["target_missing_geometry_residual"] = target_delta[:, index]
            item["saturation_patch_mask"] = saturation_patch[:, index]
            item["missing_residual_target_available"] = item["depth"].new_tensor(
                float(have_reference)
            )
            item["ablate_event_attribution"] = item["depth"].new_tensor(
                float(self.ablate_event_attribution)
            )
            item["ablate_missing_residual"] = item["depth"].new_tensor(
                float(self.ablate_missing_residual)
            )
        return output


__all__ = [
    "AttributionResidualLinearVoxelModel",
    "MissingGeometryResidualAdapter",
]
