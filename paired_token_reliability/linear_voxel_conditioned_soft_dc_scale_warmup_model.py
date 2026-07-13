"""Soft-DC model with a true scale-only optimization warmup.

During the first ``scale_warmup_steps`` training forwards, event geometry is
still computed (so DDP sees the branch) but is not injected into final depth.
The calibrated coarse depth is therefore the only depth prediction that can
explain metric scale during this interval.
"""
from __future__ import annotations

import os
import torch

from paired_token_reliability.linear_voxel_conditioned_soft_dc_model import (
    ConditionedSoftDCLinearVoxelModel,
)
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


class ScaleWarmupConditionedSoftDCLinearVoxelModel(
    ConditionedSoftDCLinearVoxelModel
):
    checkpoint_schema = "linear_time_voxel_conditioned_soft_dc_scale_warmup_v1"

    def __init__(self, *args, scale_warmup_steps=1000,
                 event_min_pixel_mass=0.10, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_warmup_steps = max(0, int(scale_warmup_steps))
        self.event_min_pixel_mass = max(0.0, float(event_min_pixel_mass))
        self._scale_warmup_forward_step = 0
        self._last_raw_event_support = None
        self._last_filtered_event_support = None

    def _decayed_signed(self, views, split_event):
        """Reject only weak resize tails before temporal decay/encoding.

        The threshold is applied to total nonnegative mass over all temporal
        and polarity channels.  Surviving voxels retain their original bin,
        polarity and linear interpolation weights; this is not a binary event
        conversion.
        """
        nonnegative = split_event.float().clamp_min(0.0)
        pixel_mass = nonnegative.sum(dim=2, keepdim=True)
        raw_support = pixel_mass > 0.0
        filtered_support = pixel_mass >= self.event_min_pixel_mass
        self._last_raw_event_support = raw_support.detach()
        self._last_filtered_event_support = filtered_support.detach()
        filtered = split_event * filtered_support.to(split_event)
        return super()._decayed_signed(views, filtered)

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        if self.training:
            self._scale_warmup_forward_step += 1
        warmup = self.training and (
            self._scale_warmup_forward_step <= self.scale_warmup_steps
        )

        for view, item in zip(views, output.ress):
            item["scale_warmup_active"] = item["depth"].new_tensor(float(warmup))
            item["scale_warmup_step"] = item["depth"].new_tensor(
                float(self._scale_warmup_forward_step)
            )
            if not warmup:
                continue

            # Preserve a zero-gradient connection to the event branch while
            # making calibrated coarse depth the actual prediction.
            coarse = item["depth_coarse"][..., 0]
            zero_ratio = item["depth_delta_ratio"] * 0.0
            final = coarse * (1.0 + zero_ratio)
            item["depth_delta_ratio"] = zero_ratio
            item["depth"] = final.unsqueeze(-1)
            item["normal"] = depth_to_normals(
                final.float(), view["camera_intrinsics"].to(final).float()
            )
            item["depth_pixel_update"] = coarse * zero_ratio
            item["depth_total_update"] = item["depth_pixel_update"]
            item["depth_update_detail_ratio"] = (
                item["depth_update_detail_ratio"] * 0.0
            )
            item["depth_update_centered_ratio"] = zero_ratio
            item["depth_update_final_absolute"] = item["depth_pixel_update"]
            item["adapter_update_loss"] = item["adapter_update_loss"] * 0.0
            item["depth_update_dc_excess_loss"] = (
                item["depth_update_dc_excess_loss"] * 0.0
            )

        if (
            self.training
            and self._scale_warmup_forward_step % 500 == 0
            and int(os.environ.get("RANK", "0")) == 0
            and self._last_raw_event_support is not None
        ):
            raw = self._last_raw_event_support.float().mean()
            kept = self._last_filtered_event_support.float().mean()
            ratio = kept / raw.clamp_min(1e-12)
            print(
                f"[event-mass-filter@{self._scale_warmup_forward_step:05d}] "
                f"min_mass={self.event_min_pixel_mass:.4f} "
                f"raw_support={float(raw):.6f} "
                f"kept_support={float(kept):.6f} kept/raw={float(ratio):.4f}",
                flush=True,
            )

        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["ScaleWarmupConditionedSoftDCLinearVoxelModel"]
