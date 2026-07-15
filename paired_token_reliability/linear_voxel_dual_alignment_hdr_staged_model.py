"""V10 comparison model with a pretrained, frozen E_geo teacher."""
from __future__ import annotations

import copy
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_dual_alignment_hdr_model import (
    DualAlignmentHDRLinearVoxelModel,
)
from paired_token_reliability.signed_multiscale_model import signed_support


class StagedGeoTeacherDualAlignmentModel(DualAlignmentHDRLinearVoxelModel):
    checkpoint_schema = "linear_time_voxel_dual_alignment_hdr_staged_geo_teacher_no_point_refiner_v10"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geo_event_encoder = copy.deepcopy(self.event_encoder)
        self.geo_normal_decoder = copy.deepcopy(self.event_normal_decoder)
        self.register_buffer("full_initialized_from_geo", self.metric_depth_scale.new_zeros((), dtype=None))

    def _encode_geo_event(self, views, voxel):
        representation, decay = self._decayed_signed(views, voxel)
        feature = self.geo_event_encoder(representation)
        support = signed_support(
            representation, self.support_dilation_kernel
        )[:, :, 0] > 0
        return representation, decay, feature, support

    def _decode_geo_normal(self, feature):
        b, v, channels, height, width = feature.shape
        normal = self.geo_normal_decoder(
            feature.reshape(b * v, channels, height, width)
        )
        return F.normalize(normal.float(), dim=1, eps=1e-6).reshape(
            b, v, 3, height, width
        ).movedim(2, -1)

    def initialize_full_student_from_geo(self):
        if bool(self.full_initialized_from_geo.item()):
            return False
        self.event_encoder.load_state_dict(self.geo_event_encoder.state_dict())
        self.event_normal_decoder.load_state_dict(self.geo_normal_decoder.state_dict())
        self.full_initialized_from_geo.fill_(1)
        return True
