"""Detail-residual model exposing only real-event normal support."""
from paired_token_reliability.linear_voxel_detail_residual_model import (
    DetailResidualLinearVoxelModel,
)
from stage2_geometry_adapter.model import GeometryAdapterOutput


class DetailNormalDerivativeLinearVoxelModel(DetailResidualLinearVoxelModel):
    checkpoint_schema = "linear_time_voxel_detail_normal_derivative_v1"

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        for item in output.ress:
            # signed_event is the time-decayed polarity voxel.  A nonzero mass
            # is the only evidence allowed to activate event-normal learning.
            real_support = item["signed_event"].abs().sum(dim=1) > 0
            item["event_normal_support"] = real_support
        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["DetailNormalDerivativeLinearVoxelModel"]
