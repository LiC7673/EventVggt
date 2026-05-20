from eventvggt.models.antigrid_refiner import AntiGridDepthPointRefiner, refine_stream_output
from streamvggt.models.streamvggt import StreamVGGT as BaseStreamVGGT
from streamvggt.models.streamvggt import StreamVGGTOutput


class StreamVGGT(BaseStreamVGGT):
    """Pure RGB StreamVGGT with a zero-initialized dense anti-grid refiner."""

    def __init__(
        self,
        *args,
        refiner_hidden_dim: int = 48,
        refiner_num_blocks: int = 4,
        refiner_residual_scale: float = 0.05,
        refiner_refine_points: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.antigrid_refiner = AntiGridDepthPointRefiner(
            image_channels=3,
            hidden_dim=refiner_hidden_dim,
            num_blocks=refiner_num_blocks,
            residual_scale=refiner_residual_scale,
            refine_points=refiner_refine_points,
        )

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        return refine_stream_output(output, views, self.antigrid_refiner)

    def inference(self, frames, *args, **kwargs):
        output = super().inference(frames, *args, **kwargs)
        return refine_stream_output(output, getattr(output, "views", frames), self.antigrid_refiner)


__all__ = ["StreamVGGT", "StreamVGGTOutput"]
