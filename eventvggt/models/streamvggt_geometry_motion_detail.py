"""Temporal-detail StreamVGGT used with geometry-motion events only.

This is intentionally a separate model entry so the oracle geometry-motion
ablation cannot silently change the existing temporal-detail implementation.
The event source replacement is handled by its dedicated dataloader.
"""

from eventvggt.models.streamvggt_stable_temporal_detail import (
    StreamVGGT as _TemporalDetailStreamVGGT,
    StableTemporalVoxelDetailRefiner,
)
from eventvggt.models.streamvggt_temporal_detail import StreamVGGTOutput


class StreamVGGT(_TemporalDetailStreamVGGT):
    pass


__all__ = ["StreamVGGT", "StreamVGGTOutput", "StableTemporalVoxelDetailRefiner"]
