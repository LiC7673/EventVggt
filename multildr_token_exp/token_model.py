"""Temporal-detail VGGT with a lightweight exposure-invariant token adapter."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from eventvggt.models.streamvggt_temporal_detail import StreamVGGT as TemporalDetailVGGT


class ResidualTokenAdapter(nn.Module):
    """A bounded, zero-initialized adapter that cannot erase base tokens."""

    def __init__(self, dim: int, hidden_dim: int = 256, max_scale: float = 0.10) -> None:
        super().__init__()
        self.max_scale = float(max_scale)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        correction = self.up(self.act(self.down(self.norm(tokens))))
        return tokens + self.max_scale * torch.tanh(correction)


class StreamVGGT(TemporalDetailVGGT):
    """Expose aligned patch tokens while leaving every decoder unchanged."""

    def __init__(
        self,
        *args,
        token_adapter_hidden_dim: int = 256,
        token_adapter_max_scale: float = 0.10,
        token_adapter_layers: Sequence[int] = (4, 11, 17, 23),
        **kwargs,
    ) -> None:
        embed_dim = int(kwargs.get("embed_dim", 1024))
        super().__init__(*args, **kwargs)
        self.exposure_token_adapter = ResidualTokenAdapter(
            dim=2 * embed_dim,
            hidden_dim=int(token_adapter_hidden_dim),
            max_scale=float(token_adapter_max_scale),
        )
        self.token_adapter_layers = tuple(int(index) for index in token_adapter_layers)
        self._last_exposure_tokens = None
        self._last_raw_exposure_tokens = None
        self._token_hook = self.aggregator.register_forward_hook(self._adapt_aggregator_output)

    def _adapt_aggregator_output(self, _module, _inputs, output):
        tokens_list, patch_start_idx, *tail = output
        adapted = list(tokens_list)
        valid_layers = []
        raw_by_layer = {}
        for layer_index in self.token_adapter_layers:
            index = layer_index if layer_index >= 0 else len(adapted) + layer_index
            if index < 0 or index >= len(adapted):
                continue
            tokens = adapted[index]
            special = tokens[:, :, :patch_start_idx]
            patch = tokens[:, :, patch_start_idx:]
            raw_by_layer[index] = patch
            patch = self.exposure_token_adapter(patch)
            adapted[index] = torch.cat([special, patch], dim=2)
            valid_layers.append(index)
        if not valid_layers:
            raise RuntimeError(
                f"No token-adapter layer is valid for {len(adapted)} aggregator outputs: "
                f"{self.token_adapter_layers}"
            )
        self._last_exposure_tokens = adapted[valid_layers[-1]][:, :, patch_start_idx:]
        # Fall back to the last valid layer when negative/out-of-range layer
        # specifications make the direct capture above unavailable.
        self._last_raw_exposure_tokens = raw_by_layer[valid_layers[-1]]
        return (adapted, patch_start_idx, *tail)

    def forward(self, views, query_points=None, **kwargs):
        output = super().forward(views, query_points=query_points, **kwargs)
        if self._last_exposure_tokens is None:
            raise RuntimeError("Aggregator token hook did not capture patch tokens.")
        for frame_index, result in enumerate(output.ress):
            result["exposure_token"] = self._last_exposure_tokens[:, frame_index]
        return output


__all__ = ["StreamVGGT"]
