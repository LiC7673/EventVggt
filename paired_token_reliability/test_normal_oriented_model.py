"""CPU architectural sanity checks required by toDo.md."""

import torch

from paired_token_reliability.normal_oriented_model import (
    BoundedNormalResidualDecoder, EventOnlyFeatureAdapter,
    ZeroPreservingEventPyramid,
)


def test_zero_event_is_exactly_zero_feature_and_update():
    encoder = ZeroPreservingEventPyramid(num_bins=2, hidden_channels=8, pyramid_channels=8)
    event = torch.zeros(1, 1, 4, 16, 16)
    features, gates = encoder(event, torch.ones(1, 1, 16, 16), [(8, 8)] * 4)
    for feature, gate in zip(features, gates):
        assert torch.count_nonzero(feature) == 0
        assert torch.count_nonzero(gate) == 0


def test_zero_contribution_and_rgb_shortcut_removal():
    adapter = EventOnlyFeatureAdapter(16, 8, 16)
    adapter.alpha_logit.data.fill_(1.0)
    event = torch.randn(2, 8, 8, 8)
    rgb_a, rgb_b = torch.randn(2, 16, 8, 8), torch.randn(2, 16, 8, 8)
    _, update_a, _ = adapter(rgb_a, event, torch.ones(2, 1, 8, 8))
    _, update_b, _ = adapter(rgb_b, event, torch.ones(2, 1, 8, 8))
    torch.testing.assert_close(update_a, update_b)
    _, update_zero, _ = adapter(rgb_a, event, torch.zeros(2, 1, 8, 8))
    assert torch.count_nonzero(update_zero) == 0


def test_normal_residual_is_bounded_and_zero_gate_restores_coarse():
    decoder = BoundedNormalResidualDecoder(8, 16, normal_update_scale=0.15)
    decoder.decoder[-1].bias.data.fill_(10.0)
    feature = torch.randn(1, 8, 8, 8)
    coarse = torch.zeros(1, 16, 16, 3); coarse[..., 2] = 1
    final, delta, _ = decoder(feature, torch.ones(1, 1, 8, 8), coarse, (16, 16))
    assert delta.abs().max() <= 0.150001
    restored, zero_delta, _ = decoder(feature, torch.zeros(1, 1, 8, 8), coarse, (16, 16))
    torch.testing.assert_close(restored, coarse)
    assert torch.count_nonzero(zero_delta) == 0
