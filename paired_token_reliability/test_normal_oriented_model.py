"""CPU architectural sanity checks required by toDo.md."""

import torch

from paired_token_reliability.normal_oriented_model import (
    EventOnlyNormalDecoder, EventOnlyFeatureAdapter,
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


def test_event_normal_is_standalone_and_unit_length():
    decoder = EventOnlyNormalDecoder(8, 16)
    feature = torch.randn(1, 8, 8, 8)
    normal, _ = decoder(feature, torch.ones(1, 1, 8, 8), (16, 16))
    torch.testing.assert_close(torch.linalg.vector_norm(normal, dim=-1), torch.ones(1, 16, 16))
