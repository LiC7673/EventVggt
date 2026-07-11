"""Small CPU tests for the unified contribution losses and ablation controls."""

import torch

from paired_token_reliability.unified_loss import (
    event_mass_budget,
    geometry_contribution_rank_loss,
    pair_consistency,
)
from paired_token_reliability.unified_model import contribution_override
from paired_token_reliability.contribution_stage1 import build_bridge_masks


def test_temporal_budget_uses_event_mass():
    event = torch.zeros(1, 1, 4, 2, 2)
    event[:, :, 0] = 2.0
    contribution = torch.zeros_like(event)
    contribution[:, :, 0] = 0.5
    loss, ratio = event_mass_budget(contribution, event, rho=0.5)
    torch.testing.assert_close(ratio, torch.tensor([0.5]))
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_pair_consistency_ignores_empty_event_entries():
    event = torch.zeros(1, 1, 2, 2, 2)
    event[..., 0, 0] = 1.0
    left = torch.zeros_like(event)
    right = torch.ones_like(event)
    torch.testing.assert_close(pair_consistency(left, right, event), torch.tensor(1.0))


def test_contribution_ablation_shapes_and_values():
    event = torch.randn(2, 3, 6, 5, 7)
    assert contribution_override(event, "learned") is None
    torch.testing.assert_close(contribution_override(event, "full"), torch.ones_like(event))
    torch.testing.assert_close(contribution_override(event, "no_contribution"), torch.ones_like(event))
    torch.testing.assert_close(contribution_override(event, "none"), torch.zeros_like(event))
    random_mask = contribution_override(event, "random", 0.5, 0.5)
    assert random_mask.shape == event.shape
    assert torch.all((random_mask == 0) | (random_mask == 1))


def test_geometry_rank_prefers_matching_order():
    geometry = torch.tensor([[[[0.0, 1.0]]]])
    valid = torch.ones_like(geometry, dtype=torch.bool)
    event = torch.ones(1, 1, 2, 1, 2)
    matching = torch.tensor([[[[0.1, 0.9]]]])
    reversed_order = torch.tensor([[[[0.9, 0.1]]]])
    good = geometry_contribution_rank_loss(
        matching, geometry, valid, event, margin=0.1, difference_threshold=0.1
    )
    bad = geometry_contribution_rank_loss(
        reversed_order, geometry, valid, event, margin=0.1, difference_threshold=0.1
    )
    assert good < bad


def test_unified_bridge_does_not_treat_colored_reference_as_white_saturation():
    reference = torch.zeros(1, 1, 3, 2, 2)
    reference[:, :, 0] = 1.0
    reference[:, :, 1:] = 0.2
    bad = torch.ones_like(reference)
    event = torch.ones(1, 1, 2, 2, 2)
    relaxed = build_bridge_masks(
        reference,
        bad,
        event,
        require_reference_gradient=False,
        saturation_mode="all_channels",
    )
    strict_color = build_bridge_masks(
        reference,
        bad,
        event,
        require_reference_gradient=False,
        saturation_mode="any_channel",
    )
    assert relaxed.bridge.all()
    assert not strict_color.bridge.any()
