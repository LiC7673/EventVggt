"""Small CPU tests for the unified contribution losses and ablation controls."""

import torch

from paired_token_reliability.unified_loss import event_mass_budget, pair_consistency
from paired_token_reliability.unified_model import contribution_override


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
