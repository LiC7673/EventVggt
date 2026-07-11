"""Small CPU tests for the unified contribution losses and ablation controls."""

import torch

from eventvggt.datasets.my_event_dataset import MyEventDataset
from paired_token_reliability.unified_loss import (
    event_mass_budget,
    geometry_contribution_rank_loss,
    pair_consistency,
    supervised_log_depth_derivative_losses,
)
from paired_token_reliability.unified_model import contribution_override
from paired_token_reliability.contribution_stage1 import build_bridge_masks
from paired_token_reliability.train_unified_geometry_contribution import (
    build_alternating_phase_schedule,
    normalize_dotlist_overrides,
    use_phase_event_source,
)


def test_temporal_budget_uses_event_mass():
    event = torch.zeros(1, 1, 4, 2, 2)
    event[:, :, 0] = 2.0
    contribution = torch.zeros_like(event)
    contribution[:, :, 0] = 0.5
    loss, ratio = event_mass_budget(contribution, event, rho=0.5)
    torch.testing.assert_close(ratio, torch.tensor([0.5]))
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_supervised_derivatives_penalize_stripes_without_smoothing_gt_detail():
    target = torch.ones(1, 1, 28, 28)
    valid = torch.ones_like(target, dtype=torch.bool)
    exact = supervised_log_depth_derivative_losses(
        target, target, valid, patch_size=14
    )
    for value in exact:
        torch.testing.assert_close(value, torch.tensor(0.0))

    striped = target.clone()
    striped[..., 14:] = 1.1
    gradient, curvature, grid = supervised_log_depth_derivative_losses(
        striped, target, valid, patch_size=14
    )
    assert gradient > 0
    assert curvature > 0
    assert grid > 0


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


def test_phase_a_uses_geometry_event_and_phase_b_uses_full_event():
    full = torch.randn(2, 6, 8, 10)
    geometry = torch.randn_like(full)
    views = [{"event_voxel": full, "geometry_event_voxel": geometry}]
    phase_a = use_phase_event_source(views, "adapter")
    phase_b = use_phase_event_source(views, "contribution")
    assert phase_a[0]["event_voxel"] is geometry
    assert phase_b[0]["event_voxel"] is full
    # Do not mutate the collated full-event view needed by Phase B.
    assert views[0]["event_voxel"] is full


def test_epoch_schedule_warms_up_a_then_alternates_b_a():
    assert build_alternating_phase_schedule(2, 3, 0) == [
        "adapter",
        "adapter",
        "contribution",
        "adapter",
        "contribution",
        "adapter",
        "contribution",
        "adapter",
    ]


def test_hydra_style_prefixes_are_removed_for_plain_omegaconf():
    assert normalize_dotlist_overrides(
        [
            "+data.event_source_mode=decomposition_full",
            "++data.decomposition_supervision=true",
            "data.num_views=6",
        ]
    ) == [
        "data.event_source_mode=decomposition_full",
        "data.decomposition_supervision=true",
        "data.num_views=6",
    ]


def test_event_voxel_bins_use_shared_frame_time_window():
    voxel = MyEventDataset._events_to_antialias_voxel_resize(
        event_xy=torch.tensor([[0, 0]]).numpy(),
        event_t=torch.tensor([0.75]).numpy(),
        event_p=torch.tensor([1.0]).numpy(),
        src_width=2,
        src_height=2,
        dst_width=2,
        dst_height=2,
        num_bins=4,
        time_window=(0.0, 1.0),
    )["event_voxel"]
    assert voxel[3, 0, 0] > 0
    assert voxel[0, 0, 0] == 0


def test_existing_decomposition_streams_use_blender_vertical_flip_fallback():
    dataset = MyEventDataset.__new__(MyEventDataset)
    dataset.event_y_flip = "auto"
    dataset.decomposition_full_branch = "full"
    dataset.decomposition_geo_branch = "geometry_motion"
    metadata = {"event_time_info": {"h5_attrs": {}}, "event_dir": "full"}
    assert dataset._resolve_event_spatial_transform(metadata) == "vflip"
    # Explicit H5 metadata remains authoritative for future datasets.
    metadata["event_time_info"]["h5_attrs"]["y_origin"] = "top_left"
    assert dataset._resolve_event_spatial_transform(metadata) == "none"


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
