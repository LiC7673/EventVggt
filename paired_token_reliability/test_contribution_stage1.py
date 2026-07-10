"""CPU tests for the redesigned Stage-1 event-contribution path.

Run with:

    python -m paired_token_reliability.test_contribution_stage1

These tests use synthetic tensors and do not need the dataset or a VGGT
checkpoint.  They are deliberately focused on architectural invariants that
previous repair scripts violated.
"""

from __future__ import annotations

import unittest

import torch

from paired_token_reliability.contribution_dataset import (
    generate_ordered_pairs,
    parse_exposure_sequence,
)
from paired_token_reliability.contribution_stage1 import (
    MultiLdrEventContributionModel,
    build_bridge_masks,
    build_model_from_checkpoint,
    contribution_budget,
    contribution_condition,
    geometry_emphasis_weight,
    orient_exposure_pair,
    stage1_contribution_loss,
)


class Stage1ContributionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.batch = 1
        self.views = 2
        self.bins = 2
        self.height = 16
        self.width = 20

    def _geometry(self):
        depth = torch.full((self.batch, self.views, self.height, self.width), 2.0)
        normals = torch.zeros((self.batch, self.views, self.height, self.width, 3))
        normals[..., 2] = 1.0
        return depth, normals

    def _event(self):
        event = torch.zeros(
            self.batch, self.views, 2 * self.bins, self.height, self.width
        )
        event[..., 5:12, 7:15] = 1.0
        return event

    def test_five_exposures_generate_ten_ordered_pairs(self):
        exposures = parse_exposure_sequence("0,1,2,5,10")
        pairs = generate_ordered_pairs(exposures, "all")
        self.assertEqual(len(pairs), 10)
        self.assertEqual(pairs[0], ("ev_0", "ev_1"))
        self.assertEqual(pairs[-1], ("ev_5", "ev_10"))
        order = {value: index for index, value in enumerate(exposures)}
        self.assertTrue(all(order[left] < order[right] for left, right in pairs))

    def test_adjacent_anchor_and_explicit_pair_modes(self):
        exposures = parse_exposure_sequence("0,1,2,5,10")
        self.assertEqual(
            generate_ordered_pairs(exposures, "adjacent"),
            (("ev_0", "ev_1"), ("ev_1", "ev_2"), ("ev_2", "ev_5"), ("ev_5", "ev_10")),
        )
        self.assertEqual(
            generate_ordered_pairs(exposures, "anchor"),
            (("ev_0", "ev_1"), ("ev_0", "ev_2"), ("ev_0", "ev_5"), ("ev_0", "ev_10")),
        )
        explicit = generate_ordered_pairs(
            exposures, "explicit", ("ev_10->ev_1", "ev_0->ev_5", "ev_0->ev_5")
        )
        self.assertEqual(explicit, (("ev_1", "ev_10"), ("ev_0", "ev_5")))

    def test_exposure_sequence_must_be_increasing(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_exposure_sequence("0,2,1,5,10")

    def test_pair_is_oriented_by_measured_saturation(self):
        clear = torch.full((1, 2, 3, 8, 8), 0.4)
        bad = clear.clone()
        bad[..., :5, :] = 1.0
        reference, oriented_bad, ref_is_a, sat_ref, sat_bad = orient_exposure_pair(
            bad, clear, saturation_threshold=0.95
        )
        self.assertFalse(bool(ref_is_a.item()))
        self.assertTrue(torch.equal(reference, clear))
        self.assertTrue(torch.equal(oriented_bad, bad))
        self.assertGreater(float(sat_bad.item()), float(sat_ref.item()))

    def test_bridge_is_exact_three_way_intersection(self):
        reference = torch.zeros((1, 1, 3, 12, 12))
        reference[..., :, 6:] = 0.6
        bad = torch.ones_like(reference)
        event = torch.zeros((1, 1, 4, 12, 12))
        event[..., 5:7] = 1.0
        masks = build_bridge_masks(
            reference,
            bad,
            event,
            saturation_threshold=0.95,
            reference_gradient_threshold=0.05,
        )
        expected = masks.saturated_bad & masks.visible_reference & masks.event_support
        self.assertTrue(torch.equal(masks.bridge, expected))
        self.assertGreater(int(masks.bridge.sum()), 0)
        event_zero = torch.zeros_like(event)
        no_event = build_bridge_masks(reference, bad, event_zero)
        self.assertEqual(int(no_event.bridge.sum()), 0)

    def test_geometry_detail_is_a_soft_weight_not_a_binary_label(self):
        depth, normals = self._geometry()
        depth[..., self.width // 2 :] = 3.0
        valid = torch.ones_like(depth, dtype=torch.bool)
        weight = geometry_emphasis_weight(
            depth, normals, valid, alpha=2.0, depth_gradient_weight=1.0
        )
        self.assertTrue(bool((weight >= 1.0).all()))
        edge = weight[..., self.width // 2 - 1 : self.width // 2 + 1].mean()
        flat = weight[..., :3].mean()
        self.assertGreater(float(edge), float(flat))
        self.assertGreater(float(flat), 0.0)

    def test_event_mass_budget(self):
        event = self._event()
        half = torch.full(event.shape[:2] + event.shape[-2:], 0.5)
        ones = torch.ones_like(half)
        half_loss, half_mean = contribution_budget(half, event, target_ratio=0.5)
        one_loss, one_mean = contribution_budget(ones, event, target_ratio=0.5)
        self.assertAlmostEqual(float(half_loss), 0.0, places=6)
        self.assertAlmostEqual(float(half_mean.item()), 0.5, places=6)
        self.assertAlmostEqual(float(one_mean.item()), 1.0, places=6)
        self.assertGreater(float(one_loss), float(half_loss))

    def test_full_event_cannot_bypass_zero_contribution(self):
        model = MultiLdrEventContributionModel(
            num_bins=self.bins,
            contribution_channels=8,
            refiner_channels=8,
            coarse_feature_dim=0,
        )
        depth, normals = self._geometry()
        rgb = torch.rand(self.batch, self.views, 3, self.height, self.width)
        event_a = self._event()
        event_b = torch.rand_like(event_a) * 4.0
        zero = torch.zeros(self.batch, self.views, self.height, self.width)
        output_a = model(event_a, rgb, depth, normals, contribution_override=zero)
        output_b = model(event_b, rgb, depth, normals, contribution_override=zero)
        self.assertEqual(int(output_a["selected_event"].abs().sum()), 0)
        self.assertTrue(torch.equal(output_a["depth"], depth))
        self.assertTrue(torch.equal(output_b["depth"], depth))
        self.assertTrue(torch.equal(output_a["depth"], output_b["depth"]))
        self.assertTrue(torch.equal(output_a["normals"], output_b["normals"]))

    def test_proxy_phase_uses_exact_ones_without_running_contribution_net(self):
        model = MultiLdrEventContributionModel(
            num_bins=self.bins,
            contribution_channels=8,
            refiner_channels=8,
            coarse_feature_dim=0,
        )
        depth, normals = self._geometry()
        rgb = torch.rand(self.batch, self.views, 3, self.height, self.width)
        event = self._event()
        ones = torch.ones_like(depth)

        def forbidden(*args, **kwargs):
            raise AssertionError("ContributionNet must not run during proxy pretraining")

        model.contribution_net.forward = forbidden
        output = model(
            event,
            rgb,
            depth,
            normals,
            contribution_override=ones,
            bypass_contribution_net=True,
        )
        self.assertTrue(torch.equal(output["contribution"], ones))
        self.assertTrue(torch.equal(output["selected_event"], event))

    def test_small_contribution_cannot_unlock_full_bias_correction(self):
        model = MultiLdrEventContributionModel(
            num_bins=self.bins,
            contribution_channels=8,
            refiner_channels=8,
            coarse_feature_dim=0,
        )
        with torch.no_grad():
            model.event_refiner.output.bias[0] = 1.0
        depth, normals = self._geometry()
        rgb = torch.rand(self.batch, self.views, 3, self.height, self.width)
        event = self._event()
        low = torch.full(depth.shape, 0.1)
        high = torch.ones_like(low)
        low_output = model(event, rgb, depth, normals, contribution_override=low)
        high_output = model(event, rgb, depth, normals, contribution_override=high)
        low_delta = low_output["delta_log_depth"].abs().sum()
        high_delta = high_output["delta_log_depth"].abs().sum()
        self.assertGreater(float(high_delta), float(low_delta))

    def test_coarse_geometry_is_detached_and_gt_is_fixed(self):
        model = MultiLdrEventContributionModel(
            num_bins=self.bins,
            contribution_channels=8,
            refiner_channels=8,
            coarse_feature_dim=0,
        )
        coarse_depth, coarse_normals = self._geometry()
        coarse_depth.requires_grad_(True)
        coarse_normals.requires_grad_(True)
        rgb = torch.rand(self.batch, self.views, 3, self.height, self.width)
        event = self._event()
        prediction = model(event, rgb, coarse_depth, coarse_normals)
        depth_gt = torch.full_like(coarse_depth, 2.2)
        normal_gt = coarse_normals.detach().clone()
        valid = torch.ones_like(depth_gt, dtype=torch.bool)
        bridge = event.abs().sum(dim=2) > 0
        loss = stage1_contribution_loss(
            prediction["depth"],
            prediction["normals"],
            depth_gt,
            normal_gt,
            valid,
            bridge,
            torch.ones_like(depth_gt),
            prediction["contribution"],
            event,
            minimum_bridge_area=0.0,
            budget_weight=0.1,
        )
        loss.loss.backward()
        self.assertIsNone(coarse_depth.grad)
        self.assertIsNone(coarse_normals.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in model.event_refiner.parameters()))

    def test_inactive_bridge_is_skipped_without_nan(self):
        depth, normals = self._geometry()
        event = self._event()
        contribution = torch.full(depth.shape, 0.5, requires_grad=True)
        result = stage1_contribution_loss(
            depth,
            normals,
            depth,
            normals,
            torch.ones_like(depth, dtype=torch.bool),
            torch.zeros_like(depth, dtype=torch.bool),
            torch.ones_like(depth),
            contribution,
            event,
            minimum_bridge_area=0.01,
        )
        self.assertEqual(int(result.active_samples.sum()), 0)
        self.assertTrue(torch.isfinite(result.loss))
        self.assertAlmostEqual(float(result.loss), 0.0, places=7)

    def test_counterfactual_conditions_preserve_fair_comparisons(self):
        event = self._event()
        learned = torch.linspace(0.0, 1.0, self.height * self.width).reshape(
            1, 1, self.height, self.width
        ).repeat(1, self.views, 1, 1)
        random_map = contribution_condition(
            "random_same_mean", learned, event, generator=torch.Generator().manual_seed(3)
        )
        active = event.abs().sum(dim=2) > 0
        self.assertAlmostEqual(float(random_map[active].mean()), float(learned[active].mean()), places=6)
        full = contribution_condition("full_event", learned, event)
        zero = contribution_condition("coarse_rgb", learned, event)
        self.assertTrue(bool((full == 1).all()))
        self.assertTrue(bool((zero == 0).all()))

        drop_high = contribution_condition("drop_high", learned, event, drop_fraction=0.25)
        drop_low = contribution_condition("drop_low", learned, event, drop_fraction=0.25)
        removed_high = (learned - drop_high)[active]
        removed_low = (learned - drop_low)[active]
        self.assertEqual(int((removed_high > 0).sum()), int((removed_low > 0).sum()))
        self.assertGreater(float(removed_high.sum()), float(removed_low.sum()))

    def test_legacy_reliability_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Legacy ReliabilityUNet"):
            build_model_from_checkpoint({"model": {}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
