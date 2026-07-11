"""Small CPU invariance tests for the Stage-2 geometry adapters."""

from __future__ import annotations

import inspect
import unittest

import torch

from stage2_geometry_adapter.model import (
    GeometryFeatureAdapter,
    PolarityTemporalEventPyramid,
    StreamVGGT,
    dpt_feature_shapes,
)


class GeometryAdapterTests(unittest.TestCase):
    def test_zero_initialized_alpha_is_exact_identity(self):
        adapter = GeometryFeatureAdapter(32, 8, 16)
        rgb = torch.randn(2, 32, 10, 12)
        event = torch.randn(2, 8, 10, 12)
        contribution = torch.rand(2, 1, 10, 12)
        refined, update, penalty = adapter(rgb, event, contribution)
        self.assertTrue(torch.equal(refined, rgb))
        self.assertEqual(float(update.abs().sum()), 0.0)
        self.assertEqual(float(penalty), 0.0)

    def test_contribution_zero_blocks_feature_update(self):
        adapter = GeometryFeatureAdapter(16, 8, 16)
        with torch.no_grad():
            adapter.alpha_logit.fill_(0.5)
        rgb = torch.randn(1, 16, 8, 8)
        event = torch.randn(1, 8, 8, 8)
        contribution = torch.zeros(1, 1, 8, 8)
        refined, update, _ = adapter(rgb, event, contribution)
        self.assertTrue(torch.equal(refined, rgb))
        self.assertEqual(float(update.abs().sum()), 0.0)

    def test_nonzero_contribution_is_not_applied_twice(self):
        adapter = GeometryFeatureAdapter(16, 8, 16)
        with torch.no_grad():
            adapter.alpha_logit.fill_(0.5)
        rgb = torch.randn(1, 16, 8, 8)
        event = torch.randn(1, 8, 8, 8)
        low = torch.full((1, 1, 8, 8), 0.2)
        high = torch.full((1, 1, 8, 8), 0.8)
        _, low_update, low_penalty = adapter(rgb, event, low)
        _, high_update, high_penalty = adapter(rgb, event, high)
        torch.testing.assert_close(low_update, high_update)
        self.assertGreater(float(low_penalty), float(high_penalty))

    def test_zero_events_have_zero_multiscale_gate(self):
        encoder = PolarityTemporalEventPyramid(
            num_bins=2, hidden_channels=8, pyramid_channels=8
        )
        voxel = torch.zeros(1, 2, 4, 16, 20)
        contribution = torch.ones(1, 2, 16, 20)
        shapes = dpt_feature_shapes(16, 20, 4)
        _, gates = encoder(voxel, contribution, shapes)
        self.assertTrue(all(float(gate.abs().sum()) == 0.0 for gate in gates))

    def test_expected_four_dpt_scales(self):
        self.assertEqual(dpt_feature_shapes(392, 518, 14), [(112, 148), (56, 74), (28, 37), (14, 19)])

    def test_model_has_no_depth_residual_api(self):
        source = inspect.getsource(StreamVGGT.forward)
        self.assertNotIn("depth_residual", source)
        self.assertNotIn("delta_log_depth", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
