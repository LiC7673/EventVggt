"""CPU-only invariants for contribution-map ablation modes."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from ab_st1_st2 import METHODS
from ab_st1_st2.evaluate import ContributionStats
from ab_st1_st2.model import AblationStreamVGGT


class AblationTests(unittest.TestCase):
    def _shell(self, method):
        model = AblationStreamVGGT.__new__(AblationStreamVGGT)
        nn.Module.__init__(model)
        model.ablation_method = method
        model.saturation_threshold = 0.98
        return model

    def _views(self):
        views = []
        for _ in range(2):
            image = torch.zeros(1, 3, 8, 10)
            image[..., :4, :] = 1.0
            views.append(
                {
                    "img": image,
                    "event_voxel": torch.randn(1, 20, 8, 10),
                    "camera_intrinsics": torch.eye(3).unsqueeze(0),
                }
            )
        return views

    def test_all_requested_methods_exist(self):
        self.assertEqual(
            METHODS,
            ("rgb_only", "raw_event", "ours", "no_multildr", "saturation_mask"),
        )

    def test_rgb_and_raw_are_exact_zero_and_one(self):
        views = self._views()
        rgb = self._shell("rgb_only")._fixed_contribution(views)
        raw = self._shell("raw_event")._fixed_contribution(views)
        self.assertTrue(bool((rgb == 0).all()))
        self.assertTrue(bool((raw == 1).all()))

    def test_saturation_mask_is_spatial_and_binary(self):
        value = self._shell("saturation_mask")._fixed_contribution(self._views())
        self.assertEqual(set(value.unique().tolist()), {0.0, 1.0})
        self.assertGreater(float(value.std()), 0.0)

    def test_learned_modes_have_no_fixed_override(self):
        views = self._views()
        self.assertIsNone(self._shell("ours")._fixed_contribution(views))
        self.assertIsNone(self._shell("no_multildr")._fixed_contribution(views))

    def test_collapse_statistic_is_not_hidden(self):
        stats = ContributionStats()
        stats.update(torch.full((2, 3, 4), 0.5))
        result = stats.compute()
        self.assertAlmostEqual(result["mean"], 0.5)
        self.assertAlmostEqual(result["std"], 0.0)
        self.assertEqual(result["min"], result["max"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
