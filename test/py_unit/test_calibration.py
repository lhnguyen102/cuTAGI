import os
import sys
import unittest

import numpy as np

from pytagi import Utils

# path to binding code
sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)


class TestReadoutCalibration(unittest.TestCase):
    def setUp(self):
        self.num_classes = 8
        self.utils = Utils()
        self.hrc = self.utils.get_hierarchical_softmax(self.num_classes)
        self.num_levels = self.hrc.num_obs
        rng = np.random.default_rng(0)
        self.ma = rng.normal(size=self.hrc.len).astype(np.float32)
        self.Sa = rng.uniform(0.1, 1.0, size=self.hrc.len).astype(np.float32)

    def label_prob(self, **calibration):
        return self.utils.obs_to_label_prob(
            self.ma, self.Sa, self.hrc, self.num_classes, **calibration
        )

    def test_identity_calibration_matches_uncalibrated(self):
        uncalibrated = self.label_prob()
        calibrated = self.label_prob(
            level_scale=np.ones(self.num_levels),
            level_offset=np.zeros(self.num_levels),
        )
        np.testing.assert_allclose(calibrated, uncalibrated, rtol=1e-6)

    def test_none_matches_uncalibrated(self):
        np.testing.assert_array_equal(
            self.label_prob(level_scale=None, level_offset=None),
            self.label_prob(),
        )

    def test_zero_scale_gives_uniform_classes(self):
        # Every node probability becomes Phi(0) = 0.5 on a 3-level tree.
        prob = self.label_prob(
            level_scale=np.zeros(self.num_levels),
            level_offset=np.zeros(self.num_levels),
        )
        np.testing.assert_allclose(prob, np.full(self.num_classes, 1 / 8))

    def test_offset_acts_on_its_level_only(self):
        # Level 0 is the root split: Phi(10) ~ 1 sends all mass to one half,
        # deeper levels at Phi(0) = 0.5 split it evenly over 4 classes.
        prob = self.label_prob(
            level_scale=np.zeros(self.num_levels),
            level_offset=np.array([10.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            np.sort(prob), [0] * 4 + [0.25] * 4, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
