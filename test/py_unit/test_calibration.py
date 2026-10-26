import os
import sys
import unittest
from unittest import mock

import numpy as np

from pytagi import Utils
from pytagi.nn import Embedding, Linear, Sequential

# path to binding code
sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)

from examples.fineweb_gpt2 import (
    ALPHA,
    VOCAB_SIZE,
    fit_level_calibration,
    fit_level_links,
)

TEST_CPU_ONLY = os.getenv("TEST_CPU_ONLY") == "1"


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

    def test_batch_rows_match_single_rows(self):
        rows = 3
        rng = np.random.default_rng(1)
        ma = rng.normal(size=(rows, self.hrc.len)).astype(np.float32)
        Sa = rng.uniform(0.1, 1.0, size=(rows, self.hrc.len)).astype(np.float32)

        prob = self.utils.obs_to_label_prob(ma, Sa, self.hrc, self.num_classes)

        self.assertEqual(prob.shape, (rows, self.num_classes))
        for r in range(rows):
            np.testing.assert_array_equal(
                prob[r],
                self.utils.obs_to_label_prob(
                    ma[r], Sa[r], self.hrc, self.num_classes
                ),
            )

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_cuda_batch_matches_cpu(self):
        rows = 5
        rng = np.random.default_rng(2)
        ma = 2.0 * rng.normal(size=(rows, self.hrc.len)).astype(np.float32)
        Sa = rng.uniform(0.1, 1.0, size=(rows, self.hrc.len)).astype(np.float32)
        calibration = dict(
            level_scale=np.linspace(0.5, 1.5, self.num_levels),
            level_offset=np.linspace(-0.3, 0.3, self.num_levels),
        )
        for kwargs in ({}, calibration):
            cpu = self.utils.obs_to_label_prob(
                ma, Sa, self.hrc, self.num_classes, **kwargs
            )
            cuda = self.utils.obs_to_label_prob(
                ma, Sa, self.hrc, self.num_classes, device="cuda", **kwargs
            )
            # Nodes near saturation make |P_z - 1| an ulp-level quantity on
            # both devices, so tiny class probabilities only agree absolutely.
            np.testing.assert_allclose(cuda, cpu, rtol=1e-6, atol=2e-7)

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


class TestFitLevelCalibration(unittest.TestCase):
    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_sparse_forward_fits_the_dense_links(self):
        batch_size, seq_len, n_batches, obs_scale = 2, 4, 3, 3.0
        utils = Utils()
        hrc = utils.get_hierarchical_softmax(VOCAB_SIZE)
        net = Sequential(
            Embedding(VOCAB_SIZE, 8, input_size=seq_len), Linear(8, hrc.len)
        )
        net.to_device("cuda")
        rng = np.random.default_rng(0)
        batches = [
            (
                rng.integers(0, VOCAB_SIZE, (batch_size, seq_len, 1)).astype(
                    np.float32
                ),
                rng.integers(0, VOCAB_SIZE, batch_size * seq_len).astype(
                    np.int32
                ),
            )
            for _ in range(n_batches)
        ]

        with mock.patch(
            "examples.fineweb_gpt2.get_batch", side_effect=list(batches)
        ):
            scale, offset = fit_level_calibration(
                net, utils, hrc, batch_size, seq_len, obs_scale, n_batches
            )

        zs, obs_all = [], []
        for x, labels in batches:
            m_pred, v_pred = net(x)
            n = len(labels)
            m = m_pred.reshape(n, hrc.len) / obs_scale
            v = v_pred.reshape(n, hrc.len) / obs_scale**2
            obs, idx, n_obs = utils.label_to_obs(labels, VOCAB_SIZE)
            obs = obs.reshape(n, n_obs)
            idx = idx.reshape(n, n_obs).astype(np.int64) - 1
            rows = np.arange(n)[:, None]
            zs.append(m[rows, idx] / np.sqrt(1.0 / ALPHA**2 + v[rows, idx]))
            obs_all.append(obs)
        ref_scale, ref_offset = fit_level_links(
            np.concatenate(zs), np.concatenate(obs_all)
        )

        np.testing.assert_array_equal(scale, ref_scale)
        np.testing.assert_array_equal(offset, ref_offset)


if __name__ == "__main__":
    unittest.main()
