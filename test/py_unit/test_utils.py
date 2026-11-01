import os
import sys
import unittest

import numpy as np

from pytagi import Utils
from pytagi.nn import Linear, OutputUpdater, Sequential

# path to binding code
sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)


class TestUtils(unittest.TestCase):
    def test_get_errors_returns_one_prediction_per_sample(self):
        num_classes, batch_size = 4, 2
        utils = Utils()
        hrc = utils.get_hierarchical_softmax(num_classes)
        ma = np.zeros(batch_size * hrc.len, dtype=np.float32)
        Sa = np.ones(batch_size * hrc.len, dtype=np.float32)
        labels = np.array([0, 3], dtype=np.int32)

        pred, prob = utils.get_errors(ma, Sa, labels, num_classes, batch_size)

        self.assertEqual(len(pred), batch_size)
        self.assertEqual(len(prob), batch_size * num_classes)

    def test_label_to_obs_matches_the_cpp_wrapper(self):
        utils = Utils()
        for num_classes in (10, 50267):
            np.random.seed(0)
            labels = np.random.randint(0, num_classes, 1024)
            labels = np.concatenate([labels, [0, num_classes - 1]])
            ref_obs, ref_idx, ref_num_obs = (
                utils._cpp_backend.label_to_obs_wrapper(labels, num_classes)
            )

            obs, idx, num_obs = utils.label_to_obs(labels, num_classes)

            self.assertEqual(num_obs, ref_num_obs)
            self.assertEqual(obs.dtype, np.float32)
            self.assertEqual(idx.dtype, np.int32)
            np.testing.assert_array_equal(obs, np.asarray(ref_obs))
            np.testing.assert_array_equal(idx, np.asarray(ref_idx))


class TestOutputUpdater(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.utils = Utils()
        hrc = self.utils.get_hierarchical_softmax(self.num_classes)
        self.net = Sequential(Linear(3, hrc.len))
        self.out_updater = OutputUpdater(self.net.device)
        np.random.seed(1)
        self.net(np.random.randn(4, 3).astype(np.float32))
        labels = np.array([0, 3, 7, 9])
        obs, self.idx, _ = self.utils.label_to_obs(labels, self.num_classes)
        self.mu_obs = obs
        self.var_obs = np.full(obs.shape, 0.25, dtype=np.float32)

    def _delta_states(self):
        delta = self.net.input_delta_z_buffer
        return np.array(delta.delta_mu), np.array(delta.delta_var)

    def test_update_with_numpy_arrays_matches_the_list_path(self):
        self.out_updater._cpp_backend.update(
            self.net.output_z_buffer,
            self.mu_obs.tolist(),
            self.var_obs.tolist(),
            self.net.input_delta_z_buffer,
        )
        ref_mu, ref_var = self._delta_states()

        self.out_updater.update(
            self.net.output_z_buffer,
            self.mu_obs,
            self.var_obs,
            self.net.input_delta_z_buffer,
        )
        mu, var = self._delta_states()

        np.testing.assert_array_equal(mu, ref_mu)
        np.testing.assert_array_equal(var, ref_var)
        self.assertTrue(np.any(mu != 0))

    def test_update_using_indices_with_numpy_arrays_matches_the_list_path(
        self,
    ):
        self.out_updater._cpp_backend.update_using_indices(
            self.net.output_z_buffer,
            self.mu_obs.tolist(),
            self.var_obs.tolist(),
            self.idx.tolist(),
            self.net.input_delta_z_buffer,
        )
        ref_mu, ref_var = self._delta_states()

        self.out_updater.update_using_indices(
            self.net.output_z_buffer,
            self.mu_obs,
            self.var_obs,
            self.idx,
            self.net.input_delta_z_buffer,
        )
        mu, var = self._delta_states()

        np.testing.assert_array_equal(mu, ref_mu)
        np.testing.assert_array_equal(var, ref_var)
        self.assertTrue(np.any(mu != 0))


if __name__ == "__main__":
    unittest.main()
