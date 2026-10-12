import os
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)

from pytagi.nn import OutputUpdater, RMSNorm, Sequential


class TestRMSNorm(unittest.TestCase):
    def test_forward_pass(self):
        batch_size = 2
        embed_dim = 8

        model = Sequential(RMSNorm(normalized_shape=[embed_dim], eps=1e-6))

        np.random.seed(42)
        input_data = np.random.randn(batch_size * embed_dim).astype(np.float32)

        m_pred, v_pred = model(input_data)

        self.assertEqual(m_pred.shape[0], batch_size * embed_dim)
        self.assertEqual(v_pred.shape[0], batch_size * embed_dim)

        self.assertFalse(np.any(np.isnan(m_pred)))
        self.assertFalse(np.any(np.isnan(v_pred)))
        self.assertFalse(np.any(np.isinf(m_pred)))
        self.assertFalse(np.any(np.isinf(v_pred)))

    def test_backward_pass(self):
        batch_size = 2
        embed_dim = 8

        model = Sequential(RMSNorm(normalized_shape=[embed_dim], eps=1e-6))

        np.random.seed(456)
        input_data = np.random.randn(batch_size * embed_dim).astype(np.float32)
        target_data = np.random.randn(batch_size * embed_dim).astype(np.float32)

        m_pred, v_pred = model(input_data)

        var_obs = np.full((batch_size * embed_dim,), 0.1, dtype=np.float32)
        out_updater = OutputUpdater(model.device)
        out_updater.update(
            output_states=model.output_z_buffer,
            mu_obs=target_data,
            var_obs=var_obs,
            delta_states=model.input_delta_z_buffer,
        )

        model.backward()
        model.step()

        self.assertFalse(np.any(np.isnan(m_pred)))
        self.assertFalse(np.any(np.isnan(v_pred)))

    def test_gain_w_sets_prior_weight_variance(self):
        num_features = 8
        model = Sequential(RMSNorm(normalized_shape=[num_features], gain_w=2.0))

        mu_w, var_w, _, _ = model.parameters()[0]

        np.testing.assert_allclose(mu_w, 1.0)
        np.testing.assert_allclose(var_w, (2.0 / num_features) ** 2, rtol=1e-6)

    def test_default_eps_is_one_e_minus_six(self):
        layer = RMSNorm(normalized_shape=[4])
        self.assertAlmostEqual(layer.eps, 1e-6)


if __name__ == "__main__":
    unittest.main()
