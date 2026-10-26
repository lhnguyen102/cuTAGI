import os
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)

from pytagi.nn import (
    LayerBlock,
    Linear,
    MultiheadAttention,
    OutputUpdater,
    RMSNorm,
    Sequential,
)

TEST_CPU_ONLY = os.getenv("TEST_CPU_ONLY") == "1"


class TestMultiheadAttention(unittest.TestCase):
    def test_forward_pass(self):
        batch_size = 2
        seq_len = 4
        embed_dim = 8
        num_heads = 2

        model = Sequential(
            MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                seq_len=seq_len,
                bias=False,
                gain_weight=1.0,
                gain_bias=1.0,
                init_method="Xavier",
            )
        )

        np.random.seed(42)
        input_data = np.random.randn(batch_size, seq_len, embed_dim).astype(
            np.float32
        )

        m_pred, v_pred = model(input_data)

        self.assertEqual(m_pred.shape[0], batch_size * seq_len * embed_dim)
        self.assertEqual(v_pred.shape[0], batch_size * seq_len * embed_dim)

        self.assertFalse(np.any(np.isnan(m_pred)))
        self.assertFalse(np.any(np.isnan(v_pred)))

    def test_backward_pass(self):
        batch_size = 2
        seq_len = 4
        embed_dim = 8
        num_heads = 2

        model = Sequential(
            MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                bias=False,
                gain_weight=1.0,
                gain_bias=1.0,
                init_method="Xavier",
            )
        )

        np.random.seed(456)
        input_data = np.random.randn(batch_size * seq_len * embed_dim).astype(
            np.float32
        )
        target_data = np.random.randn(batch_size * seq_len * embed_dim).astype(
            np.float32
        )

        m_pred, v_pred = model(input_data)

        var_obs = np.full(
            (batch_size * seq_len * embed_dim,), 0.1, dtype=np.float32
        )
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


class TestAttentionScores(unittest.TestCase):
    def check_scores_are_keyed_depth_first_with_head_shapes(self, device):
        batch_size, seq_len, embed_dim = 2, 4, 8
        model = Sequential(
            MultiheadAttention(embed_dim, num_heads=2, seq_len=seq_len),
            LayerBlock(
                RMSNorm([embed_dim]),
                MultiheadAttention(embed_dim, num_heads=4, seq_len=seq_len),
                Linear(embed_dim, embed_dim),
            ),
        )
        model.to_device(device)
        x = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        model(x)

        scores = model.get_attention_scores()

        self.assertEqual(sorted(scores.keys()), [0, 1])
        for key, num_heads in [(0, 2), (1, 4)]:
            shape = (batch_size, num_heads, seq_len, seq_len)
            for array in (
                scores[key].mu_qk,
                scores[key].var_qk,
                scores[key].mu_att,
                scores[key].var_att,
            ):
                self.assertEqual(array.shape, shape)

    def test_scores_are_keyed_depth_first_with_head_shapes_on_cpu(self):
        self.check_scores_are_keyed_depth_first_with_head_shapes("cpu")

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_scores_are_keyed_depth_first_with_head_shapes_on_cuda(self):
        self.check_scores_are_keyed_depth_first_with_head_shapes("cuda")

    def test_pre_remax_scores_are_scaled_query_key_products(self):
        batch_size, seq_len, embed_dim, num_heads = 2, 3, 8, 2
        head_dim = embed_dim // num_heads
        attn = MultiheadAttention(
            embed_dim, num_heads, seq_len=seq_len, bias=False, use_rope=False
        )
        model = Sequential(attn)
        x = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        model(x)

        # Weights are output-major: rows 0..E-1 project Q, E..2E-1 project K.
        w = np.asarray(attn.mu_w).reshape(3 * embed_dim, embed_dim)
        q = (x @ w[:embed_dim].T).reshape(
            batch_size, seq_len, num_heads, head_dim
        )
        k = (x @ w[embed_dim : 2 * embed_dim].T).reshape(
            batch_size, seq_len, num_heads, head_dim
        )
        expected = np.einsum("bthd,bshd->bhts", q, k) / np.sqrt(head_dim)

        scores = model.get_attention_scores()[0]
        np.testing.assert_allclose(scores.mu_qk, expected, rtol=1e-4, atol=1e-5)
        causal_rows = np.tril(np.ones((seq_len, seq_len)))
        np.testing.assert_allclose(
            scores.mu_att.sum(axis=-1), 1.0, rtol=1e-4, atol=1e-4
        )
        self.assertTrue(np.all(scores.mu_att[..., causal_rows == 0] == 0.0))


class TestAttentionInLayerBlock(unittest.TestCase):
    def check_one_step_updates_every_layer(self, device: str):
        batch_size, seq_len, embed_dim = 2, 4, 8
        model = Sequential(
            LayerBlock(
                RMSNorm([embed_dim]),
                MultiheadAttention(embed_dim, num_heads=2, seq_len=seq_len),
                Linear(embed_dim, embed_dim),
            )
        )
        model.to_device(device)
        x = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        y = np.full(batch_size * seq_len * embed_dim, 0.5, dtype=np.float32)
        var_y = np.full_like(y, 0.1)

        mu_before = [p[0].copy() for p in model.parameters()]
        m_pred, v_pred = model(x)
        OutputUpdater(model.device).update(
            output_states=model.output_z_buffer,
            mu_obs=y,
            var_obs=var_y,
            delta_states=model.input_delta_z_buffer,
        )
        model.backward()
        model.step()
        model.params_to_host()
        mu_after = [p[0] for p in model.parameters()]

        self.assertTrue(np.all(np.isfinite(m_pred)))
        self.assertTrue(np.all(v_pred > 0))
        self.assertEqual(len(mu_after), 3)  # RMSNorm, MHA, Linear
        for before, after in zip(mu_before, mu_after):
            self.assertTrue(np.all(np.isfinite(after)))
            self.assertFalse(np.allclose(before, after))

    def test_rmsnorm_attention_linear_trains_on_cpu(self):
        self.check_one_step_updates_every_layer("cpu")

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_rmsnorm_attention_linear_trains_on_cuda(self):
        self.check_one_step_updates_every_layer("cuda")


if __name__ == "__main__":
    unittest.main()
