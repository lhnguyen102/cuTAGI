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
    Embedding,
    Linear,
    OutputUpdater,
    PositionalEncoding,
    RMSNorm,
    Sequential,
)

TEST_CPU_ONLY = os.getenv("TEST_CPU_ONLY") == "1"


class TestEmbedding(unittest.TestCase):
    def test_embedding_with_padding(self):
        """Test embedding layer with padding_idx=0"""
        num_embeddings = 10
        embedding_dim = 4
        padding_idx = 0
        batch_size = 2
        num_inputs = 3

        model = Sequential(
            Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                input_size=num_inputs,
                scale=1.0,
                padding_idx=padding_idx,
            )
        )

        input_indices = np.array(
            [[0, 1, 2], [0, 3, 4]], dtype=np.float32
        ).flatten()

        m_pred, v_pred = model(input_indices)

        mu_out = m_pred.reshape(batch_size, num_inputs, embedding_dim)
        var_out = v_pred.reshape(batch_size, num_inputs, embedding_dim)

        self.assertTrue(np.allclose(mu_out[0, 0, :], 0.0))
        self.assertTrue(np.allclose(var_out[0, 0, :], 0.0))
        self.assertTrue(np.allclose(mu_out[1, 0, :], 0.0))
        self.assertTrue(np.allclose(var_out[1, 0, :], 0.0))

        self.assertFalse(np.allclose(mu_out[0, 1, :], 0.0))
        self.assertFalse(np.allclose(var_out[0, 1, :], 0.0))
        self.assertFalse(np.allclose(mu_out[0, 2, :], 0.0))
        self.assertFalse(np.allclose(var_out[0, 2, :], 0.0))


class TestTokenModel(unittest.TestCase):
    def check_one_step_updates_every_layer(self, device: str):
        vocab_size, embed_dim, seq_len, batch_size = 10, 6, 4, 2
        num_outputs = 3
        model = Sequential(
            Embedding(vocab_size, embed_dim, input_size=seq_len, scale=0.15),
            PositionalEncoding(embed_dim, max_seq_len=8),
            RMSNorm([embed_dim]),
            Linear(embed_dim, num_outputs),
        )
        model.to_device(device)
        tokens = np.arange(1, batch_size * seq_len + 1, dtype=np.float32)
        y = np.full(batch_size * seq_len * num_outputs, 0.5, dtype=np.float32)
        var_y = np.full_like(y, 0.1)

        mu_before = [p[0].copy() for p in model.parameters()]
        m_pred, v_pred = model(tokens)
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

        self.assertEqual(m_pred.shape[0], y.shape[0])
        self.assertTrue(np.all(np.isfinite(m_pred)))
        self.assertTrue(np.all(v_pred > 0))
        self.assertEqual(len(mu_after), 3)  # Embedding, RMSNorm, Linear
        for before, after in zip(mu_before, mu_after):
            self.assertTrue(np.all(np.isfinite(after)))
            self.assertFalse(np.allclose(before, after))

    def test_embedding_pe_rmsnorm_linear_trains_on_cpu(self):
        self.check_one_step_updates_every_layer("cpu")

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_embedding_pe_rmsnorm_linear_trains_on_cuda(self):
        self.check_one_step_updates_every_layer("cuda")


if __name__ == "__main__":
    unittest.main()
