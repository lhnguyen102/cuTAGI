import os
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)

from pytagi.nn import Embedding, Sequential


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


if __name__ == "__main__":
    unittest.main()
