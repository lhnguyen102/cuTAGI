import os
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)

from pytagi import Utils
from pytagi.nn import Linear, OutputUpdater, Sequential

TEST_CPU_ONLY = os.getenv("TEST_CPU_ONLY") == "1"
VOCAB_SIZE = 50257


class TestSparseReadout(unittest.TestCase):
    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_outputs_are_the_dense_outputs_at_the_path_units(self):
        batch_size, seq_len, embed_dim = 2, 4, 768
        utils = Utils()
        hrc = utils.get_hierarchical_softmax(VOCAB_SIZE)
        net = Sequential(Linear(embed_dim, hrc.len))
        net.to_device("cuda")
        out_updater = OutputUpdater(net.device)

        np.random.seed(3)
        x = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
        labels = np.random.randint(0, VOCAB_SIZE, batch_size * seq_len)
        y_obs, y_idx, n_obs = utils.label_to_obs(
            labels=labels.astype(np.int32), num_classes=VOCAB_SIZE
        )
        n = batch_size * seq_len

        m_dense, v_dense = net(x)
        rows = np.repeat(np.arange(n), n_obs)
        idx = np.asarray(y_idx, dtype=np.int64) - 1
        m_gathered = m_dense.reshape(n, hrc.len)[rows, idx]
        v_gathered = v_dense.reshape(n, hrc.len)[rows, idx]

        net.set_output_indices(y_idx, n_obs)
        m_sparse, v_sparse = net(x)
        self.assertEqual(m_sparse.shape, (n * n_obs,))
        np.testing.assert_array_equal(m_sparse, m_gathered)
        np.testing.assert_array_equal(v_sparse, v_gathered)

        var_y = np.full((n * n_obs,), 4.0, dtype=np.float32)
        out_updater.update(
            output_states=net.output_z_buffer,
            mu_obs=np.asarray(y_obs, dtype=np.float32),
            var_obs=var_y,
            delta_states=net.input_delta_z_buffer,
        )
        net.backward()
        net.step()

        m_after, _ = net(x)
        self.assertEqual(m_after.shape, (n * hrc.len,))

    def test_cpu_linear_rejects_output_indices(self):
        net = Sequential(Linear(8, 16))
        with self.assertRaises(RuntimeError):
            net.set_output_indices(np.array([1, 2], dtype=np.int32), 2)


if __name__ == "__main__":
    unittest.main()
