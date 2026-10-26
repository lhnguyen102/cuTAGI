import os
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)

from pytagi.nn import Linear, Sequential

TEST_CPU_ONLY = os.getenv("TEST_CPU_ONLY") == "1"


class TestGetOutputs(unittest.TestCase):
    def assert_outputs_match_host_buffers(self, device):
        batch_size, input_size, output_size = 4, 8, 16
        net = Sequential(Linear(input_size, output_size))
        net.to_device(device)
        np.random.seed(5)
        x = np.random.randn(batch_size, input_size).astype(np.float32)

        m_pred, v_pred = net(x)

        net.output_to_host()
        n = batch_size * output_size
        self.assertEqual(m_pred.shape, (n,))
        self.assertEqual(v_pred.shape, (n,))
        np.testing.assert_array_equal(m_pred, net.output_z_buffer.mu_a[:n])
        np.testing.assert_array_equal(v_pred, net.output_z_buffer.var_a[:n])

    def test_cpu_outputs_match_host_buffers(self):
        self.assert_outputs_match_host_buffers("cpu")

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_cuda_outputs_match_host_buffers(self):
        self.assert_outputs_match_host_buffers("cuda")


if __name__ == "__main__":
    unittest.main()
