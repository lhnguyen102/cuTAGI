import os
import sys
import unittest

import numpy as np

import pytagi
from examples.data_loader import MnistDataLoader
from pytagi import HRCSoftmaxMetric
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    LayerNorm,
    Linear,
    MixtureReLU,
    OutputUpdater,
    ReLU,
    Sequential,
    MaxPool2d,
)

# path to binding code
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

TEST_CPU_ONLY = os.getenv("TEST_CPU_ONLY") == "1"

np.random.seed(0)


def mnist_test_runner(
    model: Sequential,
    batch_size: int = 16,
    num_iters: int = 10,
    use_cuda: bool = False,
):
    train_dtl = MnistDataLoader(
        x_file="data/mnist/train-images-idx3-ubyte",
        y_file="data/mnist/train-labels-idx1-ubyte",
        num_images=60000,
    )
    metric = HRCSoftmaxMetric(num_classes=10)
    error_rates = []
    model.to_device("cuda" if use_cuda else "cpu")

    # Training
    out_updater = OutputUpdater(model.device)
    error_rates = []
    var_y = np.full((batch_size * metric.hrc_softmax.num_obs,), 1, dtype=np.float32)
    batch_iter = train_dtl.create_data_loader(batch_size=batch_size)
    for i, (x, y, y_idx, label) in enumerate(batch_iter):
        if i >= num_iters:
            break
        # Feedforward and backward pass
        m_pred, v_pred = model(x)

        # Update output layers based on targets
        out_updater.update_using_indices(
            output_states=model.output_z_buffer,
            mu_obs=y,
            var_obs=var_y,
            selected_idx=y_idx,
            delta_states=model.input_delta_z_buffer,
        )

        # Update parameters
        model.backward()
        model.step()

        # Training metric
        error_rate = metric.error_rate(m_pred, v_pred, label)
        error_rates.append(error_rate)

    # Averaged error
    avg_error_rate = sum(error_rates[-100:]) / 100.0

    return avg_error_rate


class MnistTest(unittest.TestCase):

    def setUp(self):
        self.threshold = 0.5

    def test_fnn_CPU(self):
        model = Sequential(
            Linear(784, 32), ReLU(), Linear(32, 32), ReLU(), Linear(32, 11)
        )
        avg_error_rate = mnist_test_runner(model)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    def test_mixturerelu_CPU(self):
        model = Sequential(
            Linear(784, 32),
            MixtureReLU(),
            Linear(32, 32),
            MixtureReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    def test_batchnorm_fnn_CPU(self):
        model = Sequential(
            Linear(784, 32),
            BatchNorm2d(32),
            ReLU(),
            Linear(32, 32),
            BatchNorm2d(32),
            ReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    def test_batchnorm_without_bias_fnn_CPU(self):
        model = Sequential(
            Linear(784, 32),
            BatchNorm2d(32, bias=False),
            ReLU(),
            Linear(32, 32),
            BatchNorm2d(32, bias=False),
            ReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    def test_layernorm_fnn_CPU(self):
        model = Sequential(
            Linear(784, 32),
            ReLU(),
            LayerNorm((32,)),
            Linear(32, 32),
            ReLU(),
            LayerNorm((32,)),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    def test_cnn_CPU(self):
        model = Sequential(
            Conv2d(
                1, 8, 4, padding=1, stride=1, padding_type=1, in_width=28, in_height=28
            ),
            ReLU(),
            AvgPool2d(3, 2),
            Conv2d(8, 8, 5),
            ReLU(),
            AvgPool2d(3, 2),
            Linear(8 * 4 * 4, 32),
            ReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    def test_maxpooling_CPU(self):
        model = Sequential(
            Conv2d(
                1, 8, 4, padding=1, stride=1, padding_type=1, in_width=28, in_height=28
            ),
            ReLU(),
            MaxPool2d(3, 2),
            Conv2d(8, 8, 5),
            ReLU(),
            MaxPool2d(3, 2),
            Linear(8 * 4 * 4, 32),
            ReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    def test_batchnorm_cnn_CPU(self):
        model = Sequential(
            Conv2d(
                1,
                8,
                4,
                padding=1,
                stride=1,
                padding_type=1,
                in_width=28,
                in_height=28,
                bias=False,
            ),
            ReLU(),
            BatchNorm2d(8),
            AvgPool2d(3, 2),
            Conv2d(8, 8, 5, bias=False),
            ReLU(),
            BatchNorm2d(8),
            AvgPool2d(3, 2),
            Linear(8 * 4 * 4, 32),
            ReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    def test_layernorm_cnn_CPU(self):
        model = Sequential(
            Conv2d(
                1,
                8,
                4,
                padding=1,
                stride=1,
                padding_type=1,
                in_width=28,
                in_height=28,
                bias=False,
            ),
            ReLU(),
            LayerNorm((8, 27, 27)),
            AvgPool2d(3, 2),
            Conv2d(8, 8, 5, bias=False),
            ReLU(),
            LayerNorm((8, 9, 9)),
            AvgPool2d(3, 2),
            Linear(8 * 4 * 4, 32),
            ReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    # CUDA Tests
    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_fnn_CUDA(self):
        if not pytagi.cuda.is_available():
            self.skipTest("CUDA is not available")
        model = Sequential(
            Linear(784, 32), ReLU(), Linear(32, 32), ReLU(), Linear(32, 11)
        )
        avg_error_rate = mnist_test_runner(model, use_cuda=True)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_mixturerelu_CUDA(self):
        if not pytagi.cuda.is_available():
            self.skipTest("CUDA is not available")
        model = Sequential(
            Linear(784, 32),
            MixtureReLU(),
            Linear(32, 32),
            MixtureReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model, use_cuda=True)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_batchnorm_fnn_CUDA(self):
        if not pytagi.cuda.is_available():
            self.skipTest("CUDA is not available")
        model = Sequential(
            Linear(784, 32),
            BatchNorm2d(32),
            ReLU(),
            Linear(32, 32),
            BatchNorm2d(32),
            ReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model, use_cuda=True)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_layernorm_fnn_CUDA(self):
        if not pytagi.cuda.is_available():
            self.skipTest("CUDA is not available")
        model = Sequential(
            Linear(784, 32),
            ReLU(),
            LayerNorm((32,)),
            Linear(32, 32),
            ReLU(),
            LayerNorm((32,)),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model, use_cuda=True)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_cnn_CUDA(self):
        if not pytagi.cuda.is_available():
            self.skipTest("CUDA is not available")
        model = Sequential(
            Conv2d(
                1, 8, 4, padding=1, stride=1, padding_type=1, in_width=28, in_height=28
            ),
            ReLU(),
            AvgPool2d(3, 2),
            Conv2d(8, 8, 5),
            ReLU(),
            AvgPool2d(3, 2),
            Linear(8 * 4 * 4, 32),
            ReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model, use_cuda=True)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_maxpooling_CUDA(self):
        model = Sequential(
            Conv2d(
                1, 8, 4, padding=1, stride=1, padding_type=1, in_width=28, in_height=28
            ),
            ReLU(),
            MaxPool2d(3, 2),
            Conv2d(8, 8, 5),
            ReLU(),
            MaxPool2d(3, 2),
            Linear(8 * 4 * 4, 32),
            ReLU(),
            Linear(32, 11),
        )
        avg_error_rate = mnist_test_runner(model, use_cuda=True)
        self.assertLess(
            avg_error_rate, self.threshold, "Error rate is higher than threshold"
        )


if __name__ == "__main__":
    unittest.main()
