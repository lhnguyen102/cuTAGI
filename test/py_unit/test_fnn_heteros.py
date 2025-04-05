import os
import sys
import unittest

import numpy as np

import pytagi
import pytagi.metric as metric
from examples.data_loader import RegressionDataLoader
from pytagi import Normalizer
from pytagi.nn import EvenExp, Linear, OutputUpdater, ReLU, Sequential

# path to binding code
sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)

TEST_CPU_ONLY = os.getenv("TEST_CPU_ONLY") == "1"
np.random.seed(0)


def heteros_test_runner(
    model: Sequential,
    batch_size: int = 16,
    num_iters: int = 200,
    use_cuda: bool = False,
) -> float:
    """Run training for the regression"""
    # Dataset
    x_train_file = "./data/toy_example/x_train_noise.csv"
    y_train_file = "./data/toy_example/y_train_noise.csv"

    train_dtl = RegressionDataLoader(x_file=x_train_file, y_file=y_train_file)

    model.to_device("cuda" if use_cuda else "cpu")
    out_updater = OutputUpdater(model.device)

    # -------------------------------------------------------------------------#
    # Training
    mses = []
    for _ in np.arange(2):
        batch_iter = train_dtl.create_data_loader(batch_size)

        for x, y in batch_iter:
            # Feed forward
            m_pred, _ = model(x)

            # Update output layer
            out_updater.update_heteros(
                output_states=model.output_z_buffer,
                mu_obs=y,
                delta_states=model.input_delta_z_buffer,
            )

            # Feed backward
            model.backward()
            model.step()

            # Training metric
            pred = Normalizer.unstandardize(
                m_pred, train_dtl.y_mean, train_dtl.y_std
            )
            obs = Normalizer.unstandardize(y, train_dtl.y_mean, train_dtl.y_std)

            # Even positions correspond to Z_out
            pred = pred[::2]

            mse = metric.mse(pred, obs)
            mses.append(mse)

    return sum(mses) / len(mses)


class SineSignalHeterosTest(unittest.TestCase):

    def setUp(self):
        self.threshold = 0.3

    def test_heteros_CPU(self):
        model = Sequential(
            Linear(1, 32),
            ReLU(),
            Linear(32, 32),
            ReLU(),
            Linear(32, 2),
            EvenExp(),
        )
        mse = heteros_test_runner(model)
        self.assertLess(
            mse, self.threshold, "Error rate is higher than threshold"
        )

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_heteros_CUDA(self):
        if not pytagi.cuda.is_available():
            self.skipTest("CUDA is not available")
        model = Sequential(
            Linear(1, 32),
            ReLU(),
            Linear(32, 32),
            ReLU(),
            Linear(32, 2),
            EvenExp(),
        )
        mse = heteros_test_runner(model, use_cuda=True)
        self.assertLess(
            mse, self.threshold, "Error rate is higher than threshold"
        )


if __name__ == "__main__":
    unittest.main()
