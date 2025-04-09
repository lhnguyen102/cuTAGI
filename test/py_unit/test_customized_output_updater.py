import os
import sys
import unittest
from typing import Tuple

import numpy as np

import pytagi
import pytagi.metric as metric
from examples.data_loader import TimeSeriesDataloader
from pytagi import Normalizer as normalizer
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential

# path to binding code
sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)

TEST_CPU_ONLY = os.getenv("TEST_CPU_ONLY") == "1"


def user_output_updater(
    mu_obs: float,
    var_obs: float,
    jcb: list,
    mu_output_z: list,
    var_output_z: list,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """User-defined output updater function"""

    jcb = np.array(jcb, dtype=np.float32).reshape((-1, batch_size))[0]
    mu_output_z = np.array(mu_output_z, dtype=np.float32).reshape(
        (-1, batch_size)
    )[0]
    var_output_z = np.array(var_output_z, dtype=np.float32).reshape(
        (-1, batch_size)
    )[0]

    tmp = jcb / (var_output_z + var_obs)
    delta_mu = tmp * (mu_obs - mu_output_z)
    delta_var = -tmp * jcb
    return (
        delta_mu,
        delta_var,
    )


def output_updater_test_runner(
    model: Sequential,
    input_seq_len: int = 4,
    batch_size: int = 8,
    use_cuda: bool = False,
) -> bool:
    """Run training for time-series forecasting model"""
    # Dataset
    output_col = [0]
    num_features = 1
    output_seq_len = 1
    seq_stride = 1

    train_dtl = TimeSeriesDataloader(
        x_file="data/toy_time_series/x_train_sin_data.csv",
        date_time_file="data/toy_time_series/train_sin_datetime.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
    )

    model.to_device("cuda" if use_cuda else "cpu")
    out_updater = OutputUpdater(model.device)

    # -------------------------------------------------------------------------#
    # Training
    var_y = np.full((batch_size * len(output_col),), 0.02**2, dtype=np.float32)

    for _ in np.arange(1):
        same_delta_z = []
        batch_iter = train_dtl.create_data_loader(batch_size, False)
        for x, y in batch_iter:
            # Feed forward
            m_pred, _ = model(x)

            # Default update output layer
            out_updater.update(
                output_states=model.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=model.input_delta_z_buffer,
            )
            delta_z_mu = model.input_delta_z_buffer.delta_mu
            delta_z_var = model.input_delta_z_buffer.delta_var

            # User-defined update output layer
            (
                user_delta_z_mu,
                user_delta_z_var,
            ) = user_output_updater(
                mu_obs=y,
                var_obs=var_y,
                jcb=model.output_z_buffer.jcb,
                mu_output_z=model.output_z_buffer.mu_a,
                var_output_z=model.output_z_buffer.var_a,
                batch_size=batch_size,
            )

            # Compare default and user-defined update output layers
            mu_match = np.allclose(delta_z_mu, user_delta_z_mu, atol=1e-6)
            var_match = np.allclose(delta_z_var, user_delta_z_var, atol=1e-6)
            same_delta_z.append(mu_match and var_match)

        same_delta_z = all(same_delta_z)

    return same_delta_z


def lstm_test_runner(
    model: Sequential,
    input_seq_len: int = 4,
    batch_size: int = 8,
    use_cuda: bool = False,
) -> float:
    """Run training for time-series forecasting model"""
    # Dataset
    output_col = [0]
    num_features = 1
    output_seq_len = 1
    seq_stride = 1

    train_dtl = TimeSeriesDataloader(
        x_file="data/toy_time_series/x_train_sin_data.csv",
        date_time_file="data/toy_time_series/train_sin_datetime.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
    )

    model.to_device("cuda" if use_cuda else "cpu")

    # -------------------------------------------------------------------------#
    # Training
    var_y = np.full((batch_size * len(output_col),), 0.02**2, dtype=np.float32)

    for _ in np.arange(2):
        mses = []
        batch_iter = train_dtl.create_data_loader(batch_size, False)
        for x, y in batch_iter:
            # Feed forward
            m_pred, _ = model(x)

            (
                model.input_delta_z_buffer.delta_mu,
                model.input_delta_z_buffer.delta_var,
            ) = user_output_updater(
                mu_obs=y,
                var_obs=var_y,
                jcb=model.output_z_buffer.jcb,
                mu_output_z=model.output_z_buffer.mu_a,
                var_output_z=model.output_z_buffer.var_a,
                batch_size=batch_size,
            )
            if model.device == "cuda":
                model.delta_z_to_device()

            # Feed backward
            model.backward()
            model.step()

            # Training metric
            pred = normalizer.unstandardize(
                m_pred,
                train_dtl.x_mean[output_col],
                train_dtl.x_std[output_col],
            )
            obs = normalizer.unstandardize(
                y, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            mse = metric.mse(pred, obs)
            mses.append(mse)

    return sum(mses) / len(mses)


class SineSignalTest(unittest.TestCase):

    def setUp(self):
        self.threshold = 0.5

    def test_lstm_CPU(self):
        input_seq_len = 4
        model = Sequential(
            LSTM(1, 8, input_seq_len),
            LSTM(8, 8, input_seq_len),
            Linear(8 * input_seq_len, 1),
        )
        mse = lstm_test_runner(model, input_seq_len=input_seq_len)
        same_delta_z = output_updater_test_runner(
            model, input_seq_len=input_seq_len
        )
        self.assertLess(
            mse, self.threshold, "Error rate is higher than threshold"
        )
        assert (
            same_delta_z
        ), "Different results for default and user-defined output_updater"

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_lstm_CUDA(self):
        if not pytagi.cuda.is_available():
            self.skipTest("CUDA is not available")
        input_seq_len = 4
        model = Sequential(
            LSTM(1, 8, input_seq_len),
            LSTM(8, 8, input_seq_len),
            Linear(8 * input_seq_len, 1),
        )
        mse = lstm_test_runner(
            model, input_seq_len=input_seq_len, use_cuda=True
        )
        same_delta_z = output_updater_test_runner(
            model, input_seq_len=input_seq_len
        )
        self.assertLess(
            mse, self.threshold, "Error rate is higher than threshold"
        )
        assert (
            same_delta_z
        ), "Different results for default and user-defined output_updater"


if __name__ == "__main__":
    unittest.main()
