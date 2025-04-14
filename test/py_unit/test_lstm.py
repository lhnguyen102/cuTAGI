import os
import sys
import unittest
from typing import Tuple

import numpy as np

import pytagi
import pytagi.metric as metric
from examples.data_loader import TimeSeriesDataloader
from pytagi import Normalizer as normalizer
from pytagi.nn import LSTM, SLSTM, Linear, OutputUpdater, Sequential, SLinear

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


def delta_z_to_device_test_runner(
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
    same_delta_z = False

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

    for _ in np.arange(1):

        batch_iter = train_dtl.create_data_loader(batch_size, False)
        x, y = next(batch_iter)

        # Feed forward
        m_pred, _ = model(x)

        # Test delta_z_to_device function:
        delta_mu = np.array([1, 2], dtype=np.float32)
        delta_var = np.array([3, 4], dtype=np.float32)
        model.delta_z_to_device(delta_mu, delta_var)
        mu_match = np.allclose(delta_mu, model.input_delta_z_buffer.delta_mu)
        var_match = np.allclose(delta_var, model.input_delta_z_buffer.delta_var)
        if mu_match and var_match:
            same_delta_z = True

    return same_delta_z


def lstm_user_output_updater_test_runner(
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

            delta_mu, delta_var = user_output_updater(
                mu_obs=y,
                var_obs=var_y,
                jcb=model.output_z_buffer.jcb,
                mu_output_z=model.output_z_buffer.mu_a,
                var_output_z=model.output_z_buffer.var_a,
                batch_size=batch_size,
            )
            model.delta_z_to_device(delta_mu, delta_var)

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
    out_updater = OutputUpdater(model.device)

    # -------------------------------------------------------------------------#
    # Training
    var_y = np.full((batch_size * len(output_col),), 0.02**2, dtype=np.float32)

    for _ in np.arange(2):
        mses = []
        batch_iter = train_dtl.create_data_loader(batch_size, False)
        for x, y in batch_iter:
            # Feed forward
            m_pred, _ = model(x)

            # Update output layer
            out_updater.update(
                output_states=model.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=model.input_delta_z_buffer,
            )

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


def replace_with_prediction(x, mu_sequence):
    nan_indices = np.where(np.isnan(x))[0]
    x[nan_indices] = mu_sequence[nan_indices]
    return x


def smoother_test_runner(
    model: Sequential,
    input_seq_len: int = 24,
    batch_size: int = 1,
    num_features: int = 3,
    use_cuda: bool = False,
) -> float:
    """Run training for time-series smoothing model"""
    # Dataset
    output_col = [0]
    output_seq_len = 1
    seq_stride = 1

    # Number of observations before training time to be inferred. These
    # obervations are nan in training data.
    infer_window_len = 48

    train_dtl = TimeSeriesDataloader(
        x_file="data/toy_time_series_smoother/x_train_sin_smoother.csv",
        date_time_file="data/toy_time_series_smoother/x_train_sin_smoother_datetime.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["hour_of_day", "day_of_week"],
        keep_last_time_cov=True,
    )

    model.to_device("cuda" if use_cuda else "cpu")
    model.input_state_update = True
    model.num_samples = train_dtl.dataset["value"][0].shape[0]
    out_updater = OutputUpdater(model.device)

    # -------------------------------------------------------------------------#
    # Training
    mses = []
    # Initialize the sequence length
    mu_sequence = np.ones(input_seq_len, dtype=np.float32)
    var_y = np.full((batch_size * len(output_col),), 0.6**2, dtype=np.float32)

    for _ in np.arange(2):
        batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)
        y_train = []
        # for x, y in batch_iter:
        for idx_sample, (x, y) in enumerate(batch_iter):

            # replace nan in input x by the lstm_prediction:
            if idx_sample < input_seq_len + infer_window_len:
                x = replace_with_prediction(x, mu_sequence)

            # Feed forward
            m_pred, _ = model(x)

            # Update output layer
            out_updater.update(
                output_states=model.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=model.input_delta_z_buffer,
            )
            # Feed backward
            model.backward()
            model.step()

            # Training metric
            pred = normalizer.unstandardize(
                m_pred,
                train_dtl.x_mean[output_col],
                train_dtl.x_std[output_col],
            )
            y_train.append(y)
            obs = normalizer.unstandardize(
                y, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            if np.isnan(obs).any():
                continue
            mse = metric.mse(pred, obs)
            mses.append(mse)

            # Add new prediction to mu_sequence
            mu_sequence = np.append(mu_sequence, m_pred)
            mu_sequence = mu_sequence[-input_seq_len:]

        # Smoother
        mu_zo_smooth, _ = model.smoother()
        mu_sequence = mu_zo_smooth[:input_seq_len]

    return np.nansum(mses) / np.sum(~np.isnan(mses))


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
        self.assertLess(
            mse, self.threshold, "Error rate is higher than threshold"
        )

    def test_smoother_CPU(self):
        input_seq_len = 24
        num_features = 3
        model = Sequential(
            SLSTM(num_features + input_seq_len - 1, 8, 1),
            SLSTM(8, 8, 1),
            SLinear(8, 1),
        )
        mse = smoother_test_runner(
            model, input_seq_len=input_seq_len, num_features=num_features
        )
        self.assertLess(
            mse, self.threshold, "Error rate is higher than threshold"
        )

    def test_delta_z_to_device_CPU(self):
        input_seq_len = 4
        model = Sequential(
            LSTM(1, 8, input_seq_len),
            LSTM(8, 8, input_seq_len),
            Linear(8 * input_seq_len, 1),
        )
        same_delta_z = delta_z_to_device_test_runner(
            model, input_seq_len=input_seq_len
        )
        assert same_delta_z, "Delta_z is not sent correctlt to CPU"

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
        self.assertLess(
            mse, self.threshold, "Error rate is higher than threshold"
        )

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_lstm_user_output_updater_CUDA(self):
        if not pytagi.cuda.is_available():
            self.skipTest("CUDA is not available")
        input_seq_len = 4
        model = Sequential(
            LSTM(1, 8, input_seq_len),
            LSTM(8, 8, input_seq_len),
            Linear(8 * input_seq_len, 1),
        )
        mse = lstm_user_output_updater_test_runner(
            model, input_seq_len=input_seq_len, use_cuda=True
        )
        self.assertLess(
            mse, self.threshold, "Error rate is higher than threshold"
        )

    @unittest.skipIf(TEST_CPU_ONLY, "Skipping CUDA tests due to --cpu flag")
    def test_delta_z_to_device_CUDA(self):
        if not pytagi.cuda.is_available():
            self.skipTest("CUDA is not available")
        input_seq_len = 4
        model = Sequential(
            LSTM(1, 8, input_seq_len),
            LSTM(8, 8, input_seq_len),
            Linear(8 * input_seq_len, 1),
        )
        same_delta_z = delta_z_to_device_test_runner(
            model, input_seq_len=input_seq_len, use_cuda=True
        )
        assert same_delta_z, "Delta_z is not sent correctlt to cuda device"


if __name__ == "__main__":
    unittest.main()
