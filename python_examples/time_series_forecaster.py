###############################################################################
# File:         time_series_forecaster.py
# Description:  Example of the time series forecasting
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 26, 2022
# Updated:      November 12, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Union, Tuple

import numpy as np
import pytagi.metric as metric
from pytagi import NetProp, Param, TagiNetwork
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler
from tqdm import tqdm
from visualizer import PredictionViz


class TimeSeriesForecaster:
    """Time series forecaster using TAGI"""

    def __init__(self,
                 num_epochs: int,
                 data_loader: dict,
                 net_prop: NetProp,
                 param: Union[Param, None] = None,
                 viz: Union[PredictionViz, None] = None,
                 dtype=np.float32) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.net_prop = net_prop
        self.network = TagiNetwork(self.net_prop)
        self.viz = viz
        self.dtype = dtype
        if param is not None:
            self.network.set_parameters(param=param)

    def train(self) -> None:
        """Train LSTM network"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        # Outputs
        V_batch, ud_idx_batch = self.init_outputs(batch_size)

        input_data, output_data = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            # Decaying observation's variance
            self.net_prop.sigma_v = exponential_scheduler(
                curr_v=self.net_prop.sigma_v,
                min_v=self.net_prop.sigma_v_min,
                decaying_factor=self.net_prop.decay_factor_sigma_v,
                curr_iter=epoch)
            V_batch = V_batch * 0.0 + self.net_prop.sigma_v**2

            for i in range(num_iter):
                # Get data
                idx = np.random.choice(num_data, size=batch_size)
                x_batch = input_data[idx, :]
                y_batch = output_data[idx, :]

                # Feed forward
                self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)

                # Update hidden states
                self.network.state_feed_backward(y_batch, V_batch,
                                                 ud_idx_batch)

                # Update parameters
                self.network.param_feed_backward()

                # Loss
                norm_pred, _ = self.network.get_network_predictions()
                pred = normalizer.unstandardize(
                    norm_data=norm_pred,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"])
                obs = normalizer.unstandardize(
                    norm_data=y_batch,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"])
                mse = metric.mse(pred, obs)

                # Progress bar
                pbar.set_description(
                    f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}\t mse: {mse:>7.2f}"
                )

    def predict(self) -> None:
        """Make prediction for time series using TAGI"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        mean_predictions = []
        variance_predictions = []
        y_test = []
        x_test = []
        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            ma, Sa = self.network.get_network_predictions()

            mean_predictions.append(ma)
            variance_predictions.append(Sa + self.net_prop.sigma_v**2)
            x_test.append(x_batch)
            y_test.append(y_batch)

        mean_predictions = np.stack(mean_predictions).flatten()
        std_predictions = (np.stack(variance_predictions).flatten())**0.5
        y_test = np.stack(y_test).flatten()
        x_test = np.stack(x_test).flatten()

        mean_predictions = normalizer.unstandardize(
            norm_data=mean_predictions,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"])

        std_predictions = normalizer.unstandardize_std(
            norm_std=std_predictions, std=self.data_loader["y_norm_param_2"])

        y_test = normalizer.unstandardize(
            norm_data=y_test,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"])

        # Compute log-likelihood
        mse = metric.mse(mean_predictions, y_test)
        log_lik = metric.log_likelihood(prediction=mean_predictions,
                                        observation=y_test,
                                        std=std_predictions)

        # Visualization
        if self.viz is not None:
            self.viz.plot_predictions(
                x_train=None,
                y_train=None,
                x_test=self.data_loader["datetime_test"][:len(y_test)],
                y_test=y_test,
                y_pred=mean_predictions,
                sy_pred=std_predictions,
                std_factor=1,
                label="time_series_forecasting",
                title=r"\textbf{Time Series Forecasting}",
                time_series=True)

        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")

    def init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for inputs"""
        Sx_batch = np.zeros(
            (batch_size * self.net_prop.input_seq_len, self.net_prop.nodes[0]),
            dtype=self.dtype)

        Sx_f_batch = np.array([], dtype=self.dtype)

        return Sx_batch, Sx_f_batch

    def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for outputs"""
        # Outputs
        V_batch = np.zeros((batch_size, self.net_prop.nodes[-1]),
                           dtype=self.dtype) + self.net_prop.sigma_v**2
        ud_idx_batch = np.zeros((batch_size, 0), dtype=np.int32)

        return V_batch, ud_idx_batch
