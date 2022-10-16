###############################################################################
# File:         regression.py
# Description:  Example of regression task using pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      October 16, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Union

import numpy as np
from tqdm import tqdm
from visualizer import PredictionViz

import python_src.metric as metric
from python_src.data_loader import Normalizer as normalizer
from python_src.model import NetProp
from python_src.tagi_network import TagiNetwork


class Regression:
    """Regression task using TAGI"""

    def __init__(self,
                 num_epochs: int,
                 data_loader: dict,
                 net_prop: NetProp,
                 viz: Union[PredictionViz, None] = None) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.net_prop = net_prop
        self.network = TagiNetwork(self.net_prop)
        self.viz = viz

    def train(self) -> None:
        """Train the network using TAGI"""
        batch_size = self.net_prop.batch_size
        # Inputs
        Sx_batch = np.zeros((batch_size, self.net_prop.nodes[0]),
                            dtype=np.float32)
        Sx_f_batch = np.array([], dtype=np.float32)

        # Outputs
        V_batch = np.zeros((batch_size, self.net_prop.nodes[-1]),
                           dtype=np.float32) + self.net_prop.sigma_v**2
        ud_idx_batch = np.zeros((batch_size, 0), dtype=np.int32)

        input_data, output_data = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
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
                norm_pred, _ = self.network.get_network_outputs()
                pred = normalizer.unstandardize(
                    norm_data=norm_pred,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"])
                obs = normalizer.unstandardize(
                    norm_data=y_batch,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"])
                mse = metric.mse(pred, obs)
                pbar.set_description(
                    f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}\t mse: {mse:>7.2f}"
                )

    def predict(self) -> None:
        """Make prediction using TAGI"""
        batch_size = self.net_prop.batch_size
        # Inputs
        Sx_batch = np.zeros((batch_size, self.net_prop.nodes[0]),
                            dtype=np.float32)
        Sx_f_batch = np.array([], dtype=np.float32)

        mean_predictions = []
        variance_predictions = []
        y_test = []
        x_test = []
        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            ma, Sa = self.network.get_network_outputs()

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
                x_test=x_test,
                y_test=y_test,
                y_pred=mean_predictions,
                sy_pred=std_predictions,
                std_factor=3,
                label="diag",
                title=r"\textbf{Diagonal covariance}",
            )

        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")
