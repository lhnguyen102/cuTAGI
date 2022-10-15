###############################################################################
# File:         regression.py
# Description:  Example of regression task using pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      October 13, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
import numpy as np
from tqdm import tqdm

import python_src.metric as metric
from python_src.model import RegressionMLP
from python_src.tagi_network import TagiNetwork


class Regression:
    """Regression task using TAGI"""

    net_prop: RegressionMLP = RegressionMLP()

    def __init__(self, num_epochs: int, data_loader: dict) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.network = TagiNetwork(self.net_prop)

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
        ud_idx_batch = np.array([], dtype=np.float32)

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
                pred, _ = self.network.get_network_outputs()
                mse = metric.mse(pred, y_batch)
                pbar.set_description(
                    f"Epoch# {epoch: 0} | {i * self.batch_size + len(x_batch): 1:>5}|{num_data: 2}\t mse: {mse: 0.2f}"
                )

    def predict(self) -> None:
        """Make prediction using TAGI"""
        x_init, _ = self.data_loader["test"][0]
        Sx_batch = np.zeros(x_init.shape, dtype=np.float32)
        Sx_f_batch = np.zeros(x_init.shape, dtype=np.float32)

        mean_predictions = []
        variance_predictions = []
        observations = []
        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            ma, Sa = self.network.get_network_outputs()

            mean_predictions.append(ma)
            variance_predictions.append(Sa)
            observations.append(y_batch)

        mean_predictions = np.stack(mean_predictions)
        std_predictions = np.stack(variance_predictions)**0.5
        observations = np.stack(observations)

        # Compute log-likelihood
        mse = metric.mse(mean_predictions, observations)
        log_lik = metric.log_likelihood(prediction=mean_predictions,
                                        observation=observations,
                                        std=std_predictions)

        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")
