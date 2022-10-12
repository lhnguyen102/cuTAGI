###############################################################################
# File:         regression.py
# Description:  Example of regression task using pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      October 12, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
import pytagi
from python_src.model import RegressionMLP
import numpy as np


class Regression:
    """Regression task using TAGI"""

    net_prop: RegressionMLP = RegressionMLP()

    def __init__(self, num_epochs: int, data_loader: dict) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.network = pytagi.NetworkWrapperCPU(self.net_prop)

    def train(self) -> None:
        """Train the network using TAGI"""
        x_init, y_init = self.data_loader["train"][0]
        Sx_batch = np.zeros(x_init.shape, dtype=np.float32)
        Sx_f_batch = np.zeros(x_init.shape, dtype=np.float32)
        V_batch = np.zeros(y_init.shape,
                           dtype=np.float32) + self.net_prop.prop.sigma_v**2
        ud_idx_batch = np.zeros(y_init.shape, dtype=np.int32)

        for x_batch, y_batch in self.data_loader["train"]:
            # Feed forward
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)

            # State feed backward
            self.network.state_feed_backward(y_batch, V_batch, ud_idx_batch)

            # Update parameters
            self.network.param_feed_backward()

    def predict(self) -> None:
        """Make prediction using TAGI"""
        x_init, _ = self.data_loader["test"][0]
        Sx_batch = np.zeros(x_init.shape, dtype=np.float32)
        Sx_f_batch = np.zeros(x_init.shape, dtype=np.float32)

        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            ma, Sa = self.network.get_network_outputs()

            # Compute log-likelihood

            return ma, Sa
