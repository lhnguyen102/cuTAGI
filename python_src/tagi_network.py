###############################################################################
# File:         tagi_network.py
# Description:  Python frontend for TAGI network
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 13, 2022
# Updated:      October 15, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Tuple
import numpy as np
import pytagi as tagi


class TagiNetwork:
    """Python frontend calling TAGI network in C++/CUDA backend"""

    network: tagi.NetworkWrapper

    def __init__(self, net_prop: tagi.Network) -> None:
        self.net_prop = net_prop

    @property
    def net_prop(self) -> tagi.Network():
        """"Get network properties"""
        return self._net_prop

    @net_prop.setter
    def net_prop(self, value: tagi.Network) -> None:
        """Set network properties"""
        self._net_prop = value
        self.network = tagi.NetworkWrapper(self._net_prop)

    def feed_forward(self, x_batch: np.ndarray, Sx_batch: np.ndarray,
                     Sx_f_batch: np.ndarray) -> None:
        """Forward pass
        the size of x_batch, Sx_batch (B, N) where B is the batch size and N
        is the data dimension
        """
        self.network.feed_forward(x_batch.flatten(), Sx_batch.flatten(),
                                  Sx_f_batch.flatten())

    def state_feed_backward(self, y_batch: np.ndarray, V_batch: np.ndarray,
                            ud_idx_batch: np.ndarray) -> None:
        """Update hidden states
        the size of y_batch, V_batch (B, N) where B is the batch size and N
        is the data dimension
        """
        self.network.state_feed_backward(y_batch.flatten(), V_batch.flatten(),
                                         ud_idx_batch.flatten())

    def param_feed_backward(self) -> None:
        """Update parameters"""
        self.network.param_feed_backward()

    def get_network_outputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get last layer's hidden state distribution"""
        ma, Sa = self.network.get_network_outputs()

        return ma, Sa
