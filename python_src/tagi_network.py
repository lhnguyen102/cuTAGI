###############################################################################
# File:         tagi_network.py
# Description:  Python frontend for TAGI network
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 13, 2022
# Updated:      October 29, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Tuple

import numpy as np
import pytagi as tagi


class NetProp(tagi.Network):
    """Base class for network properties"""

    def __init__(self) -> None:
        super().__init__()


class Param(tagi.Param):
    """Front-end apt for weight and biases"""

    def __init__(self, mw: np.ndarray, Sw: np.ndarray, mb: np.ndarray,
                 Sb: np.ndarray, mw_sc: np.ndarray, Sw_sc: np.ndarray,
                 mb_sc: np.ndarray, Sb_sc: np.ndarray) -> None:
        super().__init__()
        self.mw = mw
        self.Sw = Sw
        self.mb = mb
        self.Sb = Sb
        self.mw_sc = mw_sc
        self.Sw_sc = Sw_sc
        self.mb_sc = mb_sc
        self.Sb_sc = Sb_sc


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

        return np.array(ma), np.array(Sa)

    def set_parameters(self, param: Param) -> None:
        """Set parameter values to network"""
        self.network.set_parameters(param)

    def get_parameters(self) -> tagi.Param:
        """Get parameters of network"""
        return self.network.get_parameters()
