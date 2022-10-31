###############################################################################
# File:         tagi_network.py
# Description:  Python frontend for TAGI network
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 13, 2022
# Updated:      October 30, 2022
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
        self.network.feed_forward_wrapper(x_batch.flatten(),
                                          Sx_batch.flatten(),
                                          Sx_f_batch.flatten())

    def connected_feed_forward(self, ma: np.ndarray, va: np.ndarray,
                               mz: np.ndarray, vz: np.ndarray,
                               jcb: np.ndarray) -> None:
        """Forward pass for the network that is connected to the other 
        network e.g. decoder network in autoencoder taks"""

        self.network.connected_feed_forward_wrapper(ma, va, mz, vz, jcb)

    def state_feed_backward(self, y_batch: np.ndarray, V_batch: np.ndarray,
                            ud_idx_batch: np.ndarray) -> None:
        """Update hidden states
        the size of y_batch, V_batch (B, N) where B is the batch size and N
        is the data dimension
        """
        self.network.state_feed_backward_wrapper(y_batch.flatten(),
                                                 V_batch.flatten(),
                                                 ud_idx_batch.flatten())

    def param_feed_backward(self) -> None:
        """Update parameters"""
        self.network.param_feed_backward_wrapper()

    def get_network_outputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get last layer's hidden state distribution"""
        ma, Sa = self.network.get_network_outputs_wrapper()

        return np.array(ma), np.array(Sa)

    def get_all_network_outputs(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get all hidden states of the output layers

        Returns:
            ma: Mean of activations for the output layer
            va: Variance of activations for the output layer
            mz: Mean of hidden states for the output layer
            vz: Variance of hidden states for the output layer       
            jcb: Jacobian matrix for the output layer
        """
        ma, va, mz, vz, jcb = self.network.get_all_network_outputs_wrapper()

        return (np.array(ma), np.array(va), np.array(mz), np.array(vz),
                np.array(jcb))

    def get_all_network_inputs(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get all hidden states of the output layers

        Returns:
            ma: Mean of activations for the input layer
            va: Variance of activations for the input layer
            mz: Mean of hidden states for the input layer
            vz: Variance of hidden states for the input layer     
            jcb: Jacobian matrix for the input layer
        """
        ma, va, mz, vz, jcb = self.network.get_all_network_inputs_wrapper()

        return (np.array(ma), np.array(va), np.array(mz), np.array(vz),
                np.array(jcb))

    def get_inovation_mean_var(self,
                               layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get updating quantities for the inovation"""

        delta_m, delta_v = self.network.get_inovation_mean_var_wrapper(layer)

        return np.array(delta_m), np.array(delta_v)

    def set_parameters(self, param: Param) -> None:
        """Set parameter values to network"""
        self.network.set_parameters_wrapper(param)

    def get_parameters(self) -> tagi.Param:
        """Get parameters of network"""
        return self.network.get_parameters_wrapper()
