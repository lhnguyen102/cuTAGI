###############################################################################
# File:         tagi_network.py
# Description:  Python frontend for TAGI network
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 13, 2022
# Updated:      November 12, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Tuple

import numpy as np
import cutagi as tagi


class NetProp(tagi.Network):
    """Base class for network properties defined in the backend C++/CUDA
    Layer code:
        1: Fully-connected layer
        2: Convolutional layer
        21: Transpose convolutional layer
        3: Max pooling layer (currently not supported)
        4: Average pooling
        5: Layer normalization
        6: Batch normalization
        7: LSTM layer

    Activation code:
        0: No activation 
        1: Tanh
        2: Sigmoid
        4: ReLU
        5: Softplus
        6: Leakyrelu

    Attributes:
        layers: A vector contains different layers of network architecture
        nodes: Number of hidden units
        kernels: Kernel size fo convolutional layer
        widths: Width of image
        heights: Heights of image
        filters: Number of filters i.e. depth of image for each layer
        activations: Activation function
        pads: Padding that applied to image
        pad_types: Type of padding
        shortcuts: Layer index for residual network 
        mu_v2b: Mean of the observation noise squared
        sigma_v2b: Standard deviation of the observation noise squared
        sigma_v: Observation noise
        decay_factor_sigma_v: Decaying factor for sigma v (default value: 0.99)
        sigma_v_min: Minimum value of observation noise (default value: 0.3)
        sigma_x: Input noise noise
        is_output_ud: Whether or not to update output layer
        is_idx_ud: Wheher or not to update only hidden units in the output
                   layers
        last_backward_layer: Index of last layer whose hidden states are updated
        nye: Number of observation for hierarchical softmax
        noise_gain : Gain fof biases parameters relating to noise's hidden
            states
        noise_type: homosce or heteros
        batch_size: Number of batches of data
        input_seq_len: Sequence lenth for lstm inputs
        input_seq_len: Sequence lenth for last layer's outputs
        seq_stride: Spacing between sequences for lstm layer
        multithreading: Whether or not to run parallel computing using multiple
            threads
        collect_derivative: Enable the derivative computation mode
        is_full_cov: Enable full covariance mode
        init_method: Initalization method e.g. He and Xavier
        device: Either cpu or cuda
        ra_mt: Momentum for the normalization layer
    """
    layers: list
    nodes: list
    kernels: list
    strides: list
    widths: list
    heights: list
    filters: list
    activation: list
    pads: list
    pad_types: list
    shortcuts: list
    mu_v2b: np.ndarray
    sigma_v2b: np.ndarray
    sigma_v: float
    decay_factor_sigma_v: float
    sigma_v_min: float
    sigma_x: float
    is_idx_ud: bool
    is_output_ud: bool
    last_backward_layer: int
    nye: int
    noise_gain: float
    noise_type: str
    batch_size: int
    input_seq_len: int
    output_seq_len: int
    seq_stride: int
    multithreading: bool
    collect_derivative: bool
    is_full_cov: bool
    init_method: str
    device: str
    ra_mt: float

    def __init__(self) -> None:
        super().__init__()


class Param(tagi.Param):
    """Frontend apt for weight and biases

    Attributes:
        mw: Mean of weight parameters
        Sw: Variance of weight parameters
        mb: Mean of bias parameters
        Sb: Variance of bias parameters
        mw_sc: Mean of weight parameters for the residual network
        Sw_sc: Variance of weight parameters for the residual network
        mb_sc: Mean of bias parameters for the residual network
        Sb_sc: Variance of bias parameters for the residual network
    """
    mw: np.ndarray
    Sw: np.ndarray
    mb: np.ndarray
    Sb: np.ndarray
    mw_sc: np.ndarray
    Sw_sc: np.ndarray
    mb_sc: np.ndarray
    Sb_sc: np.ndarray

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
    """Python frontend calling TAGI network in C++/CUDA backend

    Attributes:
        network: Network wrapper that calls the tagi network from
         the backend 
        net_prop: Network properties
    """

    network: tagi.NetworkWrapper

    def __init__(self, net_prop: NetProp) -> None:
        self.net_prop = net_prop

    @property
    def net_prop(self) -> NetProp():
        """"Get network properties"""
        return self._net_prop

    @net_prop.setter
    def net_prop(self, value: NetProp) -> None:
        """Set network properties"""
        self._net_prop = value
        self.network = tagi.NetworkWrapper(self._net_prop)

    def feed_forward(self, x_batch: np.ndarray, Sx_batch: np.ndarray,
                     Sx_f_batch: np.ndarray) -> None:
        """Forward pass
        the size of x_batch, Sx_batch (B, N) where B is the batch size and N
        is the data dimension

        Args:
            x_batch: Input data
            Sx_batch: Diagonal variance of input data
            Sx_f_batch: Full variance of input data
        """
        self.network.feed_forward_wrapper(x_batch.flatten(),
                                          Sx_batch.flatten(),
                                          Sx_f_batch.flatten())

    def connected_feed_forward(self, ma: np.ndarray, va: np.ndarray,
                               mz: np.ndarray, vz: np.ndarray,
                               jcb: np.ndarray) -> None:
        """Forward pass for the network that is connected to the other 
        network e.g. decoder network in autoencoder task where its inputs 
        are the output of the encoder network.

        Args:
            ma: Mean of activation units
            va: Variance of activation units
            mz: Mean of hidden states
            vz: Variance of hidden states
            jcb: Jacobian matrix (da/dz)
        """

        self.network.connected_feed_forward_wrapper(ma, va, mz, vz, jcb)

    def state_feed_backward(self, y_batch: np.ndarray, v_batch: np.ndarray,
                            ud_idx_batch: np.ndarray) -> None:
        """Update hidden states
        the size of y_batch, V_batch (B, N) where B is the batch size and N
        is the data dimension

        Args:
            y_batch: Observations 
            v_batch: Variance of observations
            ud_idx_batch: Updated indices for the last layer e.g., it 
                only required for classification task
        """
        self.network.state_feed_backward_wrapper(y_batch.flatten(),
                                                 v_batch.flatten(),
                                                 ud_idx_batch.flatten())

    def param_feed_backward(self) -> None:
        """Update parameters"""
        self.network.param_feed_backward_wrapper()

    def get_network_outputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get output layer's hidden state distribution
        
        Returns:
            ma: Mean of activation units
            va: Variance of activation units
        """
        ma, va = self.network.get_network_outputs_wrapper()

        return np.array(ma), np.array(va)

    def get_network_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get distribution of the predictions
        
        Returns:
            m_pred: Mean of predictions
            v_pred: Variance of predictions
        """
        m_pred, v_pred = self.network.get_network_prediction_wrapper()

        return np.array(m_pred), np.array(v_pred)

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

    def get_derivatives(self, layer: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """ Compute derivatives of the output layer w.r.t a given layer using TAGI
        
        Args:
            layer: Layer index of the network
        Returns:
            mdy: Mean values of derivatives
            vdy: Variance values of derivatives
        """
        mdy, vdy = self.network.get_derivative_wrapper(layer)
        return mdy, vdy

    def get_inovation_mean_var(self,
                               layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get updating quantities for the inovation

        Args: 
            layer: Layer index of the network
        
        Returns:
            delta_m: Inovation mean 
            delta_v: Inovation variance 
        """

        delta_m, delta_v = self.network.get_inovation_mean_var_wrapper(layer)

        return np.array(delta_m), np.array(delta_v)

    def get_state_delta_mean_var(self) -> None:
        """Get updating quatities for the first layer
        
        Returns:
            delta_mz: Updating quantities for the hidden-state mean of the
                first layer
            delta_vz: Updating quantities for the hidden-state variance of the
                first layer
        """
        delta_mz, delta_vz = self.network.get_state_delta_mean_var_wrapper()

        return np.array(delta_mz), np.array(delta_vz)

    def set_parameters(self, param: Param) -> None:
        """Set parameter values to network"""
        self.network.set_parameters_wrapper(param)

    def get_parameters(self) -> tagi.Param:
        """Get parameters of network"""
        return self.network.get_parameters_wrapper()
