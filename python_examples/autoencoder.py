###############################################################################
# File:         autoencoder.py
# Description:  Example of autoencoder task using pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 30, 2022
# Updated:      October 30, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Union

import numpy as np
import python_src.metric as metric
from python_src.tagi_network import NetProp, Param, TagiNetwork
from python_src.tagi_utils import Utils
from tqdm import tqdm
from visualizer import PredictionViz


class Autoencoder:
    """Autoencoder task using TAGI"""

    utils: Utils = Utils

    def __init__(self,
                 num_epochs: int,
                 data_loader: dict,
                 encoder_prop: NetProp,
                 decoder_prop: NetProp,
                 encoder_param: Union[Param, None] = None,
                 decoder_param: Union[Param, None] = None,
                 viz: Union[PredictionViz, None] = None,
                 dtype=np.float32) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader

        # Encoder network
        self.encoder_prop = encoder_prop
        self.encoder = TagiNetwork(self.encoder_prop)
        if encoder_param is not None:
            self.encoder.set_parameters(param=encoder_param)

        # Decoder network
        self.decoder_prop = decoder_prop
        self.decoder = TagiNetwork(self.decoder_prop)
        if decoder_param is not None:
            self.decoder.set_parameters(decoder_param)
        self.viz = viz
        self.dtype = dtype

    def train(self) -> None:
        """Train encoder and decoder"""
        # Initialziation
        assert self.encoder_prop.batch_size == self.decoder_prop.batch_size
        batch_size = self.encoder_prop.batch_size

        # Inputs
        Sx_batch = np.zeros((batch_size, self.encoder_prop.nodes[0]),
                            dtype=self.dtype)
        Sx_f_batch = np.array([], dtype=self.dtype)

        # Outputs
        V_batch = np.zeros((batch_size, self.decoder_prop.nodes[-1]),
                           dtype=self.dtype) + self.decoder_prop.sigma_v**2
        ud_idx_batch = np.zeros([], self.dtype)

        input_data, output_data, output_idx, labels = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))

        for epoch in pbar:
            if epoch > 0:
                self.decoder_prop.sigma_v = np.maximum(
                    self.decoder_prop.sigma_v_min, self.decoder_prop.sigma_v *
                    self.decoder_prop.decay_factor_sigma_v)
                V_batch = V_batch * 0.0 + self.decoder_prop.sigma_v**2

            for i in range(num_iter):
                # Get data
                idx = np.random.choice(num_data, size=batch_size)
                x_batch = input_data[idx, :]

                # Encoder's feed forward
                self.encoder.feed_forward(x_batch, Sx_batch, Sx_f_batch)

                # Decoder's feed forward
                ma, va, mz, vz, jcb = self.encoder.get_all_network_outputs()
                self.decoder.connected_feed_forward(ma=ma,
                                                    va=va,
                                                    mz=mz,
                                                    vz=vz,
                                                    jcb=jcb)

                # Decoder's feed backward for states & parameters
                self.decoder.state_feed_backward(x_batch, V_batch,
                                                 ud_idx_batch)
                self.decoder.param_feed_backward()

                # Encoder's feed backward for states & parameters
                # TODO: To be completed

    def predict(self) -> None:
        """Generate images"""
        raise NotImplementedError
