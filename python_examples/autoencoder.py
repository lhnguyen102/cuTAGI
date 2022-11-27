###############################################################################
# File:         autoencoder.py
# Description:  Example of autoencoder task using pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 30, 2022
# Updated:      November 12, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

from pytagi import NetProp, Param, TagiNetwork
from pytagi import Utils, exponential_scheduler
from visualizer import ImageViz


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
                 viz: Union[ImageViz, None] = None,
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

        # Inputs
        batch_size = self.encoder_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        # Outputs
        V_batch, empty_ud_idx_batch = self.init_outputs(batch_size)

        input_data, _, _, _ = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            # Decaying observation's variance
            self.decoder_prop.sigma_v = exponential_scheduler(
                curr_v=self.decoder_prop.sigma_v,
                min_v=self.decoder_prop.sigma_v_min,
                decaying_factor=self.decoder_prop.decay_factor_sigma_v,
                curr_iter=epoch)
            V_batch = V_batch * 0.0 + self.decoder_prop.sigma_v**2

            for i in range(num_iter):
                # Momentum for batch norm layer
                if (i == 0 and epoch == 0):
                    self.encoder.net_prop.ra_mt = 0.0
                    self.decoder.net_prop.ra_mt = 0.0
                else:
                    self.encoder.net_prop.ra_mt = 0.9
                    self.decoder.net_prop.ra_mt = 0.9

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
                                                 empty_ud_idx_batch)
                self.decoder.param_feed_backward()

                # Encoder's feed backward for states & parameters
                enc_delta_mz_init, enc_delta_vz_init = self.encoder.get_state_delta_mean_var(
                )

                # Encoder's feed backward for state & parameters
                self.encoder.state_feed_backward(enc_delta_mz_init,
                                                 enc_delta_vz_init,
                                                 empty_ud_idx_batch)
                self.encoder.param_feed_backward()

                # Progress bar
                pbar.set_description(
                    f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}"
                )

            self.predict()

    def predict(self) -> None:
        """Generate images"""
        # Inputs
        batch_size = self.encoder_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        generated_images = []
        for count, (x_batch, _) in enumerate(self.data_loader["test"]):
            # Disable average running for batch norm layer
            self.encoder.net_prop.ra_mt = 1.0
            self.decoder.net_prop.ra_mt = 1.0

            # Encoder's feed forward
            self.encoder.feed_forward(x_batch, Sx_batch, Sx_f_batch)

            # Decoder's feed forward
            ma, va, mz, vz, jcb = self.encoder.get_all_network_outputs()
            self.decoder.connected_feed_forward(ma=ma,
                                                va=va,
                                                mz=mz,
                                                vz=vz,
                                                jcb=jcb)

            # Get images
            norm_pred, _ = self.decoder.get_network_predictions()
            generated_images.append(norm_pred)

            # Only first 100 images
            if count * batch_size > 100:
                break

        generated_images = np.stack(generated_images).flatten()
        generated_images = generated_images[:self.encoder_prop.nodes[0] * 100]

        # Visualization
        if self.viz is not None:
            n_row = 10
            n_col = 10
            self.viz.plot_images(n_row=n_row,
                                 n_col=n_col,
                                 imgs=generated_images)

    def init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for inputs"""
        Sx_batch = np.zeros((batch_size, self.encoder_prop.nodes[0]),
                            dtype=self.dtype)

        Sx_f_batch = np.array([], dtype=self.dtype)

        return Sx_batch, Sx_f_batch

    def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for outputs"""
        # Outputs
        V_batch = np.zeros((batch_size, self.decoder_prop.nodes[-1]),
                           dtype=self.dtype) + self.decoder_prop.sigma_v**2
        ud_idx_batch = np.zeros((batch_size, 0), dtype=np.int32)

        return V_batch, ud_idx_batch
