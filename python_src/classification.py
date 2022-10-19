###############################################################################
# File:         classification.py
# Description:  Example of classification task using pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 19, 2022
# Updated:      October 19, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Union

import numpy as np
from tqdm import tqdm

import python_src.metric as metric
from python_src.data_loader import Normalizer as normalizer
from python_src.model import NetProp
from python_src.tagi_network import TagiNetwork


class Classifier:
    """Classifier images"""

    def __init__(self, num_epochs: int, data_loader: dict,
                 net_prop: NetProp) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.net_prop = net_prop
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

        input_data, output_data, output_idx = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            for i in range(num_iter):
                # Get data
                idx = np.random.choice(num_data, size=batch_size)
                x_batch = input_data[idx, :]
                y_batch = output_data[idx, :]
                ud_idx_batch = output_idx[idx, :]

                # Feed forward
                self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)

                # Update hidden states
                self.network.state_feed_backward(y_batch, V_batch,
                                                 ud_idx_batch)

                # Update parameters
                self.network.param_feed_backward()

                # Loss
                norm_pred, _ = self.network.get_network_outputs()
                # pbar.set_description(
                #     f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}\t mse: {mse:>7.2f}"
                # )