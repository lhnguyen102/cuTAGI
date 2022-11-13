###############################################################################
# File:         classification.py
# Description:  Example of classification task using pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 19, 2022
# Updated:      November 12, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Tuple

import numpy as np
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import NetProp, TagiNetwork
from pytagi import HierarchicalSoftmax, Utils


class Classifier:
    """Classifier images"""

    hr_softmax: HierarchicalSoftmax
    utils: Utils = Utils()

    def __init__(self, num_epochs: int, data_loader: dict, net_prop: NetProp,
                 num_classes: int) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.net_prop = net_prop
        self.num_classes = num_classes
        self.network = TagiNetwork(self.net_prop)

    @property
    def num_classes(self) -> int:
        """Get number of classes"""

        return self._num_classes

    @num_classes.setter
    def num_classes(self, value: int) -> None:
        """Set number of classes"""
        self._num_classes = value
        self.hr_softmax = self.utils.get_hierarchical_softmax(
            self._num_classes)
        self.net_prop.nye = self.hr_softmax.num_obs

    def train(self) -> None:
        """Train the network using TAGI"""

        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        # Outputs
        V_batch, _ = self.init_outputs(batch_size)

        input_data, output_data, output_idx, labels = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))
        error_rates = []
        for epoch in pbar:
            for i in range(num_iter):
                # Get data
                idx = np.random.choice(num_data, size=batch_size)
                x_batch = input_data[idx, :]
                y_batch = output_data[idx, :]
                ud_idx_batch = output_idx[idx, :]
                label = labels[idx]

                # Feed forward
                self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)

                # Update hidden states
                self.network.state_feed_backward(y_batch, V_batch,
                                                 ud_idx_batch)

                # Update parameters
                self.network.param_feed_backward()

                # Error rate
                ma_pred, Sa_pred = self.network.get_network_outputs()
                pred, _ = self.utils.get_labels(ma=ma_pred,
                                                Sa=Sa_pred,
                                                hr_softmax=self.hr_softmax,
                                                num_classes=self.num_classes,
                                                batch_size=batch_size)

                error_rate = metric.classification_error(prediction=pred,
                                                         label=label)
                error_rates.append(error_rate)
                if i % 1000 == 0 and i > 0:
                    extracted_error_rate = np.hstack(error_rates)
                    avg_error_rate = np.mean(extracted_error_rate[-100:])
                    pbar.set_description(
                        f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}\t Error rate: {avg_error_rate * 100:>7.2f}%"
                    )

            # Validate on test set after each epoch
            self.predict()

    def predict(self) -> None:
        """Make prediction using TAGI"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        preds = []
        labels = []
        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            ma, Sa = self.network.get_network_outputs()
            pred, _ = self.utils.get_labels(ma=ma,
                                            Sa=Sa,
                                            hr_softmax=self.hr_softmax,
                                            num_classes=self.num_classes,
                                            batch_size=batch_size)

            # Store data
            preds.append(pred)
            labels.append(y_batch)

        preds = np.stack(preds).flatten()
        labels = np.stack(labels).flatten()

        # Compute classification error rate
        error_rate = metric.classification_error(prediction=preds,
                                                 label=labels)

        print("#############")
        print(f"Error rate    : {error_rate * 100: 0.2f}%")

    def init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for inputs"""
        Sx_batch = np.zeros((batch_size, self.net_prop.nodes[0]),
                            dtype=np.float32)

        Sx_f_batch = np.array([], dtype=np.float32)

        return Sx_batch, Sx_f_batch

    def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for outputs"""
        # Outputs
        V_batch = np.zeros((batch_size, self.net_prop.nodes[-1]),
                           dtype=np.float32) + self.net_prop.sigma_v**2
        ud_idx_batch = np.zeros((batch_size, 0), dtype=np.int32)

        return V_batch, ud_idx_batch
