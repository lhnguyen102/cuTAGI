###############################################################################
# File:         tagi_utils.py
# Description:  Python frontend for TAGI utility functions
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 19, 2022
# Updated:      October 24, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Tuple
import numpy as np
from pytagi import UtilityWrapper
from pytagi import HrSoftmax


class HierarchicalSoftmax(HrSoftmax):
    """Hierarchical softmax wrapper"""

    def __init__(self) -> None:
        super().__init__()


class Utils:
    """Frontend for utility functions from C++/CUDA backend"""

    backend_utils = UtilityWrapper()

    def __init__(self) -> None:
        pass

    def label_to_obs(self, labels: np.ndarray,
                     num_classes: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Get observations and observation indices of the binary tree for
            classification"""

        obs, obs_idx, num_obs = self.backend_utils.label_to_obs_wrapper(
            labels, num_classes)

        return np.array(obs), np.array(obs_idx), np.array(num_obs)

    def load_mnist_images(self, image_file: str, label_file: str,
                          num_images: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load mnist dataset"""
        images, labels = self.backend_utils.load_mnist_dataset_wrapper(
            image_file, label_file, num_images)

        return np.array(images, dtype=np.float32), np.array(labels).reshape(
            (num_images, 1))

    def load_cifar_images(self, image_file: str,
                          num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load cifar dataset"""

        images, labels = self.backend_utils.load_cifar_dataset_wrapper(
            image_file, num)

        return np.array(images), np.array(labels)

    def get_labels(self, ma: np.ndarray, Sa: np.ndarray,
                   hr_softmax: HierarchicalSoftmax, num_classes: int,
                   batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert last layer's hidden state to labels"""

        pred, prob = self.backend_utils.get_labels_wrapper(
            ma, Sa, hr_softmax, num_classes, batch_size)

        return np.array(pred), np.array(prob)

    def get_errors(self, ma: np.ndarray, Sa: np.ndarray, labels: np.ndarray,
                   hr_softmax: HierarchicalSoftmax, num_classes: int,
                   batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert last layer's hidden state to labels"""

        pred, prob = self.backend_utils.get_error_wrapper(
            ma, Sa, labels, hr_softmax, num_classes, batch_size)

        return np.array(pred), np.array(prob)

    def get_hierarchical_softmax(self,
                                 num_classes: int) -> HierarchicalSoftmax:
        """Convert labels to binary tree"""
        hr_softmax = self.backend_utils.hierarchical_softmax_wrapper(
            num_classes)

        return hr_softmax

    def obs_to_label_prob(self, ma: np.ndarray, Sa: np.ndarray,
                          hr_softmax: HierarchicalSoftmax,
                          num_classes: int) -> np.ndarray:
        """Convert observation to label probabilities"""

        prob = self.backend_utils.obs_to_label_prob_wrapper(
            ma, Sa, hr_softmax, num_classes)

        return np.array(prob)

    def create_rolling_window(self, data: np.ndarray, output_col: np.ndarray,
                              input_seq_len: int, output_seq_len: int,
                              num_features: int,
                              stride: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create rolling window for time series data"""
        num_data = int((len(data) / num_features - input_seq_len -
                        output_seq_len) / stride + 1)

        input_data, output_data = self.backend_utils.create_rolling_window_wrapper(
            data.flatten(), output_col, input_seq_len, output_seq_len,
            num_features, stride)
        input_data = np.array(input_data).reshape((num_data, input_seq_len))
        output_data = np.array(output_data).reshape((num_data, output_seq_len))

        return input_data, output_data
