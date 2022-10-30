###############################################################################
# File:         tagi_utils.py
# Description:  Python frontend for TAGI utility functions
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 19, 2022
# Updated:      October 30, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Tuple, Union

import numpy as np
import pandas as pd
from pytagi import HrSoftmax, UtilityWrapper

from python_src.tagi_network import Param


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


class Normalizer:
    """Different method to normalize the data before feeding
    to neural networks"""

    def __init__(self, method: Union[str, None] = None) -> None:
        self.method = method

    def standardize(self, data: np.ndarray, mu: np.ndarray,
                    std: np.ndarray) -> np.ndarray:
        """Z-score normalization where 
        data_norm = (data - data_mean) / data_std """

        return (data - mu) / (std + 1e-10)

    @staticmethod
    def unstandardize(norm_data: np.ndarray, mu: np.ndarray,
                      std: np.ndarray) -> np.ndarray:
        """Transform standardized data to original space"""
        return norm_data * (std + 1e-10) + mu

    @staticmethod
    def unstandardize_std(norm_std: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Transform standardized std to original space"""

        return norm_std * (std + 1e-10)

    def max_min_norm(self, data: np.ndarray, max_value: np.ndarray,
                     min_value: np.ndarray) -> np.ndarray:
        """Normalize the data between 0 and 1"""
        assert np.all(max_value > min_value)
        return (data - min_value) / (max_value - min_value + 1e-10)

    @staticmethod
    def max_min_unnorm(norm_data: np.ndarray, max_value: np.ndarray,
                       min_value: np.ndarray) -> np.ndarray:
        """Transform max-min normalized data to original space"""

        return (norm_data * (max_value - min_value + 1e-10)) + min_value

    @staticmethod
    def max_min_unnorm_std(norm_std: np.ndarray, max_value: np.ndarray,
                           min_value: np.ndarray) -> np.ndarray:
        """Transform max-min normalized std to original space"""

        return (norm_std * (max_value - min_value + 1e-10))

    @staticmethod
    def compute_mean_std(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sample mean and standard deviation"""

        return (np.nanmean(data, axis=0), np.nanstd(data, axis=0))

    @staticmethod
    def compute_max_min(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute max min values"""

        return (np.nanmax(data, axis=0), np.nanmin(data, axis=0))


def load_param_from_files(mw_file: str, Sw_file: str, mb_file: str,
                          Sb_file: str, mw_sc_file: str, Sw_sc_file: str,
                          mb_sc_file: str, Sb_sc_file: str) -> Param:
    """Load parameter from csv file"""
    mw_df = pd.read_csv(mw_file, header=None)
    Sw_df = pd.read_csv(Sw_file, header=None)
    mb_df = pd.read_csv(mb_file, header=None)
    Sb_df = pd.read_csv(Sb_file, header=None)

    try:
        mw_sc_df = pd.read_csv(mw_sc_file, header=None)
        Sw_sc_df = pd.read_csv(Sw_sc_file, header=None)
        mb_sc_df = pd.read_csv(mb_sc_file, header=None)
        Sb_sc_df = pd.read_csv(Sb_sc_file, header=None)
    except ValueError:
        mw_sc_df = pd.DataFrame()
        Sw_sc_df = pd.DataFrame()
        mb_sc_df = pd.DataFrame()
        Sb_sc_df = pd.DataFrame()

    return Param(mw=mw_df.values,
                 Sw=Sw_df.values,
                 mb=mb_df.values,
                 Sb=Sb_df.values,
                 mw_sc=mw_sc_df.values,
                 Sw_sc=Sw_sc_df.values,
                 mb_sc=mb_sc_df.values,
                 Sb_sc=Sb_sc_df.values)
