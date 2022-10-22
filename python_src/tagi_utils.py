###############################################################################
# File:         tagi_utils.py
# Description:  Python frontend for TAGI utility functions
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 19, 2022
# Updated:      October 22, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from typing import Tuple
import numpy as np
import pytagi.UtilityWrapper as utils
from pytagi import HrSoftmax


class HierarchicalSoftmax(HrSoftmax):
    """Hierarchical softmax wrapper"""

    def __init__(self) -> None:
        super().__init__()


def get_hierarchial_softmax(
        labels: np.ndarray,
        num_classes: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Get observations and observation indices of the binary tree for
        classification"""

    obs, obs_idx, num_obs = utils.hierarchical_softmax(labels, num_classes)

    return obs, obs_idx, num_obs


def load_mnist_images(image_file: str, label_file: str,
                      num_images: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load mnist dataset"""

    images, labels = utils.load_mnist_dataset(image_file, label_file,
                                              num_images)

    return images, labels


def load_cifar_images(image_file: str,
                      num: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load cifar dataset"""

    images, labels = utils.load_cifar_dataset(image_file, num)

    return images, labels


def get_labels(ma: np.ndarray, Sa: np.ndarray, hr_softmax: HierarchicalSoftmax,
               num_classes: int,
               batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert last layer's hidden state to labels"""
    pred, prob = utils.get_labels(ma, Sa, hr_softmax, num_classes, batch_size)

    return pred, prob


def label_to_obs(num_classes: int) -> HierarchicalSoftmax:
    """Convert labels to binary tree"""
    hr_softmax = utils.class_to_label(num_classes)

    return hr_softmax


def obs_to_label_prob(ma: np.ndarray, Sa: np.ndarray,
                      hr_softmax: HierarchicalSoftmax,
                      num_classes: int) -> np.ndarray:
    """Convert observation to label probabilities"""

    prob = utils.obs_to_label_prob(ma, Sa, hr_softmax, num_classes)

    return prob
