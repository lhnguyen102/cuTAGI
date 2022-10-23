###############################################################################
# File:         metric.py
# Description:  Measure the accuracy of the prediction
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 13, 2022
# Updated:      October 21, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
import numpy as np


def mse(prediction: np.ndarray, observation: np.ndarray) -> float:
    """ Mean squared error"""
    return np.nanmean((prediction - observation)**2)


def log_likelihood(prediction: np.ndarray, observation: np.ndarray,
                   std: np.ndarray) -> float:
    """Compute the averaged log-likelihood"""

    log_lik = -0.5 * np.log(2 * np.pi * (std**2)) - 0.5 * ((
        (observation - prediction) / std)**2)

    return np.nanmean(log_lik)


def rmse(prediction: np.ndarray, observation: np.ndarray) -> None:
    """Root mean squared error"""
    mse = mse(prediction, observation)

    return mse**0.5


def classification_error(prediction: np.ndarray, label: np.ndarray) -> None:
    """Compute the classification error"""
    count = 0
    for pred, lab in zip(prediction.T, label):
        if pred != lab:
            count += 1

    return count / len(prediction)
