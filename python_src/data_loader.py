###############################################################################
# File:         dataloader.py
# Description:  Prepare data for neural networks
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      October 21, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
import math
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import pandas as pd

import python_src.tagi_utils as utils


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


class DataloaderBase(ABC):
    """Dataloader template"""

    normalizer: Normalizer = Normalizer()

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    @abstractmethod
    def process_data(self) -> dict:

        raise NotImplementedError

    def create_data_loader(self, raw_input: np.ndarray,
                           raw_output: np.ndarray) -> list:
        """Create dataloader based on batch size"""
        num_input_data = raw_input.shape[0]
        num_output_data = raw_output.shape[0]
        assert num_input_data == num_output_data

        # Even indices
        even_indices = self.split_evenly(num_input_data, self.batch_size)

        if np.mod(num_input_data, self.batch_size) != 0:
            # Remider indices
            rem_indices = self.split_reminder(num_input_data, self.batch_size)

            # Concat indices
            indices = np.concatenate((even_indices, rem_indices), axis=1)
        else:
            indices = np.stack(even_indices)

        input_data = raw_input[indices]
        output_data = raw_output[indices]
        dataset = []
        for x_batch, y_batch in zip(input_data, output_data):
            dataset.append((x_batch, y_batch))
        return dataset

    @staticmethod
    def split_data(data: int,
                   test_ratio: float = 0.2,
                   val_ratio: float = 0.0) -> dict:
        """Split data into training, validation, and test sets"""
        num_data = data.shape[1]
        splited_data = {}
        if val_ratio != 0.0:
            end_val_idx = num_data - int(test_ratio * num_data)
            end_train_idx = int(end_val_idx - val_ratio * end_val_idx)
            splited_data["train"] = data[:end_train_idx]
            splited_data["val"] = data[end_train_idx:end_val_idx]
            splited_data["test"] = data[end_val_idx:]
        else:
            end_train_idx = num_data - int(test_ratio * num_data)
            splited_data["train"] = data[:end_train_idx]
            splited_data["val"] = []
            splited_data["test"] = data[end_train_idx:]

        return splited_data

    @staticmethod
    def load_data_from_csv(data_file: str) -> pd.DataFrame:
        """Load data from csv file"""

        data = pd.read_csv(data_file, skiprows=1, header=None)

        return data.values

    @staticmethod
    def split_evenly(num_data, chunk_size: int):
        """split data evenly"""
        indices = np.arange(int(num_data - np.mod(num_data, chunk_size)))
        return np.split(indices, math.ceil(num_data / chunk_size))

    @staticmethod
    def split_reminder(num_data: int, chunk_size: int):
        """Pad the reminder"""
        indices = np.arange(num_data)
        reminder_start = math.ceil(num_data / chunk_size)
        random_idx = np.random.choice(indices,
                                      size=num_data - reminder_start,
                                      replace=False)
        reminder_idx = indices[reminder_start:]

        return np.concatenate((random_idx, reminder_idx), axis=0)


class RegressionDataLoader(DataloaderBase):
    """Load and format data that are feeded to the neural network.
     The user must provide the input and output data file in *csv"""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super.__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def process_data(self, x_train_file: str, y_train_file: str,
                     x_test_file: str, y_test_file: str) -> dict:
        """Process data from the csv file"""

        # Load data
        x_train = self.load_data_from_csv(x_train_file)
        y_train = self.load_data_from_csv(y_train_file)
        x_test = self.load_data_from_csv(x_test_file)
        y_test = self.load_data_from_csv(y_test_file)

        # Normalizer
        x_mean, x_std = self.normalizer.compute_mean_std(
            np.concatenate((x_train, x_test)))
        y_mean, y_std = self.normalizer.compute_mean_std(
            np.concatenate((y_train, y_test)))

        x_train = self.normalizer.standardize(data=x_train,
                                              mu=x_mean,
                                              std=x_std)
        y_train = self.normalizer.standardize(data=y_train,
                                              mu=y_mean,
                                              std=y_std)
        x_test = self.normalizer.standardize(data=x_test, mu=x_mean, std=x_std)
        y_test = self.normalizer.standardize(data=y_test, mu=y_mean, std=y_std)

        # Dataloader
        data_loader = {}
        data_loader["train"] = (x_train, y_train)
        data_loader["test"] = self.create_data_loader(raw_input=x_test,
                                                      raw_output=y_test)
        data_loader["x_norm_param_1"] = x_mean
        data_loader["x_norm_param_2"] = x_std
        data_loader["y_norm_param_1"] = y_mean
        data_loader["y_norm_param_2"] = y_std

        return data_loader


class MnistDataloader(DataloaderBase):
    """Data loader for mnist dataset"""

    def __init__(self) -> None:
        super.__init__()

    def process_data(self, x_train_file: str, y_train_file: str,
                     x_test_file: str, y_test_file: str) -> dict:
        """Process mnist images"""
        num_train_images = 60000
        num_test_images = 10000

        # Traininng set
        train_images, train_labels = utils.load_mnist_images(
            image_file=x_train_file,
            label_file=y_train_file,
            num_images=num_train_images)
        y_train, y_train_idx, num_enc_obs = utils.get_hierarchial_softmax(
            labels=train_labels, num_classes=10)
        x_mean, x_std = self.normalizer.compute_mean_std(train_images)
        breakpoint()

        # Test set
        test_images, test_labels = utils.load_mnist_images(
            image_file=x_test_file,
            label_file=y_test_file,
            num_images=num_train_images)

        # Normalizer
        x_train = self.normalizer.standardize(data=train_images,
                                              mu=x_mean,
                                              std=x_std)
        x_test = self.normalizer.standardize(data=test_images,
                                             mu=x_mean,
                                             std=x_std)

        y_train = y_train.reshape((num_train_images, num_enc_obs))
        y_train_idx = y_train_idx.reshape((num_train_images, num_enc_obs))
        x_train = x_train.reshape((num_train_images, 28, 28))
        x_test = x_train.reshape((num_test_images, 28, 28))

        # Data loader
        data_loader = {}
        data_loader["train"] = (x_train, y_train, y_train_idx, train_labels)
        data_loader["test"] = self.create_data_loader(raw_input=x_test,
                                                      raw_output=test_labels)
        data_loader["x_norm_param_1"] = x_mean
        data_loader["x_norm_param_2"] = x_std
