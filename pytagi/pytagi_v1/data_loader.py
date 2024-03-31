import os
from typing import Tuple

# Temporary import. It will be removed in the final vserion
import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pytagi import Normalizer


from utils import Utils


class DataloaderBase(ABC):
    """Dataloader template"""

    normalizer: Normalizer = Normalizer()

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    @abstractmethod
    def process_data(self) -> dict:
        raise NotImplementedError

    def create_data_loader(self, raw_input: np.ndarray, raw_output: np.ndarray) -> list:
        """Create dataloader based on batch size"""
        num_input_data = raw_input.shape[0]
        num_output_data = raw_output.shape[0]
        assert num_input_data == num_output_data

        # Even indices
        even_indices = self.split_evenly(num_input_data, self.batch_size)

        if np.mod(num_input_data, self.batch_size) != 0:
            # Remider indices
            rem_indices = self.split_reminder(num_input_data, self.batch_size)
            even_indices.append(rem_indices)

        indices = np.stack(even_indices)
        input_data = raw_input[indices]
        output_data = raw_output[indices]
        dataset = []
        for x_batch, y_batch in zip(input_data, output_data):
            dataset.append((x_batch, y_batch))
        return dataset

    @staticmethod
    def split_data(data: int, test_ratio: float = 0.2, val_ratio: float = 0.0) -> dict:
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

        data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

        return data.values

    @staticmethod
    def split_evenly(num_data, chunk_size: int):
        """split data evenly"""
        indices = np.arange(int(num_data - np.mod(num_data, chunk_size)))

        return np.split(indices, int(np.floor(num_data / chunk_size)))

    @staticmethod
    def split_reminder(num_data: int, chunk_size: int):
        """Pad the reminder"""
        indices = np.arange(num_data)
        reminder_start = int(num_data - np.mod(num_data, chunk_size))
        num_samples = chunk_size - (num_data - reminder_start)
        random_idx = np.random.choice(indices, size=num_samples, replace=False)
        reminder_idx = indices[reminder_start:]

        return np.concatenate((random_idx, reminder_idx))


class RegressionDataLoader(DataloaderBase):
    """Load and format data that are feeded to the neural network.
    The user must provide the input and output data file in *csv"""

    def __init__(self, batch_size: int, num_inputs: int, num_outputs: int) -> None:
        super().__init__(batch_size)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def process_data(
        self, x_train_file: str, y_train_file: str, x_test_file: str, y_test_file: str
    ) -> dict:
        """Process data from the csv file"""

        # Load data
        x_train = self.load_data_from_csv(x_train_file)
        y_train = self.load_data_from_csv(y_train_file)
        x_test = self.load_data_from_csv(x_test_file)
        y_test = self.load_data_from_csv(y_test_file)

        # Normalizer
        x_mean, x_std = self.normalizer.compute_mean_std(
            np.concatenate((x_train, x_test))
        )
        y_mean, y_std = self.normalizer.compute_mean_std(
            np.concatenate((y_train, y_test))
        )

        x_train = self.normalizer.standardize(data=x_train, mu=x_mean, std=x_std)
        y_train = self.normalizer.standardize(data=y_train, mu=y_mean, std=y_std)
        x_test = self.normalizer.standardize(data=x_test, mu=x_mean, std=x_std)
        y_test = self.normalizer.standardize(data=y_test, mu=y_mean, std=y_std)

        # Dataloader
        data_loader = {}
        data_loader["train"] = (x_train, y_train)
        data_loader["test"] = self.create_data_loader(
            raw_input=x_test, raw_output=y_test
        )
        data_loader["x_norm_param_1"] = x_mean
        data_loader["x_norm_param_2"] = x_std
        data_loader["y_norm_param_1"] = y_mean
        data_loader["y_norm_param_2"] = y_std

        return data_loader


class MnistDataloader(DataloaderBase):
    """Data loader for mnist dataset"""

    def process_data(
        self, x_train_file: str, y_train_file: str, x_test_file: str, y_test_file: str
    ) -> dict:
        """Process mnist images"""
        # Initialization
        utils = Utils()
        num_train_images = 60000
        num_test_images = 10000

        # Traininng set
        train_images, train_labels = utils.load_mnist_images(
            image_file=x_train_file,
            label_file=y_train_file,
            num_images=num_train_images,
        )

        y_train, y_train_idx, num_enc_obs = utils.label_to_obs(
            labels=train_labels, num_classes=10
        )
        x_mean, x_std = self.normalizer.compute_mean_std(train_images)
        x_std = 1

        # Test set
        test_images, test_labels = utils.load_mnist_images(
            image_file=x_test_file, label_file=y_test_file, num_images=num_test_images
        )

        # Normalizer
        x_train = self.normalizer.standardize(data=train_images, mu=x_mean, std=x_std)
        x_test = self.normalizer.standardize(data=test_images, mu=x_mean, std=x_std)

        y_train = y_train.reshape((num_train_images, num_enc_obs))
        y_train_idx = y_train_idx.reshape((num_train_images, num_enc_obs))
        x_train = x_train.reshape((num_train_images, 28, 28))
        x_test = x_test.reshape((num_test_images, 28, 28))

        # Data loader
        data_loader = {}
        data_loader["train"] = (x_train, y_train, y_train_idx, train_labels)
        data_loader["test"] = self.create_data_loader(
            raw_input=x_test, raw_output=test_labels
        )
        data_loader["x_norm_param_1"] = x_mean
        data_loader["x_norm_param_2"] = x_std

        return data_loader


class MnistOneHotDataloader(DataloaderBase):
    """Data loader for mnist dataset"""

    def process_data(
        self, x_train_file: str, y_train_file: str, x_test_file: str, y_test_file: str
    ) -> dict:
        """Process mnist images"""
        # Initialization
        utils = Utils()
        num_train_images = 60000
        num_test_images = 10000

        # Traininng set
        train_images, train_labels = utils.load_mnist_images(
            image_file=x_train_file,
            label_file=y_train_file,
            num_images=num_train_images,
        )

        y_train = utils.label_to_one_hot(labels=train_labels, num_classes=10)
        x_mean, x_std = self.normalizer.compute_mean_std(train_images)
        x_std = 1

        # Test set
        test_images, test_labels = utils.load_mnist_images(
            image_file=x_test_file, label_file=y_test_file, num_images=num_test_images
        )

        # Normalizer
        x_train = self.normalizer.standardize(data=train_images, mu=x_mean, std=x_std)
        x_test = self.normalizer.standardize(data=test_images, mu=x_mean, std=x_std)

        y_train = y_train.reshape((num_train_images, 10))
        x_train = x_train.reshape((num_train_images, 28, 28))
        x_test = x_test.reshape((num_test_images, 28, 28))

        # Data loader
        data_loader = {}
        data_loader["train"] = (x_train, y_train, train_labels)
        data_loader["test"] = self.create_data_loader(
            raw_input=x_test, raw_output=test_labels
        )
        data_loader["x_norm_param_1"] = x_mean
        data_loader["x_norm_param_2"] = x_std

        return data_loader


class TimeSeriesDataloader(DataloaderBase):
    """Data loader for time series"""

    def __init__(
        self,
        batch_size: int,
        output_col: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
        num_features: int,
        stride: int,
    ) -> None:
        super().__init__(batch_size)
        self.output_col = output_col
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.num_features = num_features
        self.stride = stride

    def process_data(
        self,
        x_train_file: str,
        datetime_train_file: str,
        x_test_file: str,
        datetime_test_file: str,
    ) -> dict:
        """Process time series"""
        # Initialization
        utils = Utils()

        # Load data
        x_train = self.load_data_from_csv(x_train_file)
        datetime_train = self.load_data_from_csv(datetime_train_file)

        x_test = self.load_data_from_csv(x_test_file)
        datetime_test = self.load_data_from_csv(datetime_test_file)

        # Normalizer
        x_mean, x_std = self.normalizer.compute_mean_std(x_train)
        x_train = self.normalizer.standardize(data=x_train, mu=x_mean, std=x_std)
        x_test = self.normalizer.standardize(data=x_test, mu=x_mean, std=x_std)

        # Create rolling windows
        x_train_rolled, y_train_rolled = utils.create_rolling_window(
            data=x_train,
            output_col=self.output_col,
            input_seq_len=self.input_seq_len,
            output_seq_len=self.output_seq_len,
            num_features=self.num_features,
            stride=self.stride,
        )

        x_test_rolled, y_test_rolled = utils.create_rolling_window(
            data=x_test,
            output_col=self.output_col,
            input_seq_len=self.input_seq_len,
            output_seq_len=self.output_seq_len,
            num_features=self.num_features,
            stride=self.stride,
        )

        # Dataloader
        data_loader = {}
        data_loader["train"] = (x_train_rolled, y_train_rolled)
        data_loader["test"] = self.create_data_loader(
            raw_input=x_test_rolled, raw_output=y_test_rolled
        )
        # Store normalization parameters
        data_loader["x_norm_param_1"] = x_mean
        data_loader["x_norm_param_2"] = x_std
        data_loader["y_norm_param_1"] = x_mean[self.output_col]
        data_loader["y_norm_param_2"] = x_std[self.output_col]

        # NOTE: Datetime is saved for the visualization purpose
        data_loader["datetime_train"] = [
            np.datetime64(date) for date in np.squeeze(datetime_train)
        ]
        data_loader["datetime_test"] = [
            np.datetime64(date) for date in np.squeeze(datetime_test)
        ]

        return data_loader
