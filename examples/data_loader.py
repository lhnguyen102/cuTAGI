from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd

from pytagi import Normalizer, Utils


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


class RegressionDataLoader:
    """Load and format data that are feeded to the neural network.
    The user must provide the input and output data file in *csv"""

    def __init__(
        self,
        x_file: str,
        y_file: str,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        y_mean: Optional[np.ndarray] = None,
        y_std: Optional[np.ndarray] = None,
    ) -> None:
        self.x_file = x_file
        self.y_file = y_file
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

        self.dataset = self.process_data()

    def load_data_from_csv(self, data_file: str) -> pd.DataFrame:
        """Load data from csv file"""

        data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

        return data.values

    @staticmethod
    def batch_generator(
        input_data: np.ndarray,
        output_data: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
        """
        Generator function to yield batches of data.
        """
        num_data = input_data.shape[0]
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, num_data, batch_size):
            # if start_idx + batch_size > num_data:
            #     continue
            end_idx = min(start_idx + batch_size, num_data)
            idx = indices[start_idx:end_idx]
            yield input_data[idx].flatten(), output_data[idx].flatten()

    def process_data(self) -> dict:
        """Process data from the csv file"""

        # Load data
        x = self.load_data_from_csv(self.x_file)
        y = self.load_data_from_csv(self.y_file)

        # Normalizer
        if self.x_mean is None:
            self.x_mean, self.x_std = Normalizer.compute_mean_std(x)
            self.y_mean, self.y_std = Normalizer.compute_mean_std(y)

        x = Normalizer.standardize(data=x, mu=self.x_mean, std=self.x_std)
        y = Normalizer.standardize(data=y, mu=self.y_mean, std=self.y_std)

        # Dataloader
        dataset = {}
        dataset["value"] = (np.float32(x), np.float32(y))

        return dataset

    def create_data_loader(self, batch_size: int, shuffle: bool = True):
        return self.batch_generator(*self.dataset["value"], batch_size, shuffle)


class MnistDataLoader:
    """Data loader for MNIST."""

    def __init__(
        self,
        x_file: Optional[str] = None,
        y_file: Optional[str] = None,
        num_images: int = None,
    ):
        if x_file is None and y_file is None:
            # Load from default location
            x_file = ("data/mnist/train-images-idx3-ubyte",)
            y_file = ("data/mnist/train-labels-idx1-ubyte",)
            num_images = 60000
        self.dataset = self.process_data(x_file, y_file, num_images)

    @staticmethod
    def load_mnist_images(image_file: str, label_file: str, num_images: int):
        """Load raw data"""
        utils = Utils()
        images, labels = utils.load_mnist_images(
            image_file=image_file, label_file=label_file, num_images=num_images
        )
        return images, labels

    @staticmethod
    def batch_generator(
        input_data: np.ndarray,
        output_data: np.ndarray,
        output_idx: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
        """
        Generator function to yield batches of data.
        """
        num_data = input_data.shape[0]
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, num_data, batch_size):
            end_idx = min(start_idx + batch_size, num_data)
            idx = indices[start_idx:end_idx]
            yield input_data[idx].flatten(), output_data[idx].flatten(), output_idx[
                idx
            ].flatten(), labels[idx].flatten()

    def process_data(
        self, x_train_file: str, y_train_file: str, num_images: int
    ) -> dict:
        """Process MNIST images."""
        # Load training and test data
        utils = Utils()
        images, labels = self.load_mnist_images(x_train_file, y_train_file, num_images)

        y, y_idx, num_enc_obs = utils.label_to_obs(labels=labels, num_classes=10)
        x_mean, x_std = Normalizer.compute_mean_std(images)
        x_std = 1

        # Normalizer
        x = Normalizer.standardize(data=images, mu=x_mean, std=x_std)

        y = np.float32(y.reshape((num_images, num_enc_obs)))
        y_idx = y_idx.reshape((num_images, num_enc_obs))
        x = x.reshape((num_images, 28, 28))

        return {"value": (x, y, y_idx, labels)}

    def create_data_loader(self, batch_size: int, shuffle: bool = True):
        return self.batch_generator(*self.dataset["value"], batch_size, shuffle)


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
        dataset = {}
        dataset["train"] = (x_train, y_train, train_labels)
        dataset["test"] = self.create_data_loader(
            raw_input=x_test, raw_output=test_labels
        )
        dataset["x_norm_param_1"] = x_mean
        dataset["x_norm_param_2"] = x_std

        return dataset


class TimeSeriesDataloader:
    """Data loader for time series"""

    def __init__(
        self,
        x_file: str,
        date_time_file: str,
        output_col: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
        num_features: int,
        stride: int,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        ts_idx: Optional[int] = None,
        time_covariates: Optional[str] = None,
    ) -> None:
        self.x_file = x_file
        self.date_time_file = date_time_file
        self.output_col = output_col
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.num_features = num_features
        self.stride = stride
        self.x_mean = x_mean
        self.x_std = x_std
        self.ts_idx = ts_idx  # add time series index when data having multiple ts
        self.time_covariates = time_covariates  # for adding time covariates
        self.dataset = self.process_data()

    def load_data_from_csv(self, data_file: str) -> pd.DataFrame:
        """Load data from csv file"""

        data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

        return data.values

    @staticmethod
    def batch_generator(
        input_data: np.ndarray,
        output_data: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
        """
        Generator function to yield batches of data.
        """
        num_data = input_data.shape[0]
        indices = np.arange(num_data)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, num_data, batch_size):
            # if start_idx + batch_size > num_data:
            #     continue
            end_idx = min(start_idx + batch_size, num_data)
            idx = indices[start_idx:end_idx]
            yield input_data[idx].flatten(), output_data[idx].flatten()

    def process_data(self) -> dict:
        """Process time series"""
        # Initialization
        utils = Utils()

        # Load data
        x = self.load_data_from_csv(self.x_file)
        if self.ts_idx is not None:
            x = x[:, self.ts_idx : self.ts_idx + 1]  # choose time series column
        date_time = self.load_data_from_csv(self.date_time_file)

        # Add time covariates
        if self.time_covariates is not None:
            date_time = np.array(date_time, dtype="datetime64")
            for time_cov in self.time_covariates:
                if time_cov == "hour_of_day":
                    hour_of_day = date_time.astype("datetime64[h]").astype(int) % 24
                    x = np.concatenate((x, hour_of_day), axis=1)
                elif time_cov == "day_of_week":
                    day_of_week = date_time.astype("datetime64[D]").astype(int) % 7
                    x = np.concatenate((x, day_of_week), axis=1)
                elif time_cov == "week_of_year":
                    week_of_year = (
                        date_time.astype("datetime64[W]").astype(int) % 52 + 1
                    )
                    x = np.concatenate((x, week_of_year), axis=1)
                elif time_cov == "month_of_year":
                    month_of_year = (
                        date_time.astype("datetime64[M]").astype(int) % 12 + 1
                    )
                    x = np.concatenate((x, month_of_year), axis=1)
                elif time_cov == "quarter_of_year":
                    month_of_year = (
                        date_time.astype("datetime64[M]").astype(int) % 12 + 1
                    )
                    quarter_of_year = (month_of_year - 1) // 3 + 1
                    x = np.concatenate((x, quarter_of_year), axis=1)

        # Normalizer
        if self.x_mean is None and self.x_std is None:
            self.x_mean, self.x_std = Normalizer.compute_mean_std(x)
        x = Normalizer.standardize(data=x, mu=self.x_mean, std=self.x_std)

        # Create rolling windows
        x_rolled, y_rolled = utils.create_rolling_window(
            data=x,
            output_col=self.output_col,
            input_seq_len=self.input_seq_len,
            output_seq_len=self.output_seq_len,
            num_features=self.num_features,
            stride=self.stride,
        )

        # Dataloader
        dataset = {}
        dataset["value"] = (x_rolled, y_rolled)

        # NOTE: Datetime is saved for the visualization purpose
        dataset["date_time"] = [np.datetime64(date) for date in np.squeeze(date_time)]

        return dataset

    def create_data_loader(self, batch_size: int, shuffle: bool = True):
        return self.batch_generator(*self.dataset["value"], batch_size, shuffle)


# class TimeSeriesDataloader:
#     """Data loader for time series"""

#     def __init__(
#         self,
#         x_file: str,
#         date_time_file: str,
#         output_col: np.ndarray,
#         input_seq_len: int,
#         output_seq_len: int,
#         num_features: int,
#         stride: int,
#         x_mean: Optional[np.ndarray] = None,
#         x_std: Optional[np.ndarray] = None,
#     ) -> None:
#         self.x_file = x_file
#         self.date_time_file = date_time_file
#         self.output_col = output_col
#         self.input_seq_len = input_seq_len
#         self.output_seq_len = output_seq_len
#         self.num_features = num_features
#         self.stride = stride
#         self.x_mean = x_mean
#         self.x_std = x_std

#         self.dataset = self.process_data()

#     def load_data_from_csv(self, data_file: str) -> pd.DataFrame:
#         """Load data from csv file"""

#         data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

#         return data.values

#     @staticmethod
#     def batch_generator(
#         input_data: np.ndarray,
#         output_data: np.ndarray,
#         batch_size: int,
#         shuffle: bool = True,
#     ) -> Generator[Tuple[np.ndarray, ...], None, None]:
#         """
#         Generator function to yield batches of data.
#         """
#         num_data = input_data.shape[0]
#         indices = np.arange(num_data)
#         if shuffle:
#             np.random.shuffle(indices)

#         for start_idx in range(0, num_data, batch_size):
#             if start_idx + batch_size > num_data:
#                 continue
#             end_idx = min(start_idx + batch_size, num_data)
#             idx = indices[start_idx:end_idx]
#             yield input_data[idx].flatten(), output_data[idx].flatten()

#     def process_data(self) -> dict:
#         """Process time series"""
#         # Initialization
#         utils = Utils()

#         # Load data
#         x = self.load_data_from_csv(self.x_file)
#         date_time = self.load_data_from_csv(self.date_time_file)

#         # Normalizer
#         if self.x_mean is None and self.x_std is None:
#             self.x_mean, self.x_std = Normalizer.compute_mean_std(x)
#         x = Normalizer.standardize(data=x, mu=self.x_mean, std=self.x_std)

#         # Create rolling windows
#         x_rolled, y_rolled = utils.create_rolling_window(
#             data=x,
#             output_col=self.output_col,
#             input_seq_len=self.input_seq_len,
#             output_seq_len=self.output_seq_len,
#             num_features=self.num_features,
#             stride=self.stride,
#         )

#         # Dataloader
#         dataset = {}
#         dataset["value"] = (x_rolled, y_rolled)

#         # NOTE: Datetime is saved for the visualization purpose
#         dataset["date_time"] = [np.datetime64(date) for date in np.squeeze(date_time)]

#         return dataset

#     def create_data_loader(self, batch_size: int, shuffle: bool = True):
#         return self.batch_generator(*self.dataset["value"], batch_size, shuffle)
