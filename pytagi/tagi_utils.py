from typing import Tuple, Union

import cutagi
import numpy as np

from pytagi.nn.data_struct import HRCSoftmax


class Utils:
    """A frontend for utility functions from the C++/CUDA backend.

    This class provides a Python interface to various utility functions implemented
    in the C++ `cutagi` library, such as data loading, preprocessing, and
    post-processing tasks related to machine learning models.

    :ivar _cpp_backend: An instance of `cutagi.Utils` which provides the
        backend functionalities.
    """

    def __init__(self) -> None:
        """Initializes the Utils class by creating an instance of the C++ backend."""
        self._cpp_backend = cutagi.Utils()

    def label_to_obs(
        self, labels: np.ndarray, num_classes: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Converts class labels into observations for a binary tree structure.

        This is used in the hierarchical classification, where each label
        is mapped to a path in a binary tree, and the observations represent
        the nodes along that path.

        :param labels: An array of class labels for the dataset.
        :type labels: numpy.ndarray
        :param num_classes: The total number of unique classes.
        :type num_classes: int
        :return: A tuple containing:
            - **obs** (*numpy.ndarray*): Encoded observations corresponding to the labels.
            - **obs_idx** (*numpy.ndarray*): Indices of the encoded observations.
            - **num_obs** (*int*): The total number of encoded observations.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray, int]
        """

        obs, obs_idx, num_obs = self._cpp_backend.label_to_obs_wrapper(
            labels, num_classes
        )

        return np.array(obs), np.array(obs_idx), int(num_obs)

    def label_to_one_hot(
        self, labels: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """Generates a one-hot encoding for the given labels.

        :param labels: An array of class labels for the dataset.
        :type labels: numpy.ndarray
        :param num_classes: The total number of unique classes.
        :type num_classes: int
        :return: A 2D array representing the one-hot encoded labels.
        :rtype: numpy.ndarray
        """

        return self._cpp_backend.label_to_one_hot_wrapper(labels, num_classes)

    def load_mnist_images(
        self, image_file: str, label_file: str, num_images: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads a specified number of images and labels from the MNIST dataset files.

        :param image_file: The file path to the MNIST image data (e.g., 'train-images-idx3-ubyte').
        :type image_file: str
        :param label_file: The file path to the MNIST label data (e.g., 'train-labels-idx1-ubyte').
        :type label_file: str
        :param num_images: The number of images to load from the files.
        :type num_images: int
        :return: A tuple containing:
            - **images** (*numpy.ndarray*): A 2D array of flattened MNIST images.
            - **labels** (*numpy.ndarray*): A 1D array of corresponding labels.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        images, labels = self._cpp_backend.load_mnist_dataset_wrapper(
            image_file, label_file, num_images
        )

        return images, labels

    def load_cifar_images(
        self, image_file: str, num: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads a specified number of images and labels from a CIFAR-10 dataset file.

        :param image_file: The file path to a CIFAR-10 data batch file.
        :type image_file: str
        :param num: The number of images to load from the file.
        :type num: int
        :return: A tuple containing:
            - **images** (*numpy.ndarray*): A 2D array of flattened CIFAR-10 images.
            - **labels** (*numpy.ndarray*): A 1D array of corresponding labels.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """

        images, labels = self._cpp_backend.load_cifar_dataset_wrapper(
            image_file, num
        )

        return images, labels

    def get_labels(
        self,
        ma: np.ndarray,
        Sa: np.ndarray,
        hr_softmax: HRCSoftmax,
        num_classes: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts class labels from the output layer's activation statistics.

        Uses hierarchical softmax to convert the mean and variance of the output
        layer's activations into class predictions and their probabilities.

        :param ma: The mean of the activation units for the output layer.
        :type ma: numpy.ndarray
        :param Sa: The variance of the activation units for the output layer.
        :type Sa: numpy.ndarray
        :param hr_softmax: An initialized hierarchical softmax structure.
        :type hr_softmax: pytagi.nn.HRCSoftmax
        :param num_classes: The total number of classes.
        :type num_classes: int
        :param batch_size: The number of samples in the batch.
        :type batch_size: int
        :return: A tuple containing:
            - **pred** (*numpy.ndarray*): The predicted class labels for the batch.
            - **prob** (*numpy.ndarray*): The probabilities for each predicted label.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """

        pred, prob = self._cpp_backend.get_labels_wrapper(
            ma, Sa, hr_softmax, num_classes, batch_size
        )

        return pred, prob

    def get_errors(
        self,
        ma: np.ndarray,
        Sa: np.ndarray,
        labels: np.ndarray,
        hr_softmax: HRCSoftmax,
        num_classes: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the prediction error given the output layer's statistics and true labels.

        This method calculates the classification error rate and probabilities based
        on the hierarchical softmax output.

        :param ma: The mean of the activation units for the output layer.
        :type ma: numpy.ndarray
        :param Sa: The variance of the activation units for the output layer.
        :type Sa: numpy.ndarray
        :param labels: The ground truth labels for the dataset.
        :type labels: numpy.ndarray
        :param hr_softmax: An initialized hierarchical softmax structure.
        :type hr_softmax: pytagi.nn.HRCSoftmax
        :param num_classes: The total number of classes.
        :type num_classes: int
        :param batch_size: The number of samples in a batch.
        :type batch_size: int
        :return: A tuple containing:
            - **pred** (*numpy.ndarray*): The prediction error for the batch.
            - **prob** (*numpy.ndarray*): The probabilities associated with the predictions.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """

        pred, prob = self._cpp_backend.get_error_wrapper(
            ma, Sa, labels, hr_softmax, num_classes, batch_size
        )

        return pred, prob

    def get_hierarchical_softmax(self, num_classes: int) -> HRCSoftmax:
        """Constructs a hierarchical softmax structure (binary tree) for classification.

        :param num_classes: The total number of classes to be included in the tree.
        :type num_classes: int
        :return: An object representing the hierarchical softmax structure.
        :rtype: pytagi.nn.HRCSoftmax
        """
        hr_softmax = self._cpp_backend.hierarchical_softmax_wrapper(num_classes)

        return hr_softmax

    def obs_to_label_prob(
        self,
        ma: np.ndarray,
        Sa: np.ndarray,
        hr_softmax: HRCSoftmax,
        num_classes: int,
    ) -> np.ndarray:
        """Converts observation probabilities to label probabilities.

        This function takes the output statistics of a model (mean and variance) and
        uses the hierarchical softmax structure to compute the probability of each class label.

        :param ma: The mean of the activation units for the output layer.
        :type ma: numpy.ndarray
        :param Sa: The variance of the activation units for the output layer.
        :type Sa: numpy.ndarray
        :param hr_softmax: An initialized hierarchical softmax structure.
        :type hr_softmax: pytagi.nn.HRCSoftmax
        :param num_classes: The total number of classes.
        :type num_classes: int
        :return: An array of probabilities for each class label.
        :rtype: numpy.ndarray
        """

        prob = self._cpp_backend.obs_to_label_prob_wrapper(
            ma, Sa, hr_softmax, num_classes
        )

        return np.array(prob)

    def create_rolling_window(
        self,
        data: np.ndarray,
        output_col: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
        num_features: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates input/output sequences for time-series forecasting using a rolling window.

        This method slides a window over the time-series data to generate
        input sequences and their corresponding future output sequences.

        :param data: The time-series dataset, typically a 2D array of shape (timesteps, features).
        :type data: numpy.ndarray
        :param output_col: The indices of the columns to be used as output targets.
        :type output_col: numpy.ndarray
        :param input_seq_len: The number of time steps in each input sequence.
        :type input_seq_len: int
        :param output_seq_len: The number of time steps in each output sequence.
        :type output_seq_len: int
        :param num_features: The total number of features in the dataset.
        :type num_features: int
        :param stride: The number of time steps to move the window forward for each new sequence.
        :type stride: int
        :return: A tuple containing:
            - **input_data** (*numpy.ndarray*): A 2D array of input sequences.
            - **output_data** (*numpy.ndarray*): A 2D array of corresponding output sequences.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        num_data = int(
            # (len(data) / num_features - input_seq_len - output_seq_len) / stride + 1
            (len(data) - input_seq_len - output_seq_len) / stride
            + 1
        )

        input_data, output_data = (
            self._cpp_backend.create_rolling_window_wrapper(
                data.flatten(),
                output_col,
                input_seq_len,
                output_seq_len,
                num_features,
                stride,
            )
        )
        # input_data = input_data.reshape((num_data, input_seq_len))
        input_data = input_data.reshape(
            (num_data, input_seq_len * num_features)
        )
        output_data = output_data.reshape((num_data, output_seq_len))

        return input_data, output_data

    def get_upper_triu_cov(
        self, batch_size: int, num_data: int, sigma: float
    ) -> np.ndarray:
        """Creates an upper triangular covariance matrix for correlated inputs.

        This is useful for models that assume temporal or spatial correlation
        in the input data, such as time-series models.

        :param batch_size: The number of samples in a batch.
        :type batch_size: int
        :param num_data: The number of data points (e.g., time steps) in each sample.
        :type num_data: int
        :param sigma: The standard deviation parameter controlling the covariance.
        :type sigma: float
        :return: A 1D array representing the flattened upper triangular part of the covariance matrix.
        :rtype: numpy.ndarray
        """

        vx_f = self._cpp_backend.get_upper_triu_cov_wrapper(
            batch_size, num_data, sigma
        )

        return np.array(vx_f)


def exponential_scheduler(
    curr_v: float, min_v: float, decaying_factor: float, curr_iter: int
) -> float:
    """Implements an exponential decay schedule for a given value.

    The value decays according to the formula:
    :math:`\\text{new_v} = \\max(\\text{curr_v} \\times (\\text{decaying_factor} ** \\text{curr_iter}), \\text{min_v})`.
    This is commonly used for learning rate scheduling or for decaying exploration rates.

    :param curr_v: The current value to be decayed.
    :type curr_v: float
    :param min_v: The minimum floor value that `curr_v` can decay to.
    :type min_v: float
    :param decaying_factor: The base of the exponential decay (e.g., 0.99).
    :type decaying_factor: float
    :param curr_iter: The current iteration number.
    :type curr_iter: int
    :return: The decayed value.
    :rtype: float
    """

    return np.maximum(curr_v * (decaying_factor**curr_iter), min_v)


class Normalizer:
    """A collection of methods for data normalization and denormalization.

    Provides common scaling techniques such as standardization (Z-score) and
    min-max normalization. It also includes methods to reverse the transformations.

    :param method: The normalization method to use. Currently, this parameter is
        not used in the methods but can be set for context.
    :type method: str or None, optional
    """

    def __init__(self, method: Union[str, None] = None) -> None:
        """Initializes the Normalizer.

        :param method: The name of the normalization method (e.g., 'standardize').
        :type method: str or None, optional
        """
        self.method = method

    @staticmethod
    def standardize(
        data: np.ndarray, mu: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """Applies Z-score normalization to the data.

        The transformation is given by: :math:`(data - \\mu) / (\\sigma + \\epsilon)`.

        :param data: The input data to normalize.
        :type data: numpy.ndarray
        :param mu: The mean of the data, typically computed per feature.
        :type mu: numpy.ndarray
        :param std: The standard deviation of the data, typically computed per feature.
        :type std: numpy.ndarray
        :return: The standardized data.
        :rtype: numpy.ndarray
        """

        return (data - mu) / (std + 1e-10)

    @staticmethod
    def unstandardize(
        norm_data: np.ndarray, mu: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """Reverts the Z-score normalization.

        The transformation is given by: :math:`\\text{norm_data} \\times (\\sigma + \\epsilon) + \\mu`.

        :param norm_data: The standardized data to transform back to the original scale.
        :type norm_data: numpy.ndarray
        :param mu: The original mean used for standardization.
        :type mu: numpy.ndarray
        :param std: The original standard deviation used for standardization.
        :type std: numpy.ndarray
        :return: The data in its original scale.
        :rtype: numpy.ndarray
        """
        return norm_data * (std + 1e-10) + mu

    @staticmethod
    def unstandardize_std(norm_std: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Scales a standardized standard deviation back to the original space.

        The transformation is given by: :math:`\\text{norm_std} \\times (\\sigma + \\epsilon)`.

        :param norm_std: The standardized standard deviation.
        :type norm_std: numpy.ndarray
        :param std: The original standard deviation of the data.
        :type std: numpy.ndarray
        :return: The standard deviation in its original scale.
        :rtype: numpy.ndarray
        """

        return norm_std * (std + 1e-10)

    def max_min_norm(
        self, data: np.ndarray, max_value: np.ndarray, min_value: np.ndarray
    ) -> np.ndarray:
        """Applies min-max normalization to scale data between 0 and 1.

        The transformation is given by:
        :math:`(\\text{data} - \\text{min_value}) / (\\text{max_value} - \\text{min_value} + \\epsilon)`.

        :param data: The input data to normalize.
        :type data: numpy.ndarray
        :param max_value: The maximum value of the data, typically per feature.
        :type max_value: numpy.ndarray
        :param min_value: The minimum value of the data, typically per feature.
        :type min_value: numpy.ndarray
        :return: The data scaled to the [0, 1] range.
        :rtype: numpy.ndarray
        """
        assert np.all(max_value >= min_value)
        return (data - min_value) / (max_value - min_value + 1e-10)

    @staticmethod
    def max_min_unnorm(
        norm_data: np.ndarray, max_value: np.ndarray, min_value: np.ndarray
    ) -> np.ndarray:
        """Reverts the min-max normalization.

        The transformation is given by:
        :math:`\\text{norm_data} \\times (\\text{max_value} - \\text{min_value} + \\epsilon) + \\text{min_value}`.

        :param norm_data: The min-max normalized data.
        :type norm_data: numpy.ndarray
        :param max_value: The original maximum value used for normalization.
        :type max_value: numpy.ndarray
        :param min_value: The original minimum value used for normalization.
        :type min_value: numpy.ndarray
        :return: The data in its original scale.
        :rtype: numpy.ndarray
        """

        return (norm_data * (max_value - min_value + 1e-10)) + min_value

    @staticmethod
    def max_min_unnorm_std(
        norm_std: np.ndarray, max_value: np.ndarray, min_value: np.ndarray
    ) -> np.ndarray:
        """Scales a standard deviation from the min-max normalized space to the original space.

        The transformation is given by:
        :math:`\\text{norm_std} \\times (\\text{max_value} - \\text{min_value} + \\epsilon)`.

        :param norm_std: The standard deviation in the normalized space.
        :type norm_std: numpy.ndarray
        :param max_value: The original maximum value of the data.
        :type max_value: numpy.ndarray
        :param min_value: The original minimum value of the data.
        :type min_value: numpy.ndarray
        :return: The standard deviation in the original data scale.
        :rtype: numpy.ndarray
        """

        return norm_std * (max_value - min_value + 1e-10)

    @staticmethod
    def compute_mean_std(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the sample mean and standard deviation of the data along axis 0.

        NaN values are ignored in the calculation.

        :param data: The input data array.
        :type data: numpy.ndarray
        :return: A tuple containing:
            - **mean** (*numpy.ndarray*): The mean of the data.
            - **std** (*numpy.ndarray*): The standard deviation of the data.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """

        return (np.nanmean(data, axis=0), np.nanstd(data, axis=0))

    @staticmethod
    def compute_max_min(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the maximum and minimum values of the data along axis 0.

        NaN values are ignored in the calculation.

        :param data: The input data array.
        :type data: numpy.ndarray
        :return: A tuple containing:
            - **max** (*numpy.ndarray*): The maximum values.
            - **min** (*numpy.ndarray*): The minimum values.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """

        return (np.nanmax(data, axis=0), np.nanmin(data, axis=0))
