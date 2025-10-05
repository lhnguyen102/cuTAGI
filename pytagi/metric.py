import numpy as np

from pytagi.nn.data_struct import HRCSoftmax
from pytagi.tagi_utils import Utils


class HRCSoftmaxMetric:
    """Classification error metric for Hierarchical Softmax.

    This class provides methods to compute the error rate and get predicted labels
    for a classification model that uses Hierarchical Softmax.
    """

    def __init__(self, num_classes: int):
        """Initializes the HRCSoftmaxMetric.

        :param num_classes: The total number of classes in the classification problem.
        :type num_classes: int
        """
        self.num_classes = num_classes
        self.utils = Utils()
        self.hrc_softmax: HRCSoftmax = self.utils.get_hierarchical_softmax(
            num_classes=num_classes
        )

    def error_rate(
        self, m_pred: np.ndarray, v_pred: np.ndarray, label: np.ndarray
    ) -> float:
        """Computes the classification error rate.

        This method calculates the proportion of incorrect predictions by comparing
        the predicted labels against the true labels.

        :param m_pred: The mean of the predictions from the model.
        :type m_pred: np.ndarray
        :param v_pred: The variance of the predictions from the model.
        :type v_pred: np.ndarray
        :param label: The ground truth labels.
        :type label: np.ndarray
        :return: The classification error rate, a value between 0 and 1.
        :rtype: float
        """
        batch_size = m_pred.shape[0] // self.hrc_softmax.len
        pred, _ = self.utils.get_labels(
            m_pred, v_pred, self.hrc_softmax, self.num_classes, batch_size
        )
        return classification_error(pred, label)

    def get_predicted_labels(
        self, m_pred: np.ndarray, v_pred: np.ndarray
    ) -> np.ndarray:
        """Gets the predicted class labels from the model's output.

        :param m_pred: The mean of the predictions from the model.
        :type m_pred: np.ndarray
        :param v_pred: The variance of the predictions from the model.
        :type v_pred: np.ndarray
        :return: An array of predicted class labels.
        :rtype: np.ndarray
        """
        batch_size = m_pred.shape[0] // self.hrc_softmax.len
        pred, _ = self.utils.get_labels(
            m_pred, v_pred, self.hrc_softmax, self.num_classes, batch_size
        )
        return pred


def mse(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Calculates the Mean Squared Error (MSE).

    MSE measures the average of the squares of the errors, i.e., the average
    squared difference between the estimated and the observed values.

    :param prediction: The predicted values.
    :type prediction: np.ndarray
    :param observation: The actual (observed) values.
    :type observation: np.ndarray
    :return: The mean squared error.
    :rtype: float
    """
    return np.nanmean((prediction - observation) ** 2)


def log_likelihood(
    prediction: np.ndarray, observation: np.ndarray, std: np.ndarray
) -> float:
    """Computes the log-likelihood.

    This function assumes the likelihood of the observation given the prediction
    is a Gaussian distribution with a given standard deviation.

    :param prediction: The predicted mean of the distribution.
    :type prediction: np.ndarray
    :param observation: The observed data points.
    :type observation: np.ndarray
    :param std: The standard deviation of the distribution.
    :type std: np.ndarray
    :return: The average log-likelihood value.
    :rtype: float
    """
    log_lik = -0.5 * np.log(2 * np.pi * (std**2)) - 0.5 * (
        ((observation - prediction) / std) ** 2
    )
    return np.nanmean(log_lik)


def rmse(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Calculates the Root Mean Squared Error (RMSE).

    RMSE is the square root of the mean of the squared errors.

    :param prediction: The predicted values.
    :type prediction: np.ndarray
    :param observation: The actual (observed) values.
    :type observation: np.ndarray
    :return: The root mean squared error.
    :rtype: float
    """
    mse_val = mse(prediction, observation)
    return mse_val**0.5


def classification_error(prediction: np.ndarray, label: np.ndarray) -> float:
    """Computes the classification error rate.

    This function calculates the fraction of predictions that do not match the
    true labels.

    :param prediction: An array of predicted labels.
    :type prediction: np.ndarray
    :param label: An array of true labels.
    :type label: np.ndarray
    :return: The classification error rate (proportion of incorrect predictions).
    :rtype: float
    """
    count = 0
    for pred, lab in zip(prediction.T, label):
        if pred != lab:
            count += 1
    return count / len(prediction)
