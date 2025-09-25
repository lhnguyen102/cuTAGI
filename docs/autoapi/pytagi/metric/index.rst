pytagi.metric
=============

.. py:module:: pytagi.metric


Classes
-------

.. autoapisummary::

   pytagi.metric.HRCSoftmaxMetric


Functions
---------

.. autoapisummary::

   pytagi.metric.mse
   pytagi.metric.log_likelihood
   pytagi.metric.rmse
   pytagi.metric.classification_error


Module Contents
---------------

.. py:class:: HRCSoftmaxMetric(num_classes: int)

   Classification error metric for Hierarchical Softmax.

   This class provides methods to compute the error rate and get predicted labels
   for a classification model that uses Hierarchical Softmax.


   .. py:method:: error_rate(m_pred: numpy.ndarray, v_pred: numpy.ndarray, label: numpy.ndarray) -> float

      Computes the classification error rate.

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



   .. py:method:: get_predicted_labels(m_pred: numpy.ndarray, v_pred: numpy.ndarray) -> numpy.ndarray

      Gets the predicted class labels from the model's output.

      :param m_pred: The mean of the predictions from the model.
      :type m_pred: np.ndarray
      :param v_pred: The variance of the predictions from the model.
      :type v_pred: np.ndarray
      :return: An array of predicted class labels.
      :rtype: np.ndarray



.. py:function:: mse(prediction: numpy.ndarray, observation: numpy.ndarray) -> float

   Calculates the Mean Squared Error (MSE).

   MSE measures the average of the squares of the errors—that is, the average
   squared difference between the estimated values and the actual value.

   :param prediction: The predicted values.
   :type prediction: np.ndarray
   :param observation: The actual (observed) values.
   :type observation: np.ndarray
   :return: The mean squared error.
   :rtype: float


.. py:function:: log_likelihood(prediction: numpy.ndarray, observation: numpy.ndarray, std: numpy.ndarray) -> float

   Computes the average Gaussian log-likelihood.

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


.. py:function:: rmse(prediction: numpy.ndarray, observation: numpy.ndarray) -> float

   Calculates the Root Mean Squared Error (RMSE).

   RMSE is the square root of the mean of the squared errors.

   :param prediction: The predicted values.
   :type prediction: np.ndarray
   :param observation: The actual (observed) values.
   :type observation: np.ndarray
   :return: The root mean squared error.
   :rtype: float


.. py:function:: classification_error(prediction: numpy.ndarray, label: numpy.ndarray) -> float

   Computes the classification error rate.

   This function calculates the fraction of predictions that do not match the
   true labels.

   :param prediction: An array of predicted labels.
   :type prediction: np.ndarray
   :param label: An array of true labels.
   :type label: np.ndarray
   :return: The classification error rate (proportion of incorrect predictions).
   :rtype: float
