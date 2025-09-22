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

   Classifcation error for hierarchical softmax used for classification


   .. py:method:: error_rate(m_pred: numpy.ndarray, v_pred: numpy.ndarray, label: numpy.ndarray) -> float

      Compute error rate for classifier



   .. py:method:: get_predicted_labels(m_pred: numpy.ndarray, v_pred: numpy.ndarray) -> numpy.ndarray

      Get the prediction



.. py:function:: mse(prediction: numpy.ndarray, observation: numpy.ndarray) -> float

   Mean squared error


.. py:function:: log_likelihood(prediction: numpy.ndarray, observation: numpy.ndarray, std: numpy.ndarray) -> float

   Compute the averaged log-likelihood


.. py:function:: rmse(prediction: numpy.ndarray, observation: numpy.ndarray) -> None

   Root mean squared error


.. py:function:: classification_error(prediction: numpy.ndarray, label: numpy.ndarray) -> None

   Compute the classification error
