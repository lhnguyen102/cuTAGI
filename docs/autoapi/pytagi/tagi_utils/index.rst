pytagi.tagi_utils
=================

.. py:module:: pytagi.tagi_utils


Classes
-------

.. autoapisummary::

   pytagi.tagi_utils.Utils
   pytagi.tagi_utils.Normalizer


Functions
---------

.. autoapisummary::

   pytagi.tagi_utils.exponential_scheduler


Module Contents
---------------

.. py:class:: Utils

   Frontend for utility functions from C++/CUDA backend

   .. attribute:: _cpp_backend

      Utility functionalities from the backend


   .. py:method:: label_to_obs(labels: numpy.ndarray, num_classes: int) -> Tuple[numpy.ndarray, numpy.ndarray, int]

      Get observations and observation indices of the binary tree for
          classification

      :param labels: Labels of dataset
      :param num_classes: Total number of classes

      :returns: Encoded observations of the labels
                obs_idx: Indices of the encoded observations in the output vector
                num_obs: Number of encoded observations
      :rtype: obs



   .. py:method:: label_to_one_hot(labels: numpy.ndarray, num_classes: int) -> numpy.ndarray

      Get the one hot encoder for each class

      :param labels: Labels of dataset
      :param num_classes: Total number of classes

      :returns: One hot encoder
      :rtype: one_hot



   .. py:method:: load_mnist_images(image_file: str, label_file: str, num_images: int) -> Tuple[numpy.ndarray, numpy.ndarray]

      Load mnist dataset

      :param image_file: Location of the Mnist image file
      :param label_file: Location of the Mnist label file
      :param num_images: Number of images to be loaded

      :returns: Image dataset
                labels: Label dataset
                num_images: Total number of images
      :rtype: images



   .. py:method:: load_cifar_images(image_file: str, num: int) -> Tuple[numpy.ndarray, numpy.ndarray]

      Load cifar dataset

      :param image_file: Location of image file
      :param num: Number of images to be loaded

      :returns: Image dataset
                labels: Label dataset
      :rtype: images



   .. py:method:: get_labels(ma: numpy.ndarray, Sa: numpy.ndarray, hr_softmax: pytagi.nn.HRCSoftmax, num_classes: int, batch_size: int) -> Tuple[numpy.ndarray, numpy.ndarray]

      Convert last layer's hidden state to labels

      :param ma: Mean of activation units for the output layer
      :param Sa: Variance of activation units for the output layer
      :param hr_softmax: Hierarchical softmax
      :param num_classes: Total number of classes
      :param batch_size: Number of data in a batch

      :returns: Label prediciton
                prob: Probability for each label
      :rtype: pred



   .. py:method:: get_errors(ma: numpy.ndarray, Sa: numpy.ndarray, labels: numpy.ndarray, hr_softmax: pytagi.nn.HRCSoftmax, num_classes: int, batch_size: int) -> Tuple[numpy.ndarray, numpy.ndarray]

      Convert last layer's hidden state to labels

      :param ma: Mean of activation units for the output layer
      :param Sa: Variance of activation units for the output layer
      :param labels: Label dataset
      :param hr_softmax: Hierarchical softmax
      :param num_classes: Total number of classes
      :param batch_size: Number of data in a batch

      :returns: Label prediction
                prob: Probability for each label
      :rtype: pred



   .. py:method:: get_hierarchical_softmax(num_classes: int) -> pytagi.nn.HRCSoftmax

      Convert labels to binary tree

      :param num_classes: Total number of classes

      :returns: Hierarchical softmax
      :rtype: hr_softmax



   .. py:method:: obs_to_label_prob(ma: numpy.ndarray, Sa: numpy.ndarray, hr_softmax: pytagi.nn.HRCSoftmax, num_classes: int) -> numpy.ndarray

      Convert observation to label probabilities

      :param ma: Mean of activation units for the output layer
      :param Sa: Variance of activation units for the output layer
      :param hr_softmax: Hierarchical softmax
      :param num_classes: Total number of classes

      :returns: Probability for each label
      :rtype: prob



   .. py:method:: create_rolling_window(data: numpy.ndarray, output_col: numpy.ndarray, input_seq_len: int, output_seq_len: int, num_features: int, stride: int) -> Tuple[numpy.ndarray, numpy.ndarray]

      Create rolling window for time series data

      :param data: dataset
      :param output_col: Indices of the output columns
      :param input_seq_len: Length of the input sequence
      :param output_seq_len: Length of the output sequence
      :param num_features: Number of features
      :param stride: Controls number of steps for the window movements

      :returns: Input data for neural networks in sequence
                output_data: Output data for neural networks in sequence
      :rtype: input_data



   .. py:method:: get_upper_triu_cov(batch_size: int, num_data: int, sigma: float) -> numpy.ndarray

      Create an upper triangle covriance matrix for inputs



.. py:function:: exponential_scheduler(curr_v: float, min_v: float, decaying_factor: float, curr_iter: float) -> float

   Exponentially decaying


.. py:class:: Normalizer(method: Union[str, None] = None)

   Different method to normalize the data before feeding
   to neural networks


   .. py:method:: standardize(data: numpy.ndarray, mu: numpy.ndarray, std: numpy.ndarray) -> numpy.ndarray
      :staticmethod:


      Z-score normalization where
      data_norm = (data - data_mean) / data_std



   .. py:method:: unstandardize(norm_data: numpy.ndarray, mu: numpy.ndarray, std: numpy.ndarray) -> numpy.ndarray
      :staticmethod:


      Transform standardized data to original space



   .. py:method:: unstandardize_std(norm_std: numpy.ndarray, std: numpy.ndarray) -> numpy.ndarray
      :staticmethod:


      Transform standardized std to original space



   .. py:method:: max_min_norm(data: numpy.ndarray, max_value: numpy.ndarray, min_value: numpy.ndarray) -> numpy.ndarray

      Normalize the data between 0 and 1



   .. py:method:: max_min_unnorm(norm_data: numpy.ndarray, max_value: numpy.ndarray, min_value: numpy.ndarray) -> numpy.ndarray
      :staticmethod:


      Transform max-min normalized data to original space



   .. py:method:: max_min_unnorm_std(norm_std: numpy.ndarray, max_value: numpy.ndarray, min_value: numpy.ndarray) -> numpy.ndarray
      :staticmethod:


      Transform max-min normalized std to original space



   .. py:method:: compute_mean_std(data: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]
      :staticmethod:


      Compute sample mean and standard deviation



   .. py:method:: compute_max_min(data: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]
      :staticmethod:


      Compute max min values
