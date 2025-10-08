pytagi.nn.batch_norm
====================

.. py:module:: pytagi.nn.batch_norm


Classes
-------

.. autoapisummary::

   pytagi.nn.batch_norm.BatchNorm2d


Module Contents
---------------

.. py:class:: BatchNorm2d(num_features: int, eps: float = 1e-05, momentum: float = 0.9, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies 2D Batch Normalization.

   Batch Normalization normalizes the inputs of a layer by re-centering and
   re-scaling them.

   :param num_features: The number of features in the input tensor.
   :type num_features: int
   :param eps: A small value added to the variance to avoid division by zero.
               Defaults to 1e-5.
   :type eps: float
   :param momentum: The momentum for the running mean and variance.
                    Defaults to 0.9.
   :type momentum: float
   :param bias: Whether to include a learnable bias term. Defaults to True.
   :type bias: bool
   :param gain_weight: Initial value for the gain (scale) parameter. Defaults to 1.0.
   :type gain_weight: float
   :param gain_bias: Initial value for the bias (shift) parameter. Defaults to 1.0.
   :type gain_bias: float

   Initializes the BatchNorm2d layer.


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the BatchNorm2d layer.

      :returns:

                A string containing the layer's information, typically delegated
                     to the C++ backend implementation.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the BatchNorm2d layer.

      :returns: The name of the layer, typically delegated to the C++ backend implementation.
      :rtype: str



   .. py:method:: init_weight_bias()

      Initializes the learnable weight (scale/gain) and bias (shift/offset)
      parameters of the batch normalization layer. This operation is
      delegated to the C++ backend.
