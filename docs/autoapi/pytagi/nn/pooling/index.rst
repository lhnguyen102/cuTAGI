pytagi.nn.pooling
=================

.. py:module:: pytagi.nn.pooling


Classes
-------

.. autoapisummary::

   pytagi.nn.pooling.AvgPool2d
   pytagi.nn.pooling.MaxPool2d


Module Contents
---------------

.. py:class:: AvgPool2d(kernel_size: int, stride: int = -1, padding: int = 0, padding_type: int = 0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   2D Average Pooling Layer.

   This layer performs 2D average pooling operation. It wraps the C++/CUDA backend
   `cutagi.AvgPool2d`.

   Initializes the AvgPool2d layer.

   :param kernel_size: The size of the pooling window (a single integer for square kernels).
   :type kernel_size: int
   :param stride: The stride of the pooling operation. Default is -1, which typically means stride=kernel_size.
   :type stride: int
   :param padding: The implicit zero padding added to both sides of the input.
   :type padding: int
   :param padding_type: The type of padding to be used (e.g., 0 for zero padding).
   :type padding_type: int


   .. py:method:: get_layer_info() -> str

      Returns a string containing information about the layer.



   .. py:method:: get_layer_name() -> str

      Returns the name of the layer (e.g., 'AvgPool2d').



.. py:class:: MaxPool2d(kernel_size: int, stride: int = 1, padding: int = 0, padding_type: int = 0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   2D Max Pooling Layer.

   This layer performs 2D max pooling operation based on the input expected values.
   It wraps the C++/CUDA backend `cutagi.MaxPool2d`.

   Initializes the MaxPool2d layer.

   :param kernel_size: The size of the pooling window (a single integer for square kernels).
   :type kernel_size: int
   :param stride: The stride of the pooling operation. Default is 1.
   :type stride: int
   :param padding: The implicit zero padding added to both sides of the input.
   :type padding: int
   :param padding_type: The type of padding to be used (e.g., 0 for zero padding).
   :type padding_type: int


   .. py:method:: get_layer_info() -> str

      Returns a string containing information about the layer.



   .. py:method:: get_layer_name() -> str

      Returns the name of the layer (e.g., 'MaxPool2d').
