pytagi.nn.conv2d
================

.. py:module:: pytagi.nn.conv2d


Classes
-------

.. autoapisummary::

   pytagi.nn.conv2d.Conv2d


Module Contents
---------------

.. py:class:: Conv2d(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, stride: int = 1, padding: int = 0, padding_type: int = 1, in_width: int = 0, in_height: int = 0, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies a 2D convolution operation.

   This layer performs a convolution operation, which is a fundamental building block
   in convolutional neural networks (CNNs). It slides a kernel (or filter) over
   an input tensor to produce an output tensor.

   :param in_channels: Number of input channels.
   :type in_channels: int
   :param out_channels: Number of output channels.
   :type out_channels: int
   :param kernel_size: Size of the convolutional kernel.
   :type kernel_size: int
   :param bias: Whether to include a learnable bias term. Defaults to True.
   :type bias: bool
   :param stride: The step size of the kernel. Defaults to 1.
   :type stride: int
   :param padding: Amount of zero-padding added to the input. Defaults to 0.
   :type padding: int
   :param padding_type: Type of padding. Defaults to 1 (likely 'zeros' or similar).
   :type padding_type: int
   :param in_width: Input width. If 0, it might be inferred or set by the backend. Defaults to 0.
   :type in_width: int
   :param in_height: Input height. If 0, it might be inferred or set by the backend. Defaults to 0.
   :type in_height: int
   :param gain_weight: Initial value for the gain (scale) parameter of weights. Defaults to 1.0.
   :type gain_weight: float
   :param gain_bias: Initial value for the gain (scale) parameter of biases. Defaults to 1.0.
   :type gain_bias: float
   :param init_method: Method used for initializing weights. Defaults to "He".
   :type init_method: str

   Initializes the Conv2d layer.


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the Conv2d layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the Conv2d layer.

      :returns: The name of the layer.
      :rtype: str



   .. py:method:: init_weight_bias()

      Initializes the learnable weight (kernel) and bias parameters of the convolutional layer.
      This initialization is delegated to the C++ backend using the 'init_method' specified (e.g., "He").
