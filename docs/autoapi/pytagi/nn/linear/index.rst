pytagi.nn.linear
================

.. py:module:: pytagi.nn.linear


Classes
-------

.. autoapisummary::

   pytagi.nn.linear.Linear


Module Contents
---------------

.. py:class:: Linear(input_size: int, output_size: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Implements a **Fully-connected layer**, also known as a dense layer.
   This layer performs a linear transformation on the input data:
   :math:`y = xW^T + b`, where :math:`x` is the input, :math:`W` is the weight matrix,
   and :math:`b` is the optional bias vector. It inherits from BaseLayer.

   Initializes the Linear layer.

   :param input_size: The number of features in the input tensor (the
                      size of the last dimension).
   :param output_size: The number of features in the output tensor. This
                       determines the number of neurons in the layer.
   :param bias: If True, an additive bias vector 'b' is included in the
                linear transformation. Defaults to True.
   :param gain_weight: Scaling factor applied to the initialized weights
                       (:math:`W`). Defaults to 1.0.
   :param gain_bias: Scaling factor applied to the initialized biases
                     (:math:`b`). Defaults to 1.0.
   :param init_method: The method used for initializing the weights and
                       biases (e.g., "He", "Xavier", "Normal"). Defaults
                       to "He".


   .. py:method:: get_layer_info() -> str

      Retrieves a descriptive string containing information about the layer's
      configuration (e.g., input/output size, whether bias is used) from the
      C++ backend.



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer (e.g., 'Linear')
      from the C++ backend.



   .. py:method:: init_weight_bias()

      Initializes the layer's parameters—the weight matrix (:math:`W`) and the
      optional bias vector (:math:`b`)—using the specified initialization method
      and gain factors. This task is delegated to the C++ backend.
