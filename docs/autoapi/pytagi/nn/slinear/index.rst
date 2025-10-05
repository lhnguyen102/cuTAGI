pytagi.nn.slinear
=================

.. py:module:: pytagi.nn.slinear


Classes
-------

.. autoapisummary::

   pytagi.nn.slinear.SLinear


Module Contents
---------------

.. py:class:: SLinear(input_size: int, output_size: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Smoother Linear layer for the SLSTM architecture.

   This layer performs a linear transformation (:math:`y = xW^T + b'), specifically designed
   to be used within SLSTM where a hidden- and cell-state smoothing through time is applied.
   It wraps the C++/CUDA backend `cutagi.SLinear`.


   .. py:method:: get_layer_info() -> str

      Returns a string containing information about the layer's configuration (sizes, bias, etc.).



   .. py:method:: get_layer_name() -> str

      Returns the name of the layer (e.g., 'SLinear').



   .. py:method:: init_weight_bias()

      Initializes the layer's weight matrix and bias vector based on the configured method.
