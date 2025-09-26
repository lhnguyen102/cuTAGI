pytagi.nn.slstm
===============

.. py:module:: pytagi.nn.slstm


Classes
-------

.. autoapisummary::

   pytagi.nn.slstm.SLSTM


Module Contents
---------------

.. py:class:: SLSTM(input_size: int, output_size: int, seq_len: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Smoothing Long Short-Term Memory (LSTM) layer.

   This layer is a variation of the standard LSTM, likely incorporating a mechanism
   for **smoothing** the hidden states or outputs. It's designed for sequence
   processing tasks. It wraps the C++/CUDA backend `cutagi.SLSTM`.


   .. py:method:: get_layer_info() -> str

      Returns a string containing detailed information about the layer's configuration.



   .. py:method:: get_layer_name() -> str

      Returns the name of the layer (e.g., 'SLSTM').



   .. py:method:: init_weight_bias()

      Initializes all the layer's internal weight matrices and bias vectors (for gates and cell) based on the configured method.
