pytagi.nn.lstm
==============

.. py:module:: pytagi.nn.lstm


Classes
-------

.. autoapisummary::

   pytagi.nn.lstm.LSTM


Module Contents
---------------

.. py:class:: LSTM(input_size: int, output_size: int, seq_len: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   A **Long Short-Term Memory (LSTM)** layer for RNNs. It inherits from BaseLayer.

   Initializes the LSTM layer.

   :param input_size: The number of features in the input tensor at each time
                      step.
   :param output_size: The size of the hidden state (:math:`h_t`), which is the
                       number of features in the output tensor at each time
                       step.
   :param seq_len: The maximum length of the input sequence. This is often
                   required for efficient memory allocation in C++/CUDA
                   backends like cuTAGI.
   :param bias: If True, the internal gates and cell state updates will include
                an additive bias vector. Defaults to True.
   :param gain_weight: Scaling factor applied to the initialized weights
                       (:math:`W`). Defaults to 1.0.
   :param gain_bias: Scaling factor applied to the initialized biases
                     (:math:`b`). Defaults to 1.0.
   :param init_method: The method used for initializing the weights and
                       biases (e.g., "He", "Xavier"). Defaults to "He".


   .. py:method:: get_layer_info() -> str

      Retrieves a descriptive string containing information about the layer's
      configuration (e.g., input/output size, sequence length) from the
      C++ backend.



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer (e.g., 'LSTM') from the C++ backend.



   .. py:method:: init_weight_bias()

      Initializes the various weight matrices and bias vectors used by the
      LSTM's gates (input, forget, output) and cell state updates, using
      the specified method and gain factors. This task is delegated to the
      C++ backend.
