pytagi.nn.base_layer
====================

.. py:module:: pytagi.nn.base_layer


Classes
-------

.. autoapisummary::

   pytagi.nn.base_layer.BaseLayer


Module Contents
---------------

.. py:class:: BaseLayer

   Base layer


   .. py:method:: to_cuda()


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: get_max_num_states() -> int


   .. py:property:: input_size
      :type: int



   .. py:property:: output_size
      :type: int



   .. py:property:: in_width
      :type: int



   .. py:property:: in_height
      :type: int



   .. py:property:: in_channels
      :type: int



   .. py:property:: out_width
      :type: int



   .. py:property:: out_height
      :type: int



   .. py:property:: out_channels
      :type: int



   .. py:property:: bias
      :type: bool



   .. py:property:: num_weights
      :type: int



   .. py:property:: num_biases
      :type: int



   .. py:property:: mu_w
      :type: numpy.ndarray



   .. py:property:: var_w
      :type: numpy.ndarray



   .. py:property:: mu_b
      :type: numpy.ndarray



   .. py:property:: var_b
      :type: numpy.ndarray



   .. py:property:: delta_mu_w
      :type: numpy.ndarray



   .. py:property:: delta_var_w
      :type: numpy.ndarray



   .. py:property:: delta_mu_b
      :type: numpy.ndarray



   .. py:property:: delta_var_b
      :type: numpy.ndarray



   .. py:property:: num_threads
      :type: int



   .. py:property:: training
      :type: bool



   .. py:property:: device
      :type: bool
