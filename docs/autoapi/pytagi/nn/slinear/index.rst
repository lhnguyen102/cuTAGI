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


   Smoothering Linear layer


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



   .. py:method:: init_weight_bias()
