pytagi.nn.layer_norm
====================

.. py:module:: pytagi.nn.layer_norm


Classes
-------

.. autoapisummary::

   pytagi.nn.layer_norm.LayerNorm


Module Contents
---------------

.. py:class:: LayerNorm(normalized_shape: List[int], eps: float = 0.0001, bias: bool = True)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Layer normalization


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



   .. py:method:: init_weight_bias()
