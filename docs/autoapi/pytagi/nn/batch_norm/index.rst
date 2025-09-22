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


   Batch normalization


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()
