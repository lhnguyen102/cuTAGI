pytagi.nn.convtranspose2d
=========================

.. py:module:: pytagi.nn.convtranspose2d


Classes
-------

.. autoapisummary::

   pytagi.nn.convtranspose2d.ConvTranspose2d


Module Contents
---------------

.. py:class:: ConvTranspose2d(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, stride: int = 1, padding: int = 0, padding_type: int = 1, in_width: int = 0, in_height: int = 0, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Tranposed convolutional layer


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()
