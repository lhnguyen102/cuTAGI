pytagi.nn.linear
================

.. py:module:: pytagi.nn.linear

.. autoapi-nested-parse::

   Introduction about the script: Fully-connected layer etc ...
   Hello world



Classes
-------

.. autoapisummary::

   pytagi.nn.linear.Linear


Module Contents
---------------

.. py:class:: Linear(input_size: int, output_size: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Fully-connected layer

   :param input_size: Input size of the layer
   :type input_size: int

   .. attribute:: input_size

      Input size of the layer

      :type: int

   .. attribute:: output_size

      Output size of the layer

      :type: int

   .. attribute:: bias

      If True, adding biases along with the weights

      :type: boolen


   .. py:method:: get_layer_info() -> str

      get layer information



   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()
