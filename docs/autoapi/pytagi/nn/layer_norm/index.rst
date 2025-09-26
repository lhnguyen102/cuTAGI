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


   Implements Layer Normalization, a technique often used in neural networks
   to stabilize the learning process by normalizing the inputs across the
   features dimension. It inherits from BaseLayer.


   .. py:method:: get_layer_info() -> str

      Retrieves a descriptive string containing information about the layer's
      configuration (e.g., its shape and parameters) from the C++ backend.



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer (e.g., 'LayerNorm') from the C++ backend.



   .. py:method:: init_weight_bias()

      Initializes the layer's internal parameters, specifically the learnable
      scale (gamma) and, if 'bias' is True, the learnable offset (beta).
      This task is delegated to the C++ backend.
