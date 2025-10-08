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


   Implements Layer Normalization by normalizing the inputs across the
   features dimension. It inherits from BaseLayer.

   Initializes the LayerNorm layer.

   :param normalized_shape: The shape of the input to normalize over (e.g.,
                            the size of the feature dimension). Expected to be
                            a list of integers.
   :param eps: A small value added to the denominator for numerical stability
               to prevent division by zero. Defaults to 1e-4.
   :param bias: If True, the layer will use an additive bias (beta) during
                normalization. Defaults to True.


   .. py:method:: get_layer_info() -> str

      Retrieves a descriptive string containing information about the layer's
      configuration (e.g., its shape and parameters) from the C++ backend.



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer (e.g., 'LayerNorm') from the C++ backend.



   .. py:method:: init_weight_bias()

      Initializes the layer's internal parameters, specifically the learnable
      scale (gamma) and, if 'bias' is True, the learnable offset (beta).
      This task is delegated to the C++ backend.
