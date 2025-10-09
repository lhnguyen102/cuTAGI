pytagi.nn.embedding
===================

.. py:module:: pytagi.nn.embedding


Classes
-------

.. autoapisummary::

   pytagi.nn.embedding.Embedding


Module Contents
---------------

.. py:class:: Embedding(num_embeddings: int, embedding_dim: int, input_size: int = 0, scale: float = 1.0, padding_idx: int = -1)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Embedding layer

   The embedding layer maps discrete categorical indices to continuous vector representations.

    Args:
       num_embeddings (int): The size of the vocabulary (the total number of possible indices).
       embedding_dim (int): The dimensionality of the embedding vectors.
       input_size (int): The size of the input sequence. Defaults to 0.
       scale (float): A scaling factor applied to the embedding vectors. Defaults to 1.0.
       padding_idx (int): If specified, the embedding vector at this index is initialized
                          to zeros and is not updated during training. Defaults to -1 (disabled).

   Initializes the Embedding layer.


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the Embedding layer.

      :returns: A string containing the layer's configuration.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the Embedding layer.

      :returns: The name of the layer.
      :rtype: str



   .. py:method:: init_weight_bias()

      Initializes the embedding matrix (the learnable weights of the layer).
