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

   :param num_embeddings: The size of the vocabulary (the total number of possible indices).
   :type num_embeddings: int
   :param embedding_dim: The dimensionality of the embedding vectors.
   :type embedding_dim: int
   :param input_size: The size of the input sequence. Defaults to 0.
   :type input_size: int
   :param scale: A scaling factor applied to the embedding vectors. Defaults to 1.0.
   :type scale: float
   :param padding_idx: If specified, the embedding vector at this index is initialized
                       to zeros and is not updated during training. Defaults to -1 (disabled).
   :type padding_idx: int

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
