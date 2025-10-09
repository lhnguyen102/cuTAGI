import cutagi

from pytagi.nn.base_layer import BaseLayer


class Embedding(BaseLayer):
    """
    Embedding layer

    The embedding layer maps discrete categorical indices to continuous vector representations.

    Args:
        num_embeddings (int): The size of the vocabulary (the total number of possible indices).
        embedding_dim (int): The dimensionality of the embedding vectors.
        input_size (int): The size of the input sequence. Defaults to 0.
        scale (float): A scaling factor applied to the embedding vectors. Defaults to 1.0.
        padding_idx (int): If specified, the embedding vector at this index is initialized
                           to zeros and is not updated during training. Defaults to -1 (disabled).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        input_size: int = 0,
        scale: float = 1.0,
        padding_idx: int = -1,
    ):
        """Initializes the Embedding layer."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.scale = scale
        self.padding_idx = padding_idx

        self._cpp_backend = cutagi.Embedding(
            num_embeddings, embedding_dim, input_size, scale, padding_idx
        )

    def get_layer_info(self) -> str:
        """
        Retrieves detailed information about the Embedding layer.

        Returns:
            str: A string containing the layer's configuration.
        """
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """
        Retrieves the name of the Embedding layer.

        Returns:
            str: The name of the layer.
        """
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        """
        Initializes the embedding matrix (the learnable weights of the layer).
        """
        self._cpp_backend.init_weight_bias()
