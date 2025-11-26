import cutagi

from pytagi.nn.base_layer import BaseLayer


class MultiheadAttention(BaseLayer):
    """
    Implements a **Multi-head Attention layer** with uncertainty quantification.
    This layer applies scaled dot-product attention with multiple attention heads,
    allowing the model to jointly attend to information from different representation
    subspaces. It inherits from BaseLayer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int = None,
        bias: bool = True,
        gain_weight: float = 1.0,
        gain_bias: float = 1.0,
        init_method: str = "Xavier",
    ):
        """
        Initializes the MultiheadAttention layer.

        Args:
            embed_dim: The dimensionality of the input embeddings and output.
            num_heads: The number of attention heads.
            num_kv_heads: The number of key-value heads for grouped-query attention.
                         If None, defaults to num_heads (standard multi-head attention).
            bias: If True, additive bias is included in the linear projections.
                  Defaults to True.
            gain_weight: Scaling factor applied to initialized weights. Defaults to 1.0.
            gain_bias: Scaling factor applied to initialized biases. Defaults to 1.0.
            init_method: The method used for initializing weights and biases
                        (e.g., "Xavier", "He"). Defaults to "Xavier".
        """
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.bias = bias
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.init_method = init_method

        self._cpp_backend = cutagi.MultiheadAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            bias,
            gain_weight,
            gain_bias,
            init_method,
        )

    def get_layer_info(self) -> str:
        """
        Retrieves a descriptive string containing information about the layer's
        configuration from the C++ backend.
        """
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """
        Retrieves the name of the layer from the C++ backend.
        """
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        """
        Initializes the layer's parameters for query, key, and value projections
        using the specified initialization method and gain factors.
        This task is delegated to the C++ backend.
        """
        self._cpp_backend.init_weight_bias()
