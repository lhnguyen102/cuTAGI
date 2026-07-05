import cutagi

from pytagi.nn.base_layer import BaseLayer


def _validate_attention_config(
    embed_dim: int, num_heads: int, num_kv_heads: int, pos_emb: str
):
    """Reject configurations the C++/CUDA attention kernels do not support."""
    if num_kv_heads != num_heads:
        raise ValueError(
            "Grouped-query attention is not supported: num_kv_heads "
            f"({num_kv_heads}) must equal num_heads ({num_heads})"
        )
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim ({embed_dim}) must be divisible by num_heads "
            f"({num_heads})"
        )
    if pos_emb == "rope" and (embed_dim // num_heads) % 2 != 0:
        raise ValueError(
            f"RoPE requires an even head_dim: got {embed_dim // num_heads} "
            f"(embed_dim={embed_dim}, num_heads={num_heads})"
        )


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
        seq_len: int = 1,
        bias: bool = True,
        gain_weight: float = 1.0,
        gain_bias: float = 1.0,
        init_method: str = "He",
        pos_emb: str = "rope",
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        use_causal_mask: bool = True,
        prior_pull: float = 0.0,
        center_score_delta: bool = False,
        debug: bool = False,
        debug_interval: int = 1,
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
            pos_emb: Positional embedding method. Options: "rope" (rotary),
                     "sinusoidal" (Attention Is All You Need), or "" (none).
            rope_theta: Base frequency for RoPE. Only used when pos_emb="rope".
            max_seq_len: Maximum sequence length for positional encoding cache.
            use_causal_mask: If True, apply causal (lower-triangular) mask to
                            prevent attending to future positions. Defaults to True.
        """
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads
        _validate_attention_config(embed_dim, num_heads, num_kv_heads, pos_emb)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.seq_len = seq_len
        self.bias = bias
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.init_method = init_method
        self.pos_emb = pos_emb
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.use_causal_mask = use_causal_mask

        self._cpp_backend = cutagi.MultiheadAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            seq_len,
            bias,
            gain_weight,
            gain_bias,
            init_method,
            pos_emb,
            rope_theta,
            max_seq_len,
            use_causal_mask,
        )
        self._cpp_backend.debug = debug
        self._cpp_backend.debug_interval = debug_interval
        self._cpp_backend.prior_pull = prior_pull  # prior_mu = 0 for W_qkv
        self._cpp_backend.center_score_delta = center_score_delta

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

    @property
    def debug(self) -> bool:
        return self._cpp_backend.debug

    @debug.setter
    def debug(self, value: bool):
        self._cpp_backend.debug = value

    @property
    def debug_interval(self) -> int:
        return self._cpp_backend.debug_interval

    @debug_interval.setter
    def debug_interval(self, value: int):
        self._cpp_backend.debug_interval = value


class MultiheadAttentionV2(BaseLayer):
    """Multi-head Attention with separate Q, K, V projections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int = None,
        seq_len: int = 1,
        bias: bool = True,
        gain_weight: float = 1.0,
        gain_bias: float = 1.0,
        init_method: str = "He",
        pos_emb: str = "rope",
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        use_causal_mask: bool = True,
        debug: bool = False,
        debug_interval: int = 1,
    ):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads
        _validate_attention_config(embed_dim, num_heads, num_kv_heads, pos_emb)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.seq_len = seq_len
        self.bias = bias
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.init_method = init_method
        self.pos_emb = pos_emb
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.use_causal_mask = use_causal_mask

        self._cpp_backend = cutagi.MultiheadAttentionV2(
            embed_dim,
            num_heads,
            num_kv_heads,
            seq_len,
            bias,
            gain_weight,
            gain_bias,
            init_method,
            pos_emb,
            rope_theta,
            max_seq_len,
            use_causal_mask,
        )
        self._cpp_backend.debug = debug
        self._cpp_backend.debug_interval = debug_interval

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        self._cpp_backend.init_weight_bias()

    @property
    def debug(self) -> bool:
        return self._cpp_backend.debug

    @debug.setter
    def debug(self, value: bool):
        self._cpp_backend.debug = value

    @property
    def debug_interval(self) -> int:
        return self._cpp_backend.debug_interval

    @debug_interval.setter
    def debug_interval(self, value: int):
        self._cpp_backend.debug_interval = value
