import cutagi

from pytagi.nn.base_layer import BaseLayer


class PositionalEncoding(BaseLayer):
    """Adds fixed sinusoidal positional encodings to token embeddings.

    PE[t, d] = sin(t / 10000^(2*floor(d/2)/embed_dim)) for even d, cos for
    odd d. The variance passes through unchanged and the backward pass is the
    identity, because PE is a constant.

    Args:
        embed_dim: Size of each token's feature vector. Must match the input.
        max_seq_len: Longest sequence the layer accepts. Defaults to 2048.
        device_idx: CUDA device index. Defaults to 0.
    """

    def __init__(
        self, embed_dim: int, max_seq_len: int = 2048, device_idx: int = 0
    ):
        super().__init__()
        self._cpp_backend = cutagi.PositionalEncoding(
            embed_dim, max_seq_len, device_idx
        )

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()
