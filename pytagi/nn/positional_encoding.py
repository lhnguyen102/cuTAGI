import cutagi

from pytagi.nn.base_layer import BaseLayer


class PositionalEncoding(BaseLayer):
    """Adds sinusoidal positional encoding to the input embeddings.

    This layer adds fixed sinusoidal positional encodings to the input,
    allowing the model to use position information. The backward pass
    is identity (jcb=1.0) since PE is a constant addition.
    """

    def __init__(self, embed_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self._cpp_backend = cutagi.PositionalEncoding(embed_dim, max_seq_len)

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()
