import cutagi

from pytagi.nn.base_layer import BaseLayer


class Embedding(BaseLayer):
    """Embedding layer"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 1.0,
        padding_idx: int = -1,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.padding_idx = padding_idx

        self._cpp_backend = cutagi.Embedding(
            num_embeddings, embedding_dim, scale, padding_idx
        )

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        self._cpp_backend.init_weight_bias()
