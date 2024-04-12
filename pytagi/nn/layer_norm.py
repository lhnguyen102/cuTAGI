from typing import List

import cutagi

from pytagi.nn.base_layer import BaseLayer


class LayerNorm(BaseLayer):
    """Layer normalization"""

    def __init__(
        self, normalized_shape: List[int], eps: float = 1e-4, bias: bool = True
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.is_bias = bias

        self._cpp_backend = cutagi.LayerNorm(normalized_shape, eps, bias)

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        self._cpp_backend.init_weight_bias()
