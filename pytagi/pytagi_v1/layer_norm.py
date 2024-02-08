# Temporary import. It will be removed in the final verion
import sys
import os
from typing import List

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

import cutagitest
from base_layer import BaseLayer


class LayerNorm(BaseLayer):
    """Layer normalization"""

    def __init__(
        self,
        normalized_shape: List[int],
        eps: float = 1e-4,
        momentum: float = 0.9,
        bias: bool = True,
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.is_bias = bias

        self._cpp_backend = cutagitest.LayerNorm(normalized_shape, eps, momentum, bias)

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        self._cpp_backend.init_weight_bias()
