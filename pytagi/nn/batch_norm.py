from typing import List

import cutagi

from pytagi.nn.base_layer import BaseLayer


class BatchNorm2d(BaseLayer):
    """Layer normalization"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-4,
        momentum: float = 0.9,
        bias: bool = True,
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.is_bias = bias
        self._cpp_backend = cutagi.BatchNorm2d(num_features, eps, momentum, bias)

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        self._cpp_backend.init_weight_bias()
