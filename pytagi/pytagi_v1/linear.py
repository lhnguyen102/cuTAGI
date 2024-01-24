# Temporary import. It will be removed in the final vserion
import sys
import os

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

import cutagitest
from base_layer import BaseLayer


class Linear(BaseLayer):
    """Fully-connected layer"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        gain_weight: float = 1.0,
        gain_bias: float = 1.0,
        init_method: str = "He",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.init_method = init_method

        self._cpp_backend = cutagitest.Linear(
            input_size, output_size, gain_weight, gain_bias, init_method
        )

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        self._cpp_backend.init_weight_bias()
