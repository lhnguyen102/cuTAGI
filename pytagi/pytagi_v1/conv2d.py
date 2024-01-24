# Temporary import. It will be removed in the final vserion
import sys
import os

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

import cutagitest
from base_layer import BaseLayer


class Conv2d(BaseLayer):
    """Fully-connected layer"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        padding_type: int = 1,
        in_width: int = 0,
        in_height: int = 0,
        gain_weight: float = 1.0,
        gain_bias: float = 1.0,
        init_method: str = "He",
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_type = padding_type
        self.in_width = in_width
        self.in_height = in_height
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.init_method = init_method

        self._cpp_backend = cutagitest.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_type,
            in_width,
            in_height,
            gain_weight,
            gain_bias,
            init_method,
            bias,
        )

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        self._cpp_backend.init_weight_bias()
