# Temporary import. It will be removed in the final vserion
import sys
import os

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

import cutagitest
from base_layer import BaseLayer


class AvgPool2d(BaseLayer):
    """Fully-connected layer"""

    def __init__(
        self,
        kernel_size: int,
        stride: int = -1,
        padding: int = 0,
        padding_type: int = 0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_type = padding_type

        self._cpp_backend = cutagitest.AvgPool2d(
            kernel_size, stride, padding, padding_type
        )

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()
