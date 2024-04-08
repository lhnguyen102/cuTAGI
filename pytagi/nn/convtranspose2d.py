import cutagi

from pytagi.nn.base_layer import BaseLayer


class ConvTranspose2d(BaseLayer):
    """Tranposed convolutional layer"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        stride: int = 1,
        padding: int = 0,
        padding_type: int = 1,
        in_width: int = 0,
        in_height: int = 0,
        gain_weight: float = 1.0,
        gain_bias: float = 1.0,
        init_method: str = "He",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.is_bias = bias
        self.stride = stride
        self.padding = padding
        self.padding_type = padding_type
        self.in_width = in_width
        self.in_height = in_height
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.init_method = init_method

        self._cpp_backend = cutagi.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            stride,
            padding,
            padding_type,
            in_width,
            in_height,
            gain_weight,
            gain_bias,
            init_method,
        )

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        self._cpp_backend.init_weight_bias()
