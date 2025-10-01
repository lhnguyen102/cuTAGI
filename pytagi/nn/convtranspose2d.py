import cutagi

from pytagi.nn.base_layer import BaseLayer


class ConvTranspose2d(BaseLayer):
    """
    Applies a 2D transposed convolution operation (also known as deconvolution).

    This layer performs a transposed convolution, which is used in tasks
    like image generation or segmentation to upsample feature maps. It
    reverses the convolution operation, increasing the spatial dimensions of the input.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        bias (bool): Whether to include a learnable bias term. Defaults to True.
        stride (int): The step size of the kernel. Defaults to 1.
        padding (int): Amount of zero-padding added to the input. Defaults to 0.
        padding_type (int): Type of padding. Defaults to 1 (likely 'zeros' or similar).
        in_width (int): Input width. If 0, it might be inferred or set by the backend. Defaults to 0.
        in_height (int): Input height. If 0, it might be inferred or set by the backend. Defaults to 0.
        gain_weight (float): Initial value for the gain (scale) parameter of weights. Defaults to 1.0.
        gain_bias (float): Initial value for the gain (scale) parameter of biases. Defaults to 1.0.
        init_method (str): Method used for initializing weights. Defaults to "He".
    """

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
        """Initializes the ConvTranspose2d layer."""
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
        """
        Retrieves detailed information about the ConvTranspose2d layer.

        Returns:
            str: A string containing the layer's information.
        """
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """
        Retrieves the name of the ConvTranspose2d layer.

        Returns:
            str: The name of the layer.
        """
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        """
        Initializes the learnable weight and bias parameters of the transposed convolutional layer.
        """
        self._cpp_backend.init_weight_bias()
