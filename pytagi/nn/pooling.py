import cutagi

from pytagi.nn.base_layer import BaseLayer


class AvgPool2d(BaseLayer):
    """2D Average Pooling Layer.

    This layer performs 2D average pooling operation. It wraps the C++/CUDA backend
    `cutagi.AvgPool2d`.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = -1,
        padding: int = 0,
        padding_type: int = 0,
    ):
        """Initializes the AvgPool2d layer.

        :param kernel_size: The size of the pooling window (a single integer for square kernels).
        :type kernel_size: int
        :param stride: The stride of the pooling operation. Default is -1, which typically means stride=kernel_size.
        :type stride: int
        :param padding: The implicit zero padding added to both sides of the input.
        :type padding: int
        :param padding_type: The type of padding to be used (e.g., 0 for zero padding).
        :type padding_type: int
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_type = padding_type

        self._cpp_backend = cutagi.AvgPool2d(
            kernel_size, stride, padding, padding_type
        )

    def get_layer_info(self) -> str:
        """Returns a string containing information about the layer."""
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """Returns the name of the layer (e.g., 'AvgPool2d')."""
        return self._cpp_backend.get_layer_name()


class MaxPool2d(BaseLayer):
    """2D Max Pooling Layer.

    This layer performs 2D max pooling operation based on the input expected values.
    It wraps the C++/CUDA backend `cutagi.MaxPool2d`.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        padding_type: int = 0,
    ):
        """Initializes the MaxPool2d layer.

        :param kernel_size: The size of the pooling window (a single integer for square kernels).
        :type kernel_size: int
        :param stride: The stride of the pooling operation. Default is 1.
        :type stride: int
        :param padding: The implicit zero padding added to both sides of the input.
        :type padding: int
        :param padding_type: The type of padding to be used (e.g., 0 for zero padding).
        :type padding_type: int
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_type = padding_type

        self._cpp_backend = cutagi.MaxPool2d(
            kernel_size, stride, padding, padding_type
        )

    def get_layer_info(self) -> str:
        """Returns a string containing information about the layer."""
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """Returns the name of the layer (e.g., 'MaxPool2d')."""
        return self._cpp_backend.get_layer_name()
