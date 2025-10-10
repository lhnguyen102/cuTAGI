from typing import List

import cutagi

from pytagi.nn.base_layer import BaseLayer


class BatchNorm2d(BaseLayer):
    """
    Applies 2D Batch Normalization.

    Batch Normalization normalizes the inputs of a layer by re-centering and
    re-scaling them.

    Args:
        num_features (int): The number of features in the input tensor.
        eps (float): A small value added to the variance to avoid division by zero.
                     Defaults to 1e-5.
        momentum (float): The momentum for the running mean and variance.
                          Defaults to 0.9.
        bias (bool): Whether to include a learnable bias term. Defaults to True.
        gain_weight (float): Initial value for the gain (scale) parameter. Defaults to 1.0.
        gain_bias (float): Initial value for the bias (shift) parameter. Defaults to 1.0.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.9,
        bias: bool = True,
        gain_weight: float = 1.0,
        gain_bias: float = 1.0,
    ):
        """Initializes the BatchNorm2d layer."""
        # Store essential configuration parameters as instance attributes.
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.is_bias = bias

        self._cpp_backend = cutagi.BatchNorm2d(
            num_features, eps, momentum, bias, gain_weight, gain_bias
        )

    def get_layer_info(self) -> str:
        """
        Retrieves detailed information about the BatchNorm2d layer.

        Returns:
            str: A string containing the layer's information, typically delegated
                 to the C++ backend implementation.
        """
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """
        Retrieves the name of the BatchNorm2d layer.

        Returns:
            str: The name of the layer, typically delegated to the C++ backend implementation.
        """
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        """
        Initializes the learnable weight (scale/gain) and bias (shift/offset)
        parameters of the batch normalization layer. This operation is
        delegated to the C++ backend.
        """
        self._cpp_backend.init_weight_bias()
