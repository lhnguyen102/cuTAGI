from typing import List

import cutagi

from pytagi.nn.base_layer import BaseLayer


class RMSNorm(BaseLayer):
    """
    Implements Root Mean Square Layer Normalization (RMSNorm) by normalizing
    the inputs using the RMS statistic. It inherits from BaseLayer.
    """

    def __init__(self, normalized_shape: List[int], eps: float = 1e-6):
        """
        Initializes the RMSNorm layer.

        Args:
            normalized_shape: The shape of the input to normalize over (e.g.,
                              the size of the feature dimension). Expected to be
                              a list of integers.
            eps: A small value added to the denominator for numerical stability
                 to prevent division by zero. Defaults to 1e-6.
        """
        self.normalized_shape = normalized_shape
        self.eps = eps

        self._cpp_backend = cutagi.RMSNorm(normalized_shape, eps)

    def get_layer_info(self) -> str:
        """
        Retrieves a descriptive string containing information about the layer's
        configuration from the C++ backend.
        """
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """
        Retrieves the name of the layer (e.g., 'RMSNorm') from the C++ backend.
        """
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        """
        Initializes the layer's internal parameters, specifically the learnable
        scale (gamma). This task is delegated to the C++ backend.
        """
        self._cpp_backend.init_weight_bias()
