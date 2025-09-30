from typing import List

import cutagi

from pytagi.nn.base_layer import BaseLayer


class LayerNorm(BaseLayer):
    """
    Implements Layer Normalization by normalizing the inputs across the
    features dimension. It inherits from BaseLayer.
    """

    def __init__(
        self, normalized_shape: List[int], eps: float = 1e-4, bias: bool = True
    ):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape: The shape of the input to normalize over (e.g.,
                              the size of the feature dimension). Expected to be
                              a list of integers.
            eps: A small value added to the denominator for numerical stability
                 to prevent division by zero. Defaults to 1e-4.
            bias: If True, the layer will use an additive bias (beta) during
                  normalization. Defaults to True.
        """
        # Store Python-side attributes for configuration
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.is_bias = bias

        self._cpp_backend = cutagi.LayerNorm(normalized_shape, eps, bias)

    def get_layer_info(self) -> str:
        """
        Retrieves a descriptive string containing information about the layer's
        configuration (e.g., its shape and parameters) from the C++ backend.
        """
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """
        Retrieves the name of the layer (e.g., 'LayerNorm') from the C++ backend.
        """
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        """
        Initializes the layer's internal parameters, specifically the learnable
        scale (gamma) and, if 'bias' is True, the learnable offset (beta).
        This task is delegated to the C++ backend.
        """
        self._cpp_backend.init_weight_bias()
