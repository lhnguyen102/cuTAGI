import cutagi

from pytagi.nn.base_layer import BaseLayer


class SLinear(BaseLayer):
    """Smoother Linear layer for the SLSTM architecture.

    This layer performs a linear transformation (:math:`y = xW^T + b'), specifically designed
    to be used within SLSTM where a hidden- and cell-state smoothing through time is applied.
    It wraps the C++/CUDA backend `cutagi.SLinear`.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gain_weight: float = 1.0,
        gain_bias: float = 1.0,
        init_method: str = "He",
    ):
        """Initializes the SLinear layer.

        :param input_size: The number of input features.
        :type input_size: int
        :param output_size: The number of output features.
        :type output_size: int
        :param bias: If ``True``, adds a learnable bias to the output.
        :type bias: bool
        :param gain_weight: A scaling factor applied to the initialized weights.
        :type gain_weight: float
        :param gain_bias: A scaling factor applied to the initialized bias terms.
        :type gain_bias: float
        :param init_method: The method used for initializing weights and biases (e.g., 'He', 'Xavier').
        :type init_method: str
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.init_method = init_method

        self._cpp_backend = cutagi.SLinear(
            input_size, output_size, bias, gain_weight, gain_bias, init_method
        )

    def get_layer_info(self) -> str:
        """Returns a string containing information about the layer's configuration (sizes, bias, etc.)."""
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """Returns the name of the layer (e.g., 'SLinear')."""
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        """Initializes the layer's weight matrix and bias vector based on the configured method."""
        self._cpp_backend.init_weight_bias()
