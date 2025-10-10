import cutagi

from pytagi.nn.base_layer import BaseLayer


class Linear(BaseLayer):
    """
    Implements a **Fully-connected layer**, also known as a dense layer.
    This layer performs a linear transformation on the input data:
    :math:`y = xW^T + b`, where :math:`x` is the input, :math:`W` is the weight matrix,
    and :math:`b` is the optional bias vector. It inherits from BaseLayer.
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
        """
        Initializes the Linear layer.

        Args:
            input_size: The number of features in the input tensor (the
                        size of the last dimension).
            output_size: The number of features in the output tensor. This
                         determines the number of neurons in the layer.
            bias: If True, an additive bias vector 'b' is included in the
                  linear transformation. Defaults to True.
            gain_weight: Scaling factor applied to the initialized weights
                         (:math:`W`). Defaults to 1.0.
            gain_bias: Scaling factor applied to the initialized biases
                       (:math:`b`). Defaults to 1.0.
            init_method: The method used for initializing the weights and
                         biases (e.g., "He", "Xavier", "Normal"). Defaults
                         to "He".
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.init_method = init_method

        self._cpp_backend = cutagi.Linear(
            input_size, output_size, bias, gain_weight, gain_bias, init_method
        )

    def get_layer_info(self) -> str:
        """
        Retrieves a descriptive string containing information about the layer's
        configuration (e.g., input/output size, whether bias is used) from the
        C++ backend.
        """
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """
        Retrieves the name of the layer (e.g., 'Linear')
        from the C++ backend.
        """
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        """
        Initializes the layer's parameters—the weight matrix (:math:`W`) and the
        optional bias vector (:math:`b`)—using the specified initialization method
        and gain factors. This task is delegated to the C++ backend.
        """
        self._cpp_backend.init_weight_bias()
