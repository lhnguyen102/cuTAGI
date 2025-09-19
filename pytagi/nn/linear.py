"""
Introduction about the script: Fully-connected layer etc ...
Hello world

"""

import cutagi

from pytagi.nn.base_layer import BaseLayer


class Linear(BaseLayer):
    """
    Fully-connected layer

    Args:
        input_size (int): Input size of the layer

    Attributes:
        input_size (int): Input size of the layer
        output_size (int): Output size of the layer
        bias (boolen): If True, adding biases along with the weights
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
        get layer information
        """
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        self._cpp_backend.init_weight_bias()
