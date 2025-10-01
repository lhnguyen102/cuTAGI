import cutagi

from pytagi.nn.base_layer import BaseLayer


class SLSTM(BaseLayer):
    """Smoothing Long Short-Term Memory (LSTM) layer.

    This layer is a variation of the standard LSTM, incorporating a mechanism
    for **smoothing** the hidden- and cell-states. It wraps the C++/CUDA backend
    `cutagi.SLSTM`.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        seq_len: int,
        bias: bool = True,
        gain_weight: float = 1.0,
        gain_bias: float = 1.0,
        init_method: str = "He",
    ):
        """Initializes the SLSTM layer.

        :param input_size: The number of expected features in the input $x$.
        :type input_size: int
        :param output_size: The number of features in the hidden state $h$ (and the output).
        :type output_size: int
        :param seq_len: The maximum sequence length this layer is configured to handle.
        :type seq_len: int
        :param bias: If ``True``, use bias weights in the internal linear transformations.
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
        self.seq_len = seq_len
        self.bias = bias
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.init_method = init_method

        self._cpp_backend = cutagi.SLSTM(
            input_size,
            output_size,
            seq_len,
            bias,
            gain_weight,
            gain_bias,
            init_method,
        )

    def get_layer_info(self) -> str:
        """Returns a string containing detailed information about the layer's configuration."""
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """Returns the name of the layer (e.g., 'SLSTM')."""
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        """Initializes all the layer's internal weight matrices and bias vectors (for gates and cell) based on the configured method."""
        self._cpp_backend.init_weight_bias()
