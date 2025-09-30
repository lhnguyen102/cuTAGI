import cutagi

from pytagi.nn.base_layer import BaseLayer


class LSTM(BaseLayer):
    """
    A **Long Short-Term Memory (LSTM)** layer for RNNs. It inherits from BaseLayer.
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
        """
        Initializes the LSTM layer.

        Args:
            input_size: The number of features in the input tensor at each time
                        step.
            output_size: The size of the hidden state (:math:`h_t`), which is the
                         number of features in the output tensor at each time
                         step.
            seq_len: The maximum length of the input sequence. This is often
                     required for efficient memory allocation in C++/CUDA
                     backends like cuTAGI.
            bias: If True, the internal gates and cell state updates will include
                  an additive bias vector. Defaults to True.
            gain_weight: Scaling factor applied to the initialized weights
                         (:math:`W`). Defaults to 1.0.
            gain_bias: Scaling factor applied to the initialized biases
                       (:math:`b`). Defaults to 1.0.
            init_method: The method used for initializing the weights and
                         biases (e.g., "He", "Xavier"). Defaults to "He".
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.bias = bias
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.init_method = init_method

        self._cpp_backend = cutagi.LSTM(
            input_size,
            output_size,
            seq_len,
            bias,
            gain_weight,
            gain_bias,
            init_method,
        )

    def get_layer_info(self) -> str:
        """
        Retrieves a descriptive string containing information about the layer's
        configuration (e.g., input/output size, sequence length) from the
        C++ backend.
        """
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """
        Retrieves the name of the layer (e.g., 'LSTM') from the C++ backend.
        """
        return self._cpp_backend.get_layer_name()

    def init_weight_bias(self):
        """
        Initializes the various weight matrices and bias vectors used by the
        LSTM's gates (input, forget, output) and cell state updates, using
        the specified method and gain factors. This task is delegated to the
        C++ backend.
        """
        self._cpp_backend.init_weight_bias()
