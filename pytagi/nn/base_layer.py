import cutagi
import numpy as np


class BaseLayer:
    """
    Base layer class providing common functionality and properties for neural network layers.
    This class acts as a Python wrapper for the C++ backend, exposing layer attributes
    and methods for managing layer information, device placement, and parameters.
    """

    def __init__(self):
        """
        Initializes the BaseLayer with a C++ backend instance.
        """
        self._cpp_backend = cutagi.BaseLayer()

    def to_cuda(self):
        """
        Moves the layer's parameters and computations to the CUDA device.
        """
        self._cpp_backend.to_cuda()

    def get_layer_info(self) -> str:
        """
        Retrieves detailed information about the layer.

        Returns:
            str: A string containing the layer's information.
        """
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        """
        Retrieves the name of the layer.

        Returns:
            str: The name of the layer.
        """
        return self._cpp_backend.get_layer_name()

    def get_max_num_states(self) -> int:
        """
        Retrieves the maximum number of states the layer can hold.

        Returns:
            int: The maximum number of states.
        """
        return self._cpp_backend.get_max_num_states()

    @property
    def input_size(self) -> int:
        """
        Gets the input size of the layer.
        """
        return self._cpp_backend.input_size

    @input_size.setter
    def input_size(self, value: int):
        """
        Sets the input size of the layer.
        """
        self._cpp_backend.input_size = value

    @property
    def output_size(self) -> int:
        """
        Gets the output size of the layer.
        """
        return self._cpp_backend.output_size

    @output_size.setter
    def output_size(self, value: int):
        """
        Sets the output size of the layer.
        """
        self._cpp_backend.output_size = value

    @property
    def in_width(self) -> int:
        """
        Gets the input width of the layer (for convolutional layers).
        """
        return self._cpp_backend.in_width

    @in_width.setter
    def in_width(self, value: int):
        """
        Sets the input width of the layer (for convolutional layers).
        """
        self._cpp_backend.in_width = value

    @property
    def in_height(self) -> int:
        """
        Gets the input height of the layer (for convolutional layers).
        """
        return self._cpp_backend.in_height

    @in_height.setter
    def in_height(self, value: int):
        """
        Sets the input height of the layer (for convolutional layers).
        """
        self._cpp_backend.in_height = value

    @property
    def in_channels(self) -> int:
        """
        Gets the input channels of the layer (for convolutional layers).
        """
        return self._cpp_backend.in_channels

    @in_channels.setter
    def in_channels(self, value: int):
        """
        Sets the input channels of the layer (for convolutional layers).
        """
        self._cpp_backend.in_channels = value

    @property
    def out_width(self) -> int:
        """
        Gets the output width of the layer (for convolutional layers).
        """
        return self._cpp_backend.out_width

    @out_width.setter
    def out_width(self, value: int):
        """
        Sets the output width of the layer (for convolutional layers).
        """
        self._cpp_backend.out_width = value

    @property
    def out_height(self) -> int:
        """
        Gets the output height of the layer (for convolutional layers).
        """
        return self._cpp_backend.out_height

    @out_height.setter
    def out_height(self, value: int):
        """
        Sets the output height of the layer (for convolutional layers).
        """
        self._cpp_backend.out_height = value

    @property
    def out_channels(self) -> int:
        """
        Gets the output channels of the layer (for convolutional layers).
        """
        return self._cpp_backend.out_channels

    @out_channels.setter
    def out_channels(self, value: int):
        """
        Sets the output channels of the layer (for convolutional layers).
        """
        self._cpp_backend.out_channels = value

    @property
    def bias(self) -> bool:
        """
        Gets a boolean indicating whether the layer has a bias term.
        """
        return self._cpp_backend.bias

    @bias.setter
    def bias(self, value: bool):
        """
        Sets a boolean indicating whether the layer has a bias term.
        """
        self._cpp_backend.bias = value

    @property
    def num_weights(self) -> int:
        """
        Gets the total number of weights in the layer.
        """
        return self._cpp_backend.num_weights

    @num_weights.setter
    def num_weights(self, value: int):
        """
        Sets the total number of weights in the layer.
        """
        self._cpp_backend.num_weights = value

    @property
    def num_biases(self) -> int:
        """
        Gets the total number of biases in the layer.
        """
        return self._cpp_backend.num_biases

    @num_biases.setter
    def num_biases(self, value: int):
        """
        Sets the total number of biases in the layer.
        """
        self._cpp_backend.num_biases = value

    @property
    def mu_w(self) -> np.ndarray:
        """
        Gets the mean of the weights (mu_w) as a NumPy array.
        """
        return self._cpp_backend.mu_w

    @mu_w.setter
    def mu_w(self, value: np.ndarray):
        """
        Sets the mean of the weights (mu_w) as a NumPy array.
        """
        self._cpp_backend.mu_w = value

    @property
    def var_w(self) -> np.ndarray:
        """
        Gets the variance of the weights (var_w) as a NumPy array.
        """
        return self._cpp_backend.var_w

    @var_w.setter
    def var_w(self, value: np.ndarray):
        """
        Sets the variance of the weights (var_w) as a NumPy array.
        """
        self._cpp_backend.var_w = value

    @property
    def mu_b(self) -> np.ndarray:
        """
        Gets the mean of the biases (mu_b) as a NumPy array.
        """
        return self._cpp_backend.mu_b

    @mu_b.setter
    def mu_b(self, value: np.ndarray):
        """
        Sets the mean of the biases (mu_b) as a NumPy array.
        """
        self._cpp_backend.mu_b = value

    @property
    def var_b(self) -> np.ndarray:
        """
        Gets the variance of the biases (var_b) as a NumPy array.
        """
        return self._cpp_backend.var_b

    @var_b.setter
    def var_b(self, value: np.ndarray):
        """
        Sets the variance of the biases (var_b) as a NumPy array.
        """
        self._cpp_backend.var_b = value

    @property
    def delta_mu_w(self) -> np.ndarray:
        """
        Gets the delta mean of the weights (delta_mu_w) as a NumPy array.
        """
        return self._cpp_backend.delta_mu_w

    @delta_mu_w.setter
    def delta_mu_w(self, value: np.ndarray):
        """
        Sets the delta mean of the weights (delta_mu_w) as a NumPy array.
        """
        self._cpp_backend.delta_mu_w = value

    @property
    def delta_var_w(self) -> np.ndarray:
        """
        Gets the delta variance of the weights (delta_var_w) as a NumPy array.
        The delta corresponds to the amount of change induced by the update step.
        """
        return self._cpp_backend.delta_var_w

    @delta_var_w.setter
    def delta_var_w(self, value: np.ndarray):
        """
        Sets the delta variance of the weights (delta_var_w) as a NumPy array.
        The delta corresponds to the amount of change induced by the update step.
        """
        self._cpp_backend.delta_var_w = value

    @property
    def delta_mu_b(self) -> np.ndarray:
        """
        Gets the delta mean of the biases (delta_mu_b) as a NumPy array.
        This delta corresponds to the amount of change induced by the update step.
        """
        return self._cpp_backend.delta_mu_b

    @delta_mu_b.setter
    def delta_mu_b(self, value: np.ndarray):
        """
        Sets the delta mean of the biases (delta_mu_b) as a NumPy array.
        This delta corresponds to the amount of change induced by the update step.
        """
        self._cpp_backend.delta_mu_b = value

    @property
    def delta_var_b(self) -> np.ndarray:
        """
        Gets the delta variance of the biases (delta_var_b) as a NumPy array.
        This delta corresponds to the amount of change induced by the update step.
        """
        return self._cpp_backend.delta_var_b

    @delta_var_b.setter
    def delta_var_b(self, value: np.ndarray):
        """
        Sets the delta variance of the biases (delta_var_b) as a NumPy array.
        This delta corresponds to the amount of change induced by the update step.
        """
        self._cpp_backend.delta_var_b = value

    @property
    def num_threads(self) -> int:
        """
        Gets the number of threads to use for computations.
        """
        return self._cpp_backend.num_threads

    @num_threads.setter
    def num_threads(self, value: int):
        """
        Sets the number of threads to use for computations.
        """
        self._cpp_backend.num_threads = value

    @property
    def training(self) -> bool:
        """
        Gets a boolean indicating whether the layer is in training mode.
        """
        return self._cpp_backend.training

    @training.setter
    def training(self, value: bool):
        """
        Sets a boolean indicating whether the layer is in training mode.
        """
        self._cpp_backend.training = value

    @property
    def device(self) -> bool:
        """
        Gets a boolean indicating whether the layer is on the GPU ('cuda') or CPU ('cpu').
        """
        return self._cpp_backend.device

    @training.setter
    def device(self, value: str):
        """
        Sets the device on which the layer operates ('cpu' or 'cuda').
        """
        self._cpp_backend.device = value
