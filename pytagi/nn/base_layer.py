import cutagi
import numpy as np


class BaseLayer:
    """Base layer"""

    def __init__(self):
        self._cpp_backend = cutagi.BaseLayer()

    def to_cuda(self):
        self._cpp_backend.to_cuda()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()

    def get_max_num_states(self) -> int:
        return self._cpp_backend.get_max_num_states()

    @property
    def input_size(self) -> int:
        return self._cpp_backend.input_size

    @input_size.setter
    def input_size(self, value: int):
        self._cpp_backend.input_size = value

    @property
    def output_size(self) -> int:
        return self._cpp_backend.output_size

    @output_size.setter
    def output_size(self, value: int):
        self._cpp_backend.output_size = value

    @property
    def in_width(self) -> int:
        return self._cpp_backend.in_width

    @in_width.setter
    def in_width(self, value: int):
        self._cpp_backend.in_width = value

    @property
    def in_height(self) -> int:
        return self._cpp_backend.in_height

    @in_height.setter
    def in_height(self, value: int):
        self._cpp_backend.in_height = value

    @property
    def in_channels(self) -> int:
        return self._cpp_backend.in_channels

    @in_channels.setter
    def in_channels(self, value: int):
        self._cpp_backend.in_channels = value

    @property
    def out_width(self) -> int:
        return self._cpp_backend.out_width

    @out_width.setter
    def out_width(self, value: int):
        self._cpp_backend.out_width = value

    @property
    def out_height(self) -> int:
        return self._cpp_backend.out_height

    @out_height.setter
    def out_height(self, value: int):
        self._cpp_backend.out_height = value

    @property
    def out_channels(self) -> int:
        return self._cpp_backend.out_channels

    @out_channels.setter
    def out_channels(self, value: int):
        self._cpp_backend.out_channels = value

    @property
    def bias(self) -> bool:
        return self._cpp_backend.bias

    @bias.setter
    def bias(self, value: bool):
        self._cpp_backend.bias = value

    @property
    def num_weights(self) -> int:
        return self._cpp_backend.num_weights

    @num_weights.setter
    def num_weights(self, value: int):
        self._cpp_backend.num_weights = value

    @property
    def num_biases(self) -> int:
        return self._cpp_backend.num_biases

    @num_biases.setter
    def num_biases(self, value: int):
        self._cpp_backend.num_biases = value

    @property
    def mu_w(self) -> np.ndarray:
        return self._cpp_backend.mu_w

    @mu_w.setter
    def mu_w(self, value: np.ndarray):
        self._cpp_backend.mu_w = value

    @property
    def var_w(self) -> np.ndarray:
        return self._cpp_backend.var_w

    @var_w.setter
    def var_w(self, value: np.ndarray):
        self._cpp_backend.var_w = value

    @property
    def mu_b(self) -> np.ndarray:
        return self._cpp_backend.mu_b

    @mu_b.setter
    def mu_b(self, value: np.ndarray):
        self._cpp_backend.mu_b = value

    @property
    def var_b(self) -> np.ndarray:
        return self._cpp_backend.var_b

    @var_b.setter
    def var_b(self, value: np.ndarray):
        self._cpp_backend.var_b = value

    @property
    def delta_mu_w(self) -> np.ndarray:
        return self._cpp_backend.delta_mu_w

    @delta_mu_w.setter
    def delta_mu_w(self, value: np.ndarray):
        self._cpp_backend.delta_mu_w = value

    @property
    def delta_var_w(self) -> np.ndarray:
        return self._cpp_backend.delta_var_w

    @delta_var_w.setter
    def delta_var_w(self, value: np.ndarray):
        self._cpp_backend.delta_var_w = value

    @property
    def delta_mu_b(self) -> np.ndarray:
        return self._cpp_backend.delta_mu_b

    @delta_mu_b.setter
    def delta_mu_b(self, value: np.ndarray):
        self._cpp_backend.delta_mu_b = value

    @property
    def delta_var_b(self) -> np.ndarray:
        return self._cpp_backend.delta_var_b

    @delta_var_b.setter
    def delta_var_b(self, value: np.ndarray):
        self._cpp_backend.delta_var_b = value

    @property
    def num_threads(self) -> int:
        return self._cpp_backend.num_threads

    @num_threads.setter
    def num_threads(self, value: int):
        self._cpp_backend.num_threads = value

    @property
    def training(self) -> bool:
        return self._cpp_backend.training

    @training.setter
    def training(self, value: bool):
        self._cpp_backend.training = value

    @property
    def device(self) -> bool:
        return self._cpp_backend.device

    @training.setter
    def device(self, value: str):
        self._cpp_backend.device = value
