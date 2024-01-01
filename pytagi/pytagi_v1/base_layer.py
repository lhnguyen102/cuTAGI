# Temporary import. It will be removed in the final vserion
import sys
import os

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

import cutagitest
import numpy as np


class BaseLayer:
    """Base layer"""

    def __init__(self):
        self._cpp_backend = cutagitest.BaseLayer()

    def to_cuda(self):
        self._cpp_backend.to_cuda()

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
