# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

import cutagitest
from pytagi.nn.base_layer import BaseLayer


class ReLU(BaseLayer):
    """ReLU"""

    def __init__(self):
        self._cpp_backend = cutagitest.ReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Sigmoid(BaseLayer):
    """Sigmoid"""

    def __init__(self):
        self._cpp_backend = cutagitest.Sigmoid()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Tanh(BaseLayer):
    """Tanh"""

    def __init__(self):
        self._cpp_backend = cutagitest.Tanh()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureReLU(BaseLayer):
    """Mixture ReLU"""

    def __init__(self):
        self._cpp_backend = cutagitest.MixtureReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureSigmoid(BaseLayer):
    """Mixture Sigmoid"""

    def __init__(self):
        self._cpp_backend = cutagitest.MixtureSigmoid()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureTanh(BaseLayer):
    """Mixture Tanh"""

    def __init__(self):
        self._cpp_backend = cutagitest.MixtureTanh()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Softplus(BaseLayer):
    """Softplus"""

    def __init__(self):
        self._cpp_backend = cutagitest.Softplus()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class LeakyReLU(BaseLayer):
    """Leaky ReLU"""

    def __init__(self):
        self._cpp_backend = cutagitest.LeakyReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Softmax(BaseLayer):
    """Softmax"""

    def __init__(self):
        self._cpp_backend = cutagitest.Softmax()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()
