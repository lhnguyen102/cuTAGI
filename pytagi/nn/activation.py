import cutagi

from pytagi.nn.base_layer import BaseLayer


class ReLU(BaseLayer):
    """ReLU"""

    def __init__(self):
        self._cpp_backend = cutagi.ReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Sigmoid(BaseLayer):
    """Sigmoid"""

    def __init__(self):
        self._cpp_backend = cutagi.Sigmoid()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Tanh(BaseLayer):
    """Tanh"""

    def __init__(self):
        self._cpp_backend = cutagi.Tanh()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureReLU(BaseLayer):
    """Mixture ReLU"""

    def __init__(self):
        self._cpp_backend = cutagi.MixtureReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureSigmoid(BaseLayer):
    """Mixture Sigmoid"""

    def __init__(self):
        self._cpp_backend = cutagi.MixtureSigmoid()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureTanh(BaseLayer):
    """Mixture Tanh"""

    def __init__(self):
        self._cpp_backend = cutagi.MixtureTanh()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Softplus(BaseLayer):
    """Softplus"""

    def __init__(self):
        self._cpp_backend = cutagi.Softplus()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class LeakyReLU(BaseLayer):
    """Leaky ReLU"""

    def __init__(self):
        self._cpp_backend = cutagi.LeakyReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Softmax(BaseLayer):
    """Softmax"""

    def __init__(self):
        self._cpp_backend = cutagi.Softmax()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()
