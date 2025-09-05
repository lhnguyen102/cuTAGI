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


class CELU(BaseLayer):
    """CELU"""

    def __init__(self):
        self._cpp_backend = cutagi.CELU()

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


class Remax(BaseLayer):
    """Remax"""

    def __init__(self):
        self._cpp_backend = cutagi.Remax()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class ClosedFormSoftmax(BaseLayer):
    """ClosedFormSoftmax"""

    def __init__(self):
        self._cpp_backend = cutagi.ClosedFormSoftmax()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class SplitActivation(BaseLayer):
    """
    Applies a specified activation to odd-indexed inputs and another
    (or an identity function) to even-indexed inputs.

    Args:
        odd_layer (BaseLayer): The activation layer to apply to odd-indexed elements.
        even_layer (BaseLayer, optional): The activation layer to apply to
                                          even-indexed elements. Defaults to None,
                                          which is an identity transformation.
    """

    def __init__(self, odd_layer: BaseLayer, even_layer: BaseLayer = None):
        if even_layer is None:
            # Call C++ constructor with one argument
            self._cpp_backend = cutagi.SplitActivation(odd_layer._cpp_backend)
        else:
            # Call C++ constructor with two arguments
            self._cpp_backend = cutagi.SplitActivation(
                odd_layer._cpp_backend, even_layer._cpp_backend
            )

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Exp(BaseLayer):
    """Exp"""

    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        self._cpp_backend = cutagi.Exp(scale, shift)

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class AGVI(BaseLayer):
    """AGVI (Approximate Gaussian Variance Inference)"""

    def __init__(
        self,
        activation_layer: BaseLayer,
        overfit_mu: bool = True,
        agvi: bool = True,
    ):
        """
        Initializes the AGVI layer.

        Args:
            activation_layer: The inner activation layer to be used.
            overfit_mu: If true, uses a different Kalman gain for the mean delta
                        to encourage the mean overfitting. Defaults to True.
        """
        self._cpp_backend = cutagi.AGVI(
            activation_layer._cpp_backend, overfit_mu, agvi
        )

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()
