import cutagi

from pytagi.nn.base_layer import BaseLayer


class ReLU(BaseLayer):
    r"""Applies the Rectified Linear Unit function.

    This layer processes an input Gaussian distribution and outputs the moments
    for a rectified linear unit. This layer relies on a first-order
    Taylor-series approximation where the activation function is locally
    linearized at the input expected value.

    .. math::
        \text{ReLU}(x) = (x)^+ = \max(0, x)

    .. image:: ../../../../_static/relu_simplified_gaussian_io.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.ReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Sigmoid(BaseLayer):
    r"""Applies the Sigmoid function element-wise.

    This layer approximates the moments after applying the sigmoid function whose
    values are constrained to the range (0, 1). This layer relies on a first-order
    Taylor-series approximation where the activation function is locally
    linearized at the input expected value.

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + e^{-x}}

    .. image:: ../../../../_static/activation_io_sigmoid.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.Sigmoid()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Tanh(BaseLayer):
    r"""Applies the Hyperbolic Tangent function.

    This layer approximates the moments after applying the Tanh function whose
    values are constrained to the range (-1, 1). This layer relies on a first-order
    Taylor-series approximation where the activation function is locally
    linearized at the input expected value.

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

    .. image:: ../../../../_static/activation_io_tanh.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.Tanh()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureReLU(BaseLayer):
    r"""Applies a probabilistic Rectified Linear Unit approximation.

    This layer processes an input Gaussian distribution and outputs the moments
    for a rectified linear unit. This layer relies on exact moment calculations.

    For an input random variable :math:`X \sim \mathcal{N}(\mu, \sigma^2)`, the output
    :math:`Y = \max(0, X)` results in a rectified Gaussian.

    .. image:: ../../../../_static/activation_io_mixture_relu.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.MixtureReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureSigmoid(BaseLayer):
    r"""Applies a probabilistic picewise-linear Sigmoid-like function.

    This layer processes an input Gaussian distribution and outputs the moments
    for a picewise-linear Sigmoid-like function. This layer relies on exact
    moment calculations.

    .. image:: ../../../../_static/activation_io_mixture_sigmoid.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.MixtureSigmoid()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureTanh(BaseLayer):
    r"""Applies a probabilistic piecewise-linear Hyperbolic Tangent function.

    This layer processes an input Gaussian distribution and outputs the moments
    for a picewise-linear Tanh-like function. This layer relies on exact
    moment calculations.

    .. image:: ../../../../_static/activation_io_mixture_tanh.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.MixtureTanh()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Softplus(BaseLayer):
    r"""Applies the Softplus function element-wise.

    Softplus is a smooth approximation of the ReLU function. This layer relies
    on a first-order Taylor-series approximation where the activation function
    is locally linearized at the input expected value.

    .. math::
        \text{Softplus}(x) = \log(1 + e^{x})

    .. image:: ../../../../_static/activation_io_softplus.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.Softplus()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class LeakyReLU(BaseLayer):
    r"""Applies the Leaky Rectified Linear Unit function element-wise.

    This is a variant of ReLU that allows a small, non-zero gradient
    when the unit is not active. This layer relies on a first-order
    Taylor-series approximation where the activation function is locally
    linearized at the input expected value.

    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            \alpha x & \text{ otherwise }
        \end{cases}

    Where :math:`\alpha` is the `negative_slope` and is set to 0.1.

    .. image:: ../../../../_static/activation_io_leaky_relu.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.LeakyReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Softmax(BaseLayer):
    r"""Applies a Local-Linearization of the Softmax function to an n-dimensional input.

    The Softmax function rescales the input so that the elements of the output
    lie in the range [0,1] and sum to 1. It is commonly used as the final
    activation function in a classification network to produce probability
    distributions over classes.

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    """

    def __init__(self):
        self._cpp_backend = cutagi.Softmax()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class EvenExp(BaseLayer):
    r"""Applies the EvenExp activation function.

    This function allows passing only the odd postions of the output layer through
    an exponential activation function. This is used for going from V2_bar to V2_bar_tilde
    for the aleatoric uncertainty inference in the case of heteroscedastic regression.

    .. math::
        \text{EvenExp}(x) = \begin{cases}
            \exp(x) & \text{if } x \text{ is at an odd position}\\
            x & \text{if } x \text{ is at an even position}
        \end{cases}

    """

    def __init__(self):
        self._cpp_backend = cutagi.EvenExp()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Remax(BaseLayer):
    r"""Applies a probabilistic Remax approximation function.

    Remax is a softmax-like activation function which replaces the exponential function by a
    mixtureRelu. It rescales the input so that the elements of the output
    lie in the range [0,1] and sum to 1. It is commonly used as the final
    activation function in a classification network to produce probability
    distributions over classes.

    .. math::
        \text{Remax}(x_{i}) = \frac{\text{ReLU}(x_i)}{\sum_j \text{ReLU}(x_j)}
    """

    def __init__(self):
        self._cpp_backend = cutagi.Remax()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class ClosedFormSoftmax(BaseLayer):
    r"""Applies a probabilistic Softmax approximation function.

    Closed-form softmax is an approximation of the deterministic softmax function that provides
    a closed-form solution for the output moments of Gaussian inputs. It is commonly
    used as the final activation function in a classification network to produce
    probability distributions over classes.

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    """

    def __init__(self):
        self._cpp_backend = cutagi.ClosedFormSoftmax()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()
