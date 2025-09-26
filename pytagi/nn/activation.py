import cutagi

from pytagi.nn.base_layer import BaseLayer


class ReLU(BaseLayer):
    r"""Applies the Rectified Linear Unit function element-wise.

    In the context of pyTAGI, which handles distributions, this layer processes an
    input Gaussian distribution and outputs a rectified Gaussian distribution.

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

    When processing a Gaussian distribution, this layer approximates the
    output distribution after applying the sigmoid function. The output
    values are constrained to the range (0, 1).

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
    r"""Applies the Hyperbolic Tangent function element-wise.

    When processing a Gaussian distribution, this layer approximates the
    output distribution after applying the Tanh function. The output
    values are constrained to the range (-1, 1).

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

    This activation function is designed for probabilistic neural networks where
    activations are represented by distributions. It takes a Gaussian distribution
    as input and computes the exact moments (mean and variance) of the output,
    which is a truncated Gaussian distribution.

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
    r"""Applies a probabilistic Sigmoid function approximation.

    This activation function processes an input Gaussian distribution and
    approximates the output distribution after applying the sigmoid function.
    The resulting distribution is confined to the range (0, 1).

    For an input random variable :math:`X \sim \mathcal{N}(\mu, \sigma^2)`, this layer
    approximates the distribution of :math:`Y = \frac{1}{1 + e^{-X}}`.

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
    r"""Applies a probabilistic Hyperbolic Tangent function approximation.

    This activation function processes an input Gaussian distribution and
    approximates the output distribution after applying the Tanh function.
    The resulting distribution is confined to the range (-1, 1).

    For an input random variable :math:`X \sim \mathcal{N}(\mu, \sigma^2)`, this layer
    approximates the distribution of :math:`Y = \tanh(X)`.

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

    Softplus is a smooth approximation of the ReLU function.

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
    when the unit is not active, which can help mitigate the "dying ReLU" problem.

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

    This is an even function allows to pass just the odd postions of the output layer through
    an exponential activation function. So it allows passing from V2_bar to V2_bar_tilde for
    the correct aleatoric uncertainty inference in the case of heteroscedastic regression.

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

    Remax is a softmax-like activation function wich replaces the exponential function by a
    rectified linear unit. It rescales the input so that the elements of the output
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

    Closed-form softmax is an approximation of the softmax function that provides
    a closed-form solution for the output distribution when the input is a Gaussian
    distribution. It is commonly used as the final activation function in a classification
    network to produce probability distributions over classes.

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    """

    def __init__(self):
        self._cpp_backend = cutagi.ClosedFormSoftmax()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()
