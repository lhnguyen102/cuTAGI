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

    .. image:: _images/Sigmoid.png
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

    .. image:: _images/Tanh.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.Tanh()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureReLU(BaseLayer):
    r"""Applies the probabilistic Rectified Linear Unit.

    This activation function is designed for probabilistic neural networks where
    activations are represented by distributions. It takes a Gaussian distribution
    as input and computes the exact moments (mean and variance) of the output,
    which is a rectified Gaussian distribution.

    For an input random variable :math:`X \sim \mathcal{N}(\mu, \sigma^2)`, the output
    :math:`Y = \max(0, X)` results in a rectified Gaussian.

    .. image:: _images/MixtureReLU.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.MixtureReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureSigmoid(BaseLayer):
    r"""Applies the probabilistic Sigmoid function.

    This activation function processes an input Gaussian distribution and
    approximates the output distribution after applying the sigmoid function.
    The resulting distribution is confined to the range (0, 1).

    For an input random variable :math:`X \sim \mathcal{N}(\mu, \sigma^2)`, this layer
    approximates the distribution of :math:`Y = \frac{1}{1 + e^{-X}}`.
    """

    def __init__(self):
        self._cpp_backend = cutagi.MixtureSigmoid()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class MixtureTanh(BaseLayer):
    r"""Applies the probabilistic Hyperbolic Tangent function.

    This activation function processes an input Gaussian distribution and
    approximates the output distribution after applying the Tanh function.
    The resulting distribution is confined to the range (-1, 1).

    For an input random variable :math:`X \sim \mathcal{N}(\mu, \sigma^2)`, this layer
    approximates the distribution of :math:`Y = \tanh(X)`.
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

    .. image:: _images/Softplus.png
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

    .. image:: _images/LeakyReLU.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.LeakyReLU()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Softmax(BaseLayer):
    r"""Applies the Softmax function to an n-dimensional input Tensor.

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

    This is an even function (symmetric about the y-axis) based on the
    exponential function, related to the hyperbolic cosine. It can be useful
    in specific network architectures where such symmetry is desired.

    .. math::
        \text{EvenExp}(x) = \exp(x) + \exp(-x) = 2 \cosh(x)

    .. image:: _images/EvenExp.png
       :align: center
    """

    def __init__(self):
        self._cpp_backend = cutagi.EvenExp()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class Remax(BaseLayer):
    """Applies the Remax activation function.

    Remax is a softmax-like activation function, often used in attention
    mechanisms. It is designed to be a more expressive alternative to softmax,
    particularly for tasks involving ranking or selection, and is based on
    the recursive application of the max operator.
    """

    def __init__(self):
        self._cpp_backend = cutagi.Remax()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()


class ClosedFormSoftmax(BaseLayer):
    """Applies a closed-form variant of the Softmax function.

    This layer is a specific implementation of Softmax designed for efficient,
    closed-form updates within the pyTAGI probabilistic framework. It computes
    the output distribution that results from applying the Softmax transformation
    to an input Gaussian distribution.
    """

    def __init__(self):
        self._cpp_backend = cutagi.ClosedFormSoftmax()

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()
