pytagi.nn.activation
====================

.. py:module:: pytagi.nn.activation


Classes
-------

.. autoapisummary::

   pytagi.nn.activation.ReLU
   pytagi.nn.activation.Sigmoid
   pytagi.nn.activation.Tanh
   pytagi.nn.activation.MixtureReLU
   pytagi.nn.activation.MixtureSigmoid
   pytagi.nn.activation.MixtureTanh
   pytagi.nn.activation.Softplus
   pytagi.nn.activation.LeakyReLU
   pytagi.nn.activation.Softmax
   pytagi.nn.activation.EvenExp
   pytagi.nn.activation.Remax
   pytagi.nn.activation.ClosedFormSoftmax


Module Contents
---------------

.. py:class:: ReLU

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the Rectified Linear Unit function element-wise.

   In the context of pyTAGI, which handles distributions, this layer processes an
   input Gaussian distribution and outputs a rectified Gaussian distribution.

   .. math::
       \text{ReLU}(x) = (x)^+ = \max(0, x)

   .. image:: ../../../../_static/relu_simplified_gaussian_io.png
      :align: center


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: Sigmoid

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the Sigmoid function element-wise.

   When processing a Gaussian distribution, this layer approximates the
   output distribution after applying the sigmoid function. The output
   values are constrained to the range (0, 1).

   .. math::
       \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + e^{-x}}

   .. image:: _images/Sigmoid.png
      :align: center


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: Tanh

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the Hyperbolic Tangent function element-wise.

   When processing a Gaussian distribution, this layer approximates the
   output distribution after applying the Tanh function. The output
   values are constrained to the range (-1, 1).

   .. math::
       \text{Tanh}(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

   .. image:: _images/Tanh.png
      :align: center


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: MixtureReLU

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the probabilistic Rectified Linear Unit.

   This activation function is designed for probabilistic neural networks where
   activations are represented by distributions. It takes a Gaussian distribution
   as input and computes the exact moments (mean and variance) of the output,
   which is a rectified Gaussian distribution.

   For an input random variable :math:`X \sim \mathcal{N}(\mu, \sigma^2)`, the output
   :math:`Y = \max(0, X)` results in a rectified Gaussian.

   .. image:: _images/MixtureReLU.png
      :align: center


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: MixtureSigmoid

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the probabilistic Sigmoid function.

   This activation function processes an input Gaussian distribution and
   approximates the output distribution after applying the sigmoid function.
   The resulting distribution is confined to the range (0, 1).

   For an input random variable :math:`X \sim \mathcal{N}(\mu, \sigma^2)`, this layer
   approximates the distribution of :math:`Y = \frac{1}{1 + e^{-X}}`.


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: MixtureTanh

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the probabilistic Hyperbolic Tangent function.

   This activation function processes an input Gaussian distribution and
   approximates the output distribution after applying the Tanh function.
   The resulting distribution is confined to the range (-1, 1).

   For an input random variable :math:`X \sim \mathcal{N}(\mu, \sigma^2)`, this layer
   approximates the distribution of :math:`Y = \tanh(X)`.


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: Softplus

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the Softplus function element-wise.

   Softplus is a smooth approximation of the ReLU function.

   .. math::
       \text{Softplus}(x) = \log(1 + e^{x})

   .. image:: _images/Softplus.png
      :align: center


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: LeakyReLU

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the Leaky Rectified Linear Unit function element-wise.

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


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: Softmax

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the Softmax function to an n-dimensional input Tensor.

   The Softmax function rescales the input so that the elements of the output
   lie in the range [0,1] and sum to 1. It is commonly used as the final
   activation function in a classification network to produce probability
   distributions over classes.

   .. math::
       \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: EvenExp

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the EvenExp activation function.

   This is an even function (symmetric about the y-axis) based on the
   exponential function, related to the hyperbolic cosine. It can be useful
   in specific network architectures where such symmetry is desired.

   .. math::
       \text{EvenExp}(x) = \exp(x) + \exp(-x) = 2 \cosh(x)

   .. image:: _images/EvenExp.png
      :align: center


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: Remax

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies the Remax activation function.

   Remax is a softmax-like activation function, often used in attention
   mechanisms. It is designed to be a more expressive alternative to softmax,
   particularly for tasks involving ranking or selection, and is based on
   the recursive application of the max operator.


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: ClosedFormSoftmax

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies a closed-form variant of the Softmax function.

   This layer is a specific implementation of Softmax designed for efficient,
   closed-form updates within the pyTAGI probabilistic framework. It computes
   the output distribution that results from applying the Softmax transformation
   to an input Gaussian distribution.


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str
