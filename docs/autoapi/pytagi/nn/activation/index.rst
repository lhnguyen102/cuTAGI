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


   Applies the Rectified Linear Unit function.

   This layer processes an input Gaussian distribution and outputs the moments
   for a rectified linear unit. This layer relies on a first-order
   Taylor-series approximation where the activation function is locally
   linearized at the input expected value.

   .. math::
       \text{ReLU}(x) = (x)^+ = \max(0, x)

   .. image:: ../../../../_static/relu_simplified_gaussian_io.png
      :align: center

   Initializes the BaseLayer with a C++ backend instance.


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

   This layer approximates the moments after applying the sigmoid function whose
   values are constrained to the range (0, 1). This layer relies on a first-order
   Taylor-series approximation where the activation function is locally
   linearized at the input expected value.

   .. math::
       \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + e^{-x}}

   .. image:: ../../../../_static/activation_io_sigmoid.png
      :align: center

   Initializes the BaseLayer with a C++ backend instance.


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


   Applies the Hyperbolic Tangent function.

   This layer approximates the moments after applying the Tanh function whose
   values are constrained to the range (-1, 1). This layer relies on a first-order
   Taylor-series approximation where the activation function is locally
   linearized at the input expected value.

   .. math::
       \text{Tanh}(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

   .. image:: ../../../../_static/activation_io_tanh.png
      :align: center

   Initializes the BaseLayer with a C++ backend instance.


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


   Applies a probabilistic Rectified Linear Unit approximation.

   This layer processes an input Gaussian distribution and outputs the moments
   for a rectified linear unit. This layer relies on exact moment calculations.

   For an input random variable :math:`X \sim \mathcal{N}(\mu, \sigma^2)`, the output
   :math:`Y = \max(0, X)` results in a rectified Gaussian.

   .. image:: ../../../../_static/activation_io_mixture_relu.png
      :align: center

   Initializes the BaseLayer with a C++ backend instance.


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


   Applies a probabilistic picewise-linear Sigmoid-like function.

   This layer processes an input Gaussian distribution and outputs the moments
   for a picewise-linear Sigmoid-like function. This layer relies on exact
   moment calculations.

   .. image:: ../../../../_static/activation_io_mixture_sigmoid.png
      :align: center

   Initializes the BaseLayer with a C++ backend instance.


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


   Applies a probabilistic piecewise-linear Hyperbolic Tangent function.

   This layer processes an input Gaussian distribution and outputs the moments
   for a picewise-linear Tanh-like function. This layer relies on exact
   moment calculations.

   .. image:: ../../../../_static/activation_io_mixture_tanh.png
      :align: center

   Initializes the BaseLayer with a C++ backend instance.


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

   Softplus is a smooth approximation of the ReLU function. This layer relies
   on a first-order Taylor-series approximation where the activation function
   is locally linearized at the input expected value.

   .. math::
       \text{Softplus}(x) = \log(1 + e^{x})

   .. image:: ../../../../_static/activation_io_softplus.png
      :align: center

   Initializes the BaseLayer with a C++ backend instance.


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

   Initializes the BaseLayer with a C++ backend instance.


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


   Applies a Local-Linearization of the Softmax function to an n-dimensional input.

   The Softmax function rescales the input so that the elements of the output
   lie in the range [0,1] and sum to 1. It is commonly used as the final
   activation function in a classification network to produce probability
   distributions over classes.

   .. math::
       \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

   Initializes the BaseLayer with a C++ backend instance.


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

   This function allows passing only the odd postions of the output layer through
   an exponential activation function. This is used for going from V2_bar to V2_bar_tilde
   for the aleatoric uncertainty inference in the case of heteroscedastic regression.

   .. math::
       \text{EvenExp}(x) = \begin{cases}
           \exp(x) & \text{if } x \text{ is at an odd position}\\
           x & \text{if } x \text{ is at an even position}
       \end{cases}


   Initializes the BaseLayer with a C++ backend instance.


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


   Applies a probabilistic Remax approximation function.

   Remax is a softmax-like activation function which replaces the exponential function by a
   mixtureRelu. It rescales the input so that the elements of the output
   lie in the range [0,1] and sum to 1. It is commonly used as the final
   activation function in a classification network to produce probability
   distributions over classes.

   .. math::
       \text{Remax}(x_{i}) = \frac{\text{ReLU}(x_i)}{\sum_j \text{ReLU}(x_j)}

   Initializes the BaseLayer with a C++ backend instance.


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


   Applies a probabilistic Softmax approximation function.

   Closed-form softmax is an approximation of the deterministic softmax function that provides
   a closed-form solution for the output moments of Gaussian inputs. It is commonly
   used as the final activation function in a classification network to produce
   probability distributions over classes.

   .. math::
       \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

   Initializes the BaseLayer with a C++ backend instance.


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str
