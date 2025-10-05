pytagi.nn
=========

.. py:module:: pytagi.nn

.. autoapi-nested-parse::

   Neural Network module for pyTAGI.

   This module provides various neural network layers and components,
   including activation functions, base layers, convolutional layers,
   recurrent layers, and utility modules. These components are designed
   to work with probabilistic data structures and leverage a C++ backend
   for performance.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/pytagi/nn/activation/index
   /autoapi/pytagi/nn/base_layer/index
   /autoapi/pytagi/nn/batch_norm/index
   /autoapi/pytagi/nn/conv2d/index
   /autoapi/pytagi/nn/convtranspose2d/index
   /autoapi/pytagi/nn/data_struct/index
   /autoapi/pytagi/nn/ddp/index
   /autoapi/pytagi/nn/layer_block/index
   /autoapi/pytagi/nn/layer_norm/index
   /autoapi/pytagi/nn/linear/index
   /autoapi/pytagi/nn/lstm/index
   /autoapi/pytagi/nn/output_updater/index
   /autoapi/pytagi/nn/pooling/index
   /autoapi/pytagi/nn/resnet_block/index
   /autoapi/pytagi/nn/sequential/index
   /autoapi/pytagi/nn/slinear/index
   /autoapi/pytagi/nn/slstm/index


Classes
-------

.. autoapisummary::

   pytagi.nn.ClosedFormSoftmax
   pytagi.nn.EvenExp
   pytagi.nn.LeakyReLU
   pytagi.nn.MixtureReLU
   pytagi.nn.MixtureSigmoid
   pytagi.nn.MixtureTanh
   pytagi.nn.ReLU
   pytagi.nn.Remax
   pytagi.nn.Sigmoid
   pytagi.nn.Softmax
   pytagi.nn.Softplus
   pytagi.nn.Tanh
   pytagi.nn.BaseLayer
   pytagi.nn.BatchNorm2d
   pytagi.nn.Conv2d
   pytagi.nn.ConvTranspose2d
   pytagi.nn.BaseDeltaStates
   pytagi.nn.BaseHiddenStates
   pytagi.nn.HRCSoftmax
   pytagi.nn.DDPConfig
   pytagi.nn.DDPSequential
   pytagi.nn.LayerBlock
   pytagi.nn.LayerNorm
   pytagi.nn.Linear
   pytagi.nn.LSTM
   pytagi.nn.OutputUpdater
   pytagi.nn.AvgPool2d
   pytagi.nn.MaxPool2d
   pytagi.nn.ResNetBlock
   pytagi.nn.Sequential
   pytagi.nn.SLinear
   pytagi.nn.SLSTM


Package Contents
----------------

.. py:class:: ClosedFormSoftmax

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies a probabilistic Softmax approximation function.

   Closed-form softmax is an approximation of the deterministic softmax function that provides
   a closed-form solution for the output moments of Gaussian inputs. It is commonly
   used as the final activation function in a classification network to produce
   probability distributions over classes.

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

   This function allows passing only the odd postions of the output layer through
   an exponential activation function. This is used for going from V2_bar to V2_bar_tilde
   for the aleatoric uncertainty inference in the case of heteroscedastic regression.

   .. math::
       \text{EvenExp}(x) = \begin{cases}
           \exp(x) & \text{if } x \text{ is at an odd position}\\
           x & \text{if } x \text{ is at an even position}
       \end{cases}



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


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



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


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



.. py:class:: BaseLayer

   Base layer class providing common functionality and properties for neural network layers.
   This class acts as a Python wrapper for the C++ backend, exposing layer attributes
   and methods for managing layer information, device placement, and parameters.


   .. py:method:: to_cuda()

      Moves the layer's parameters and computations to the CUDA device.



   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer.

      :returns: The name of the layer.
      :rtype: str



   .. py:method:: get_max_num_states() -> int

      Retrieves the maximum number of states the layer can hold.

      :returns: The maximum number of states.
      :rtype: int



   .. py:property:: input_size
      :type: int


      Gets the input size of the layer.


   .. py:property:: output_size
      :type: int


      Gets the output size of the layer.


   .. py:property:: in_width
      :type: int


      Gets the input width of the layer (for convolutional layers).


   .. py:property:: in_height
      :type: int


      Gets the input height of the layer (for convolutional layers).


   .. py:property:: in_channels
      :type: int


      Gets the input channels of the layer (for convolutional layers).


   .. py:property:: out_width
      :type: int


      Gets the output width of the layer (for convolutional layers).


   .. py:property:: out_height
      :type: int


      Gets the output height of the layer (for convolutional layers).


   .. py:property:: out_channels
      :type: int


      Gets the output channels of the layer (for convolutional layers).


   .. py:property:: bias
      :type: bool


      Gets a boolean indicating whether the layer has a bias term.


   .. py:property:: num_weights
      :type: int


      Gets the total number of weights in the layer.


   .. py:property:: num_biases
      :type: int


      Gets the total number of biases in the layer.


   .. py:property:: mu_w
      :type: numpy.ndarray


      Gets the mean of the weights (mu_w) as a NumPy array.


   .. py:property:: var_w
      :type: numpy.ndarray


      Gets the variance of the weights (var_w) as a NumPy array.


   .. py:property:: mu_b
      :type: numpy.ndarray


      Gets the mean of the biases (mu_b) as a NumPy array.


   .. py:property:: var_b
      :type: numpy.ndarray


      Gets the variance of the biases (var_b) as a NumPy array.


   .. py:property:: delta_mu_w
      :type: numpy.ndarray


      Gets the delta mean of the weights (delta_mu_w) as a NumPy array.


   .. py:property:: delta_var_w
      :type: numpy.ndarray


      Gets the delta variance of the weights (delta_var_w) as a NumPy array.
      The delta corresponds to the amount of change induced by the update step.


   .. py:property:: delta_mu_b
      :type: numpy.ndarray


      Gets the delta mean of the biases (delta_mu_b) as a NumPy array.
      This delta corresponds to the amount of change induced by the update step.


   .. py:property:: delta_var_b
      :type: numpy.ndarray


      Gets the delta variance of the biases (delta_var_b) as a NumPy array.
      This delta corresponds to the amount of change induced by the update step.


   .. py:property:: num_threads
      :type: int


      Gets the number of threads to use for computations.


   .. py:property:: training
      :type: bool


      Gets a boolean indicating whether the layer is in training mode.


   .. py:property:: device
      :type: bool


      Gets a boolean indicating whether the layer is on the GPU ('cuda') or CPU ('cpu').


.. py:class:: BatchNorm2d(num_features: int, eps: float = 1e-05, momentum: float = 0.9, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies 2D Batch Normalization.

   Batch Normalization normalizes the inputs of a layer by re-centering and
   re-scaling them.

   :param num_features: The number of features in the input tensor.
   :type num_features: int
   :param eps: A small value added to the variance to avoid division by zero.
               Defaults to 1e-5.
   :type eps: float
   :param momentum: The momentum for the running mean and variance.
                    Defaults to 0.9.
   :type momentum: float
   :param bias: Whether to include a learnable bias term. Defaults to True.
   :type bias: bool
   :param gain_weight: Initial value for the gain (scale) parameter. Defaults to 1.0.
   :type gain_weight: float
   :param gain_bias: Initial value for the bias (shift) parameter. Defaults to 1.0.
   :type gain_bias: float


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the BatchNorm2d layer.

      :returns:

                A string containing the layer's information, typically delegated
                     to the C++ backend implementation.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the BatchNorm2d layer.

      :returns: The name of the layer, typically delegated to the C++ backend implementation.
      :rtype: str



   .. py:method:: init_weight_bias()

      Initializes the learnable weight (scale/gain) and bias (shift/offset)
      parameters of the batch normalization layer. This operation is
      delegated to the C++ backend.



.. py:class:: Conv2d(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, stride: int = 1, padding: int = 0, padding_type: int = 1, in_width: int = 0, in_height: int = 0, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies a 2D convolution operation.

   This layer performs a convolution operation, which is a fundamental building block
   in convolutional neural networks (CNNs). It slides a kernel (or filter) over
   an input tensor to produce an output tensor.

   :param in_channels: Number of input channels.
   :type in_channels: int
   :param out_channels: Number of output channels.
   :type out_channels: int
   :param kernel_size: Size of the convolutional kernel.
   :type kernel_size: int
   :param bias: Whether to include a learnable bias term. Defaults to True.
   :type bias: bool
   :param stride: The step size of the kernel. Defaults to 1.
   :type stride: int
   :param padding: Amount of zero-padding added to the input. Defaults to 0.
   :type padding: int
   :param padding_type: Type of padding. Defaults to 1 (likely 'zeros' or similar).
   :type padding_type: int
   :param in_width: Input width. If 0, it might be inferred or set by the backend. Defaults to 0.
   :type in_width: int
   :param in_height: Input height. If 0, it might be inferred or set by the backend. Defaults to 0.
   :type in_height: int
   :param gain_weight: Initial value for the gain (scale) parameter of weights. Defaults to 1.0.
   :type gain_weight: float
   :param gain_bias: Initial value for the gain (scale) parameter of biases. Defaults to 1.0.
   :type gain_bias: float
   :param init_method: Method used for initializing weights. Defaults to "He".
   :type init_method: str


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the Conv2d layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the Conv2d layer.

      :returns: The name of the layer.
      :rtype: str



   .. py:method:: init_weight_bias()

      Initializes the learnable weight (kernel) and bias parameters of the convolutional layer.
      This initialization is delegated to the C++ backend using the 'init_method' specified (e.g., "He").



.. py:class:: ConvTranspose2d(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, stride: int = 1, padding: int = 0, padding_type: int = 1, in_width: int = 0, in_height: int = 0, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Applies a 2D transposed convolution operation (also known as deconvolution).

   This layer performs a transposed convolution, which is used in tasks
   like image generation or segmentation to upsample feature maps. It
   reverses the convolution operation, increasing the spatial dimensions of the input.

   :param in_channels: Number of input channels.
   :type in_channels: int
   :param out_channels: Number of output channels.
   :type out_channels: int
   :param kernel_size: Size of the convolutional kernel.
   :type kernel_size: int
   :param bias: Whether to include a learnable bias term. Defaults to True.
   :type bias: bool
   :param stride: The step size of the kernel. Defaults to 1.
   :type stride: int
   :param padding: Amount of zero-padding added to the input. Defaults to 0.
   :type padding: int
   :param padding_type: Type of padding. Defaults to 1 (likely 'zeros' or similar).
   :type padding_type: int
   :param in_width: Input width. If 0, it might be inferred or set by the backend. Defaults to 0.
   :type in_width: int
   :param in_height: Input height. If 0, it might be inferred or set by the backend. Defaults to 0.
   :type in_height: int
   :param gain_weight: Initial value for the gain (scale) parameter of weights. Defaults to 1.0.
   :type gain_weight: float
   :param gain_bias: Initial value for the gain (scale) parameter of biases. Defaults to 1.0.
   :type gain_bias: float
   :param init_method: Method used for initializing weights. Defaults to "He".
   :type init_method: str


   .. py:method:: get_layer_info() -> str

      Retrieves detailed information about the ConvTranspose2d layer.

      :returns: A string containing the layer's information.
      :rtype: str



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the ConvTranspose2d layer.

      :returns: The name of the layer.
      :rtype: str



   .. py:method:: init_weight_bias()

      Initializes the learnable weight and bias parameters of the transposed convolutional layer.



.. py:class:: BaseDeltaStates(size: Optional[int] = None, block_size: Optional[int] = None)

   Represents the base delta states, acting as a Python wrapper for the C++ backend.
   This class manages the change in mean (delta_mu) and change in variance (delta_var)
   induced by the update step.


   .. py:property:: delta_mu
      :type: List[float]


      Gets or sets the change in mean of the delta states (delta_mu).


   .. py:property:: delta_var
      :type: List[float]


      Gets or sets the change in variance of the delta states (delta_var).


   .. py:property:: size
      :type: int


      Gets the size of the delta states.


   .. py:property:: block_size
      :type: int


      Gets the block size of the delta states.


   .. py:property:: actual_size
      :type: int


      Gets the actual size of the delta states.


   .. py:method:: get_name() -> str

      Gets the name of the delta states type.

      :returns: The name of the delta states type.
      :rtype: str



   .. py:method:: reset_zeros() -> None

      Reset all delta_mu and delta_var to zeros.



   .. py:method:: copy_from(source: BaseDeltaStates, num_data: int = -1) -> None

      Copy values of delta_mu and delta_var from another delta states object.

      :param source: The source delta states object to copy from.
      :type source: BaseDeltaStates
      :param num_data: The number of data points to copy. Defaults to -1 (all).
      :type num_data: int



   .. py:method:: set_size(new_size: int, new_block_size: int) -> str

      Sets a new size and block size for the delta states.

      :param new_size: The new size.
      :type new_size: int
      :param new_block_size: The new block size.
      :type new_block_size: int

      :returns: A message indicating the success or failure of the operation.
      :rtype: str



.. py:class:: BaseHiddenStates(size: Optional[int] = None, block_size: Optional[int] = None)

   Represents the base hidden states, acting as a Python wrapper for the C++ backend.
   This class manages the mean (mu_a), variance (var_a), and Jacobian (jcb) of hidden states.


   .. py:property:: mu_a
      :type: List[float]


      Gets or sets the mean of the hidden states (mu_a).


   .. py:property:: var_a
      :type: List[float]


      Gets or sets the variance of the hidden states (var_a).


   .. py:property:: jcb
      :type: List[float]


      Gets or sets the Jacobian of the hidden states (jcb).


   .. py:property:: size
      :type: int


      Gets the size of the hidden states.


   .. py:property:: block_size
      :type: int


      Gets the block size of the hidden states.


   .. py:property:: actual_size
      :type: int


      Gets the actual size of the hidden states.


   .. py:method:: set_input_x(mu_x: List[float], var_x: List[float], block_size: int)

      Sets the input for the hidden states.

      :param mu_x: The mean of the input x.
      :type mu_x: List[float]
      :param var_x: The variance of the input x.
      :type var_x: List[float]
      :param block_size: The block size for the input.
      :type block_size: int



   .. py:method:: get_name() -> str

      Gets the name of the hidden states type.

      :returns: The name of the hidden states type.
      :rtype: str



   .. py:method:: set_size(new_size: int, new_block_size: int) -> str

      Sets a new size and block size for the hidden states.

      :param new_size: The new size.
      :type new_size: int
      :param new_block_size: The new block size.
      :type new_block_size: int

      :returns: A message indicating the success or failure of the operation.
      :rtype: str



.. py:class:: HRCSoftmax

   Hierarchical softmax wrapper from the CPP backend.

   .. attribute:: obs

      A fictive observation \in [-1, 1].

      :type: List[float]

   .. attribute:: idx

      Indices assigned to each label.

      :type: List[int]

   .. attribute:: num_obs

      Number of indices for each label.

      :type: int

   .. attribute:: len

      Length of an observation (e.g., 10 labels -> len(obs) = 11).

      :type: int


   .. py:property:: obs
      :type: List[float]


      Gets or sets the observations for hierarchical softmax.


   .. py:property:: idx
      :type: List[int]


      Gets or sets the indices assigned to each label.


   .. py:property:: num_obs
      :type: int


      Gets or sets the number of observations for each label.


   .. py:property:: len
      :type: int


      Gets or sets the length of an observation.


.. py:class:: DDPConfig(device_ids: List[int], backend: str = 'nccl', rank: int = 0, world_size: int = 1)

   Configuration for Distributed Data Parallel (DDP) training.

   This class holds all the necessary settings for initializing a distributed
   process group.


   .. py:property:: device_ids
      :type: List[int]


      The list of GPU device IDs.


   .. py:property:: backend
      :type: str


      The distributed communication backend (e.g., 'nccl').


   .. py:property:: rank
      :type: int


      The rank of the current process in the distributed group.


   .. py:property:: world_size
      :type: int


      The total number of processes in the distributed group.


.. py:class:: DDPSequential(model: pytagi.nn.sequential.Sequential, config: DDPConfig, average: bool = True)

   A wrapper for `Sequential` models to enable Distributed Data Parallel (DDP) training.

   This class handles gradient synchronization and parameter updates across multiple
   processes, allowing for scalable training on multiple GPUs.


   .. py:property:: output_z_buffer
      :type: pytagi.nn.data_struct.BaseHiddenStates


      The output hidden states buffer from the forward pass of the underlying model.


   .. py:property:: input_delta_z_buffer
      :type: pytagi.nn.data_struct.BaseDeltaStates


      The input delta states buffer for the backward pass of the underlying model.


   .. py:method:: __call__(mu_x: numpy.ndarray, var_x: numpy.ndarray = None) -> Tuple[numpy.ndarray, numpy.ndarray]

      A convenient alias for the forward pass.

      :param mu_x: The mean of the input data for the current process.
      :type mu_x: np.ndarray
      :param var_x: The variance of the input data for the current process. Defaults to None.
      :type var_x: np.ndarray, optional
      :return: A tuple containing the mean and variance of the model's output.
      :rtype: Tuple[np.ndarray, np.ndarray]



   .. py:method:: forward(mu_x: numpy.ndarray, var_x: numpy.ndarray = None) -> Tuple[numpy.ndarray, numpy.ndarray]

      Performs a forward pass on the local model replica.

      :param mu_x: The mean of the input data.
      :type mu_x: np.ndarray
      :param var_x: The variance of the input data. Defaults to None.
      :type var_x: np.ndarray, optional
      :return: A tuple containing the mean and variance of the output.
      :rtype: Tuple[np.ndarray, np.ndarray]



   .. py:method:: backward()

      Performs a backward pass and synchronizes gradients across all processes.



   .. py:method:: step()

      Performs a single parameter update step based on the synchronized gradients.



   .. py:method:: train()

      Sets the model to training mode.



   .. py:method:: eval()

      Sets the model to evaluation mode.



   .. py:method:: barrier()

      Synchronizes all processes.

      Blocks until all processes in the distributed group have reached this point.



   .. py:method:: get_outputs() -> Tuple[numpy.ndarray, numpy.ndarray]

      Gets the outputs from the last forward pass on the local replica.

      :return: A tuple containing the mean and variance of the output.
      :rtype: Tuple[np.ndarray, np.ndarray]



   .. py:method:: output_to_host()

      Copies the output data from the device to the host (CPU memory).



   .. py:method:: get_device_with_index() -> str

      Gets the device string for the current process, including its index.

      :return: The device string, e.g., 'cuda:0'.
      :rtype: str



.. py:class:: LayerBlock(*layers: pytagi.nn.base_layer.BaseLayer)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   A stack of different layers derived from BaseLayer


   .. py:method:: switch_to_cuda()

      Convert all layers to cuda layer



   .. py:property:: layers
      :type: None


      Get layers


.. py:class:: LayerNorm(normalized_shape: List[int], eps: float = 0.0001, bias: bool = True)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Implements Layer Normalization by normalizing the inputs across the
   features dimension. It inherits from BaseLayer.


   .. py:method:: get_layer_info() -> str

      Retrieves a descriptive string containing information about the layer's
      configuration (e.g., its shape and parameters) from the C++ backend.



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer (e.g., 'LayerNorm') from the C++ backend.



   .. py:method:: init_weight_bias()

      Initializes the layer's internal parameters, specifically the learnable
      scale (gamma) and, if 'bias' is True, the learnable offset (beta).
      This task is delegated to the C++ backend.



.. py:class:: Linear(input_size: int, output_size: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Implements a **Fully-connected layer**, also known as a dense layer.
   This layer performs a linear transformation on the input data:
   :math:`y = xW^T + b`, where :math:`x` is the input, :math:`W` is the weight matrix,
   and :math:`b` is the optional bias vector. It inherits from BaseLayer.


   .. py:method:: get_layer_info() -> str

      Retrieves a descriptive string containing information about the layer's
      configuration (e.g., input/output size, whether bias is used) from the
      C++ backend.



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer (e.g., 'Linear')
      from the C++ backend.



   .. py:method:: init_weight_bias()

      Initializes the layer's parameters—the weight matrix (:math:`W`) and the
      optional bias vector (:math:`b`)—using the specified initialization method
      and gain factors. This task is delegated to the C++ backend.



.. py:class:: LSTM(input_size: int, output_size: int, seq_len: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   A **Long Short-Term Memory (LSTM)** layer for RNNs. It inherits from BaseLayer.


   .. py:method:: get_layer_info() -> str

      Retrieves a descriptive string containing information about the layer's
      configuration (e.g., input/output size, sequence length) from the
      C++ backend.



   .. py:method:: get_layer_name() -> str

      Retrieves the name of the layer (e.g., 'LSTM') from the C++ backend.



   .. py:method:: init_weight_bias()

      Initializes the various weight matrices and bias vectors used by the
      LSTM's gates (input, forget, output) and cell state updates, using
      the specified method and gain factors. This task is delegated to the
      C++ backend.



.. py:class:: OutputUpdater(model_device: str)

   A utility to compute the error signal (delta states) for the output layer.

   This class calculates the difference between the model's predictions and the
   observations, which is essential for performing the backward pass
   to update the model's parameters. It wraps the C++/CUDA backend `cutagi.OutputUpdater`.


   .. py:method:: update(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, var_obs: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)

      Computes the delta states based on observations.

      This method is used for homoscedastic regression where the observation
      variance is known and provided.

      :param output_states: The hidden states (mean and variance) of the model's output layer.
      :type output_states: pytagi.nn.data_struct.BaseHiddenStates
      :param mu_obs: The mean of the ground truth observations.
      :type mu_obs: np.ndarray
      :param var_obs: The variance of the ground truth observations.
      :type var_obs: np.ndarray
      :param delta_states: The delta states object to be updated with the computed error signal.
      :type delta_states: pytagi.nn.data_struct.BaseDeltaStates



   .. py:method:: update_using_indices(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, var_obs: numpy.ndarray, selected_idx: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)

      Computes the delta states for a selected subset of outputs.

      This is useful in scenarios like hierarchical softmax or when only
      a sparse set of outputs needs to be updated.

      :param output_states: The hidden states of the model's output layer.
      :type output_states: pytagi.nn.data_struct.BaseHiddenStates
      :param mu_obs: The mean of the ground truth observations.
      :type mu_obs: np.ndarray
      :param var_obs: The variance of the ground truth observations.
      :type var_obs: np.ndarray
      :param selected_idx: An array of indices specifying which output neurons to update.
      :type selected_idx: np.ndarray
      :param delta_states: The delta states object to be updated with the computed error signal.
      :type delta_states: pytagi.nn.data_struct.BaseDeltaStates



   .. py:method:: update_heteros(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)

      Computes delta states for heteroscedastic regression.

      In this case, the model is expected to predict both the mean and the variance
      of the output. The predicted variance is taken from the `output_states`.

      :param output_states: The hidden states of the model's output layer. The model's
                            predicted variance is sourced from here.
      :type output_states: pytagi.nn.data_struct.BaseHiddenStates
      :param mu_obs: The mean of the ground truth observations.
      :type mu_obs: np.ndarray
      :param delta_states: The delta states object to be updated with the computed error signal.
      :type delta_states: pytagi.nn.data_struct.BaseDeltaStates



   .. py:property:: device
      :type: str


      The computational device ('cpu' or 'cuda') the updater is on.


.. py:class:: AvgPool2d(kernel_size: int, stride: int = -1, padding: int = 0, padding_type: int = 0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   2D Average Pooling Layer.

   This layer performs 2D average pooling operation. It wraps the C++/CUDA backend
   `cutagi.AvgPool2d`.


   .. py:method:: get_layer_info() -> str

      Returns a string containing information about the layer.



   .. py:method:: get_layer_name() -> str

      Returns the name of the layer (e.g., 'AvgPool2d').



.. py:class:: MaxPool2d(kernel_size: int, stride: int = 1, padding: int = 0, padding_type: int = 0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   2D Max Pooling Layer.

   This layer performs 2D max pooling operation based on the input expected values.
   It wraps the C++/CUDA backend `cutagi.MaxPool2d`.


   .. py:method:: get_layer_info() -> str

      Returns a string containing information about the layer.



   .. py:method:: get_layer_name() -> str

      Returns the name of the layer (e.g., 'MaxPool2d').



.. py:class:: ResNetBlock(main_block: Union[pytagi.nn.base_layer.BaseLayer, pytagi.nn.layer_block.LayerBlock], shortcut: Union[pytagi.nn.base_layer.BaseLayer, pytagi.nn.layer_block.LayerBlock] = None)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   A Residual Network (ResNet) block structure.

   This class implements the core structure of a ResNet block, consisting of a
   **main block** (which performs the main transformations) and an optional
   **shortcut** connection (which adds the input to the main block's output).
   It wraps the C++/CUDA backend `cutagi.ResNetBlock`.


   .. py:method:: init_shortcut_state() -> None

      Initializes the hidden state buffers for the shortcut layer.



   .. py:method:: init_shortcut_delta_state() -> None

      Initializes the delta state buffers (error signals) for the shortcut layer.



   .. py:method:: init_input_buffer() -> None

      Initializes the input state buffer used to hold the input for both the main block and the shortcut.



   .. py:property:: main_block
      :type: pytagi.nn.layer_block.LayerBlock


      Gets the **main block** component of the ResNet block.


   .. py:property:: shortcut
      :type: pytagi.nn.base_layer.BaseLayer


      Gets the **shortcut** component of the ResNet block.


   .. py:property:: input_z
      :type: pytagi.nn.data_struct.BaseHiddenStates


      Gets the buffered input hidden states (mean and variance) for the block.


   .. py:property:: input_delta_z
      :type: pytagi.nn.data_struct.BaseDeltaStates


      Gets the delta states (error signals) associated with the block's input.


   .. py:property:: shortcut_output_z
      :type: pytagi.nn.data_struct.BaseHiddenStates


      Gets the output hidden states (mean and variance) from the shortcut layer.


   .. py:property:: shortcut_output_delta_z
      :type: pytagi.nn.data_struct.BaseDeltaStates


      Gets the delta states (error signals) associated with the shortcut layer's output.


.. py:class:: Sequential(*layers: pytagi.nn.base_layer.BaseLayer)

   A sequential container for layers.

   Layers are added to the container in the order they are passed in the
   constructor. This class acts as a Python wrapper for the C++/CUDA
   backend `cutagi.Sequential`.

   .. rubric:: Example

   >>> import pytagi.nn as nn
   >>> model = nn.Sequential(
   ...     nn.Linear(10, 20),
   ...     nn.ReLU(),
   ...     nn.Linear(20, 5)
   ... )
   >>> mu_in = np.random.randn(1, 10)
   >>> var_in = np.abs(np.random.randn(1, 10))
   >>> mu_out, var_out = model(mu_in, var_in)


   .. py:method:: __call__(mu_x: numpy.ndarray, var_x: numpy.ndarray = None) -> Tuple[numpy.ndarray, numpy.ndarray]

      An alias for the forward pass.

      :param mu_x: The mean of the input data.
      :type mu_x: np.ndarray
      :param var_x: The variance of the input data. Defaults to None.
      :type var_x: np.ndarray, optional
      :return: A tuple containing the mean and variance of the output.
      :rtype: Tuple[np.ndarray, np.ndarray]



   .. py:property:: layers
      :type: List[pytagi.nn.base_layer.BaseLayer]


      The list of layers in the model.


   .. py:property:: output_z_buffer
      :type: pytagi.nn.data_struct.BaseHiddenStates


      The output hidden states buffer from the forward pass.


   .. py:property:: input_delta_z_buffer
      :type: pytagi.nn.data_struct.BaseDeltaStates


      The input delta states buffer used in the backward pass.


   .. py:property:: output_delta_z_buffer
      :type: pytagi.nn.data_struct.BaseDeltaStates


      The output delta states buffer from the backward pass.


   .. py:property:: z_buffer_size
      :type: int


      The size of the hidden state (`z`) buffer.


   .. py:property:: z_buffer_block_size
      :type: int


      The block size of the hidden state (`z`) buffer.


   .. py:property:: device
      :type: str


      The computational device ('cpu' or 'cuda') the model is on.


   .. py:property:: input_state_update
      :type: bool


      Flag indicating if the input state should be updated.


   .. py:property:: num_samples
      :type: int


      The number of samples used for Monte Carlo estimation. This is used
      for debugging purposes


   .. py:method:: to_device(device: str)

      Moves the model and its parameters to a specified device.

      :param device: The target device, e.g., 'cpu' or 'cuda:0'.
      :type device: str



   .. py:method:: params_to_device()

      Moves the model parameters to the currently configured CUDA device.



   .. py:method:: params_to_host()

      Moves the model parameters from the CUDA device to the host (CPU).



   .. py:method:: set_threads(num_threads: int)

      Sets the number of CPU threads to use for computation.

      :param num_threads: The number of threads.
      :type num_threads: int



   .. py:method:: train()

      Sets the model to training mode.



   .. py:method:: eval()

      Sets the model to evaluation mode.



   .. py:method:: forward(mu_x: numpy.ndarray, var_x: numpy.ndarray = None) -> Tuple[numpy.ndarray, numpy.ndarray]

      Performs a forward pass through the network.

      :param mu_x: The mean of the input data.
      :type mu_x: np.ndarray
      :param var_x: The variance of the input data. Defaults to None.
      :type var_x: np.ndarray, optional
      :return: A tuple containing the mean and variance of the output.
      :rtype: Tuple[np.ndarray, np.ndarray]



   .. py:method:: backward()

      Performs a backward pass to update the network parameters.



   .. py:method:: smoother() -> Tuple[numpy.ndarray, numpy.ndarray]

      Performs a smoother pass (e.g., Rauch-Tung-Striebel smoother).

      This is used with the SLSTM to refine estimates by running backwards
      through time.

      :return: A tuple containing the mean and variance of the smoothed output.
      :rtype: Tuple[np.ndarray, np.ndarray]



   .. py:method:: step()

      Performs a single step of inference to update the parameters.



   .. py:method:: reset_lstm_states()

      Resets the hidden and cell states of all LSTM layers in the model.



   .. py:method:: output_to_host() -> List[float]

      Copies the raw output data from the device to the host.

      :return: A list of floating-point values representing the flattened output.
      :rtype: List[float]



   .. py:method:: delta_z_to_host() -> List[float]

      Copies the raw delta Z (error signal) data from the device to the host.

      :return: A list of floating-point values representing the flattened delta Z.
      :rtype: List[float]



   .. py:method:: set_delta_z(delta_mu: numpy.ndarray, delta_var: numpy.ndarray)

      Sets the delta Z (error signal) on the device for the backward pass.

      :param delta_mu: The mean of the error signal.
      :type delta_mu: np.ndarray
      :param delta_var: The variance of the error signal.
      :type delta_var: np.ndarray



   .. py:method:: get_layer_stack_info() -> str

      Gets a string representation of the layer stack architecture.

      :return: A descriptive string of the model's layers.
      :rtype: str



   .. py:method:: preinit_layer()

      Pre-initializes the layers in the model.



   .. py:method:: get_neg_var_w_counter() -> dict

      Counts the number of negative variance weights in each layer.

      :return: A dictionary where keys are layer names and values are the counts
               of negative variances.
      :rtype: dict



   .. py:method:: save(filename: str)

      Saves the model's state to a binary file.

      :param filename: The path to the file where the model will be saved.
      :type filename: str



   .. py:method:: load(filename: str)

      Loads the model's state from a binary file.

      :param filename: The path to the file from which to load the model.
      :type filename: str



   .. py:method:: save_csv(filename: str)

      Saves the model parameters to a CSV file.

      :param filename: The base path for the CSV file(s).
      :type filename: str



   .. py:method:: load_csv(filename: str)

      Loads the model parameters from a CSV file.

      :param filename: The base path of the CSV file(s).
      :type filename: str



   .. py:method:: parameters() -> List[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]]

      Gets all model parameters.

      :return: A list where each element is a tuple containing the parameters
               for a layer: (mu_w, var_w, mu_b, var_b).
      :rtype: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]



   .. py:method:: load_state_dict(state_dict: dict)

      Loads the model's parameters from a state dictionary.

      :param state_dict: A dictionary containing the model's state.
      :type state_dict: dict



   .. py:method:: state_dict() -> dict

      Gets the model's parameters as a state dictionary.

      :return: A dictionary where each key is the layer name and the value is a
               tuple of parameters: (mu_w, var_w, mu_b, var_b).
      :rtype: dict



   .. py:method:: params_from(other: Sequential)

      Copies parameters from another Sequential model.

      :param other: The source model from which to copy parameters.
      :type other: Sequential



   .. py:method:: get_outputs() -> Tuple[numpy.ndarray, numpy.ndarray]

      Gets the outputs from the last forward pass.

      :return: A tuple containing the mean and variance of the output.
      :rtype: Tuple[np.ndarray, np.ndarray]



   .. py:method:: get_outputs_smoother() -> Tuple[numpy.ndarray, numpy.ndarray]

      Gets the outputs from the last smoother pass.

      :return: A tuple containing the mean and variance of the smoothed output.
      :rtype: Tuple[np.ndarray, np.ndarray]



   .. py:method:: get_input_states() -> Tuple[numpy.ndarray, numpy.ndarray]

      Gets the input states of the model.

      :return: A tuple containing the mean and variance of the input states.
      :rtype: Tuple[np.ndarray, np.ndarray]



   .. py:method:: get_norm_mean_var() -> dict

      Gets the mean and variance from normalization layers.

      :return: A dictionary where each key is a normalization layer name and
               the value is a tuple of four arrays:
               (mu_batch, var_batch, mu_ema_batch, var_ema_batch).
      :rtype: dict



   .. py:method:: get_lstm_states() -> dict

      Gets the states from all LSTM layers.

      :return: A dictionary where each key is the layer index and the value
               is a 4-tuple of numpy arrays:
               (mu_h_prior, var_h_prior, mu_c_prior, var_c_prior).
      :rtype: dict



   .. py:method:: set_lstm_states(states: dict) -> None

      Sets the states for all LSTM layers.

      :param states: A dictionary mapping layer indices to a 4-tuple of
                     numpy arrays: (mu_h_prior, var_h_prior, mu_c_prior, var_c_prior).
      :type states: dict



.. py:class:: SLinear(input_size: int, output_size: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Smoother Linear layer for the SLSTM architecture.

   This layer performs a linear transformation (:math:`y = xW^T + b'), specifically designed
   to be used within SLSTM where a hidden- and cell-state smoothing through time is applied.
   It wraps the C++/CUDA backend `cutagi.SLinear`.


   .. py:method:: get_layer_info() -> str

      Returns a string containing information about the layer's configuration (sizes, bias, etc.).



   .. py:method:: get_layer_name() -> str

      Returns the name of the layer (e.g., 'SLinear').



   .. py:method:: init_weight_bias()

      Initializes the layer's weight matrix and bias vector based on the configured method.



.. py:class:: SLSTM(input_size: int, output_size: int, seq_len: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Smoothing Long Short-Term Memory (LSTM) layer.

   This layer is a variation of the standard LSTM, incorporating a mechanism
   for **smoothing** the hidden- and cell-states. It wraps the C++/CUDA backend
   `cutagi.SLSTM`.


   .. py:method:: get_layer_info() -> str

      Returns a string containing detailed information about the layer's configuration.



   .. py:method:: get_layer_name() -> str

      Returns the name of the layer (e.g., 'SLSTM').



   .. py:method:: init_weight_bias()

      Initializes all the layer's internal weight matrices and bias vectors (for gates and cell) based on the configured method.
