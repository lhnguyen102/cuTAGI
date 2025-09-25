pytagi.nn
=========

.. py:module:: pytagi.nn


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


   ClosedFormSoftmax


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: EvenExp

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   EvenExp


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: LeakyReLU

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Leaky ReLU


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: MixtureReLU

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Mixture ReLU


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: MixtureSigmoid

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Mixture Sigmoid


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: MixtureTanh

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Mixture Tanh


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: ReLU

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   ReLU


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: Remax

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Remax


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: Sigmoid

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Sigmoid


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: Softmax

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Softmax


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: Softplus

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Softplus


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: Tanh

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Tanh


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: BaseLayer

   Base layer


   .. py:method:: to_cuda()


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: get_max_num_states() -> int


   .. py:property:: input_size
      :type: int



   .. py:property:: output_size
      :type: int



   .. py:property:: in_width
      :type: int



   .. py:property:: in_height
      :type: int



   .. py:property:: in_channels
      :type: int



   .. py:property:: out_width
      :type: int



   .. py:property:: out_height
      :type: int



   .. py:property:: out_channels
      :type: int



   .. py:property:: bias
      :type: bool



   .. py:property:: num_weights
      :type: int



   .. py:property:: num_biases
      :type: int



   .. py:property:: mu_w
      :type: numpy.ndarray



   .. py:property:: var_w
      :type: numpy.ndarray



   .. py:property:: mu_b
      :type: numpy.ndarray



   .. py:property:: var_b
      :type: numpy.ndarray



   .. py:property:: delta_mu_w
      :type: numpy.ndarray



   .. py:property:: delta_var_w
      :type: numpy.ndarray



   .. py:property:: delta_mu_b
      :type: numpy.ndarray



   .. py:property:: delta_var_b
      :type: numpy.ndarray



   .. py:property:: num_threads
      :type: int



   .. py:property:: training
      :type: bool



   .. py:property:: device
      :type: bool



.. py:class:: BatchNorm2d(num_features: int, eps: float = 1e-05, momentum: float = 0.9, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Batch normalization


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()


.. py:class:: Conv2d(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, stride: int = 1, padding: int = 0, padding_type: int = 1, in_width: int = 0, in_height: int = 0, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Convolutional layer


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()


.. py:class:: ConvTranspose2d(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, stride: int = 1, padding: int = 0, padding_type: int = 1, in_width: int = 0, in_height: int = 0, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Tranposed convolutional layer


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()


.. py:class:: BaseDeltaStates(size: Optional[int] = None, block_size: Optional[int] = None)

   .. py:property:: delta_mu
      :type: List[float]



   .. py:property:: delta_var
      :type: List[float]



   .. py:property:: size
      :type: int



   .. py:property:: block_size
      :type: int



   .. py:property:: actual_size
      :type: int



   .. py:method:: get_name() -> str


   .. py:method:: reset_zeros() -> None

      Reset all delta_mu and delta_var to zeros



   .. py:method:: copy_from(source: BaseDeltaStates, num_data: int = -1) -> None

      Copy values of delta_mu and delta_var from delta states



   .. py:method:: set_size(new_size: int, new_block_size: int) -> str


.. py:class:: BaseHiddenStates(size: Optional[int] = None, block_size: Optional[int] = None)

   .. py:property:: mu_a
      :type: List[float]



   .. py:property:: var_a
      :type: List[float]



   .. py:property:: jcb
      :type: List[float]



   .. py:property:: size
      :type: int



   .. py:property:: block_size
      :type: int



   .. py:property:: actual_size
      :type: int



   .. py:method:: set_input_x(mu_x: List[float], var_x: List[float], block_size: int)


   .. py:method:: get_name() -> str


   .. py:method:: set_size(new_size: int, new_block_size: int) -> str


.. py:class:: HRCSoftmax

   Hierarchical softmax wrapper from the CPP backend. Further details can be
   found here https://building-babylon.net/2017/08/01/hierarchical-softmax

   .. attribute:: obs

      A fictive observation \in [-1, 1]

   .. attribute:: idx

      Indices assigned to each label

   .. attribute:: num_obs

      Number of indices for each label

   .. attribute:: len

      Length of an observation e.g 10 labels -> len(obs) = 11


   .. py:property:: obs
      :type: List[float]



   .. py:property:: idx
      :type: List[int]



   .. py:property:: num_obs
      :type: int



   .. py:property:: len
      :type: int



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


   A stack of different layer derived from BaseLayer


   .. py:method:: switch_to_cuda()

      Convert all layers to cuda layer



   .. py:property:: layers
      :type: None


      Get layers


.. py:class:: LayerNorm(normalized_shape: List[int], eps: float = 0.0001, bias: bool = True)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Layer normalization


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()


.. py:class:: Linear(input_size: int, output_size: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Fully-connected layer

   :param input_size: Input size of the layer
   :type input_size: int

   .. attribute:: input_size

      Input size of the layer

      :type: int

   .. attribute:: output_size

      Output size of the layer

      :type: int

   .. attribute:: bias

      If True, adding biases along with the weights

      :type: boolen


   .. py:method:: get_layer_info() -> str

      get layer information



   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()


.. py:class:: LSTM(input_size: int, output_size: int, seq_len: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   LSTM layer


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()


.. py:class:: OutputUpdater(model_device: str)

   A utility to compute the error signal (delta states) for the output layer.

   This class calculates the difference between the model's predictions and the
   ground truth observations, which is essential for initiating the backward pass
   to update the model's parameters. It wraps the C++/CUDA backend `cutagi.OutputUpdater`.


   .. py:method:: update(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, var_obs: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)

      Computes the delta states based on observations.

      This method is used for homoscedastic regression where the observation
      variance is known and provided.

      :param output_states: The hidden states (mean and variance) of the model's output layer.
      :type output_states: BaseHiddenStates
      :param mu_obs: The mean of the ground truth observations.
      :type mu_obs: np.ndarray
      :param var_obs: The variance of the ground truth observations.
      :type var_obs: np.ndarray
      :param delta_states: The delta states object to be updated with the computed error signal.
      :type delta_states: BaseDeltaStates



   .. py:method:: update_using_indices(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, var_obs: numpy.ndarray, selected_idx: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)

      Computes the delta states for a selected subset of outputs.

      This is useful in scenarios like hierarchical softmax or when only
      a sparse set of outputs needs to be updated.

      :param output_states: The hidden states of the model's output layer.
      :type output_states: BaseHiddenStates
      :param mu_obs: The mean of the ground truth observations.
      :type mu_obs: np.ndarray
      :param var_obs: The variance of the ground truth observations.
      :type var_obs: np.ndarray
      :param selected_idx: An array of indices specifying which output neurons to update.
      :type selected_idx: np.ndarray
      :param delta_states: The delta states object to be updated with the computed error signal.
      :type delta_states: BaseDeltaStates



   .. py:method:: update_heteros(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)

      Computes delta states for heteroscedastic regression.

      In this case, the model is expected to predict both the mean and the variance
      of the output. The predicted variance is taken from the `output_states`.

      :param output_states: The hidden states of the model's output layer. The model's
                            predicted variance is sourced from here.
      :type output_states: BaseHiddenStates
      :param mu_obs: The mean of the ground truth observations.
      :type mu_obs: np.ndarray
      :param delta_states: The delta states object to be updated with the computed error signal.
      :type delta_states: BaseDeltaStates



   .. py:property:: device
      :type: str


      The computational device ('cpu' or 'cuda') the updater is on.


.. py:class:: AvgPool2d(kernel_size: int, stride: int = -1, padding: int = 0, padding_type: int = 0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Average Pooling layer


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: MaxPool2d(kernel_size: int, stride: int = 1, padding: int = 0, padding_type: int = 0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Max Pooling layer


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: ResNetBlock(main_block: Union[pytagi.nn.base_layer.BaseLayer, pytagi.nn.layer_block.LayerBlock], shortcut: Union[pytagi.nn.base_layer.BaseLayer, pytagi.nn.layer_block.LayerBlock] = None)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   A residual architecture contains a main block and a shortcut layer


   .. py:method:: init_shortcut_state() -> None

      Initialize state buffer for shortcut



   .. py:method:: init_shortcut_delta_state() -> None

      Initialize update values for state buffer for the shortcut



   .. py:method:: init_input_buffer() -> None

      Initialize input state buffer to hold temporary state



   .. py:property:: main_block
      :type: pytagi.nn.layer_block.LayerBlock


      Set main block


   .. py:property:: shortcut
      :type: pytagi.nn.base_layer.BaseLayer


      Set shortcut


   .. py:property:: input_z
      :type: pytagi.nn.data_struct.BaseHiddenStates


      Get output hidden states


   .. py:property:: input_delta_z
      :type: pytagi.nn.data_struct.BaseDeltaStates


      Get update values for input states


   .. py:property:: shortcut_output_z
      :type: pytagi.nn.data_struct.BaseHiddenStates


      Get output hidden states for shortcut


   .. py:property:: shortcut_output_delta_z
      :type: pytagi.nn.data_struct.BaseDeltaStates


      Get update values for output hidden states for shortcut


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

      A convenient alias for the forward pass.

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


      The number of samples used for Monte Carlo estimation.


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

      This is typically used in state-space models to refine estimates.

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


   Smoothering Linear layer


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()


.. py:class:: SLSTM(input_size: int, output_size: int, seq_len: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Smoothing LSTM layer


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()
