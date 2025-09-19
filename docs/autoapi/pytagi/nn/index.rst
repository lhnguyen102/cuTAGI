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

   Configuration for distributed training


   .. py:property:: device_ids
      :type: List[int]



   .. py:property:: backend
      :type: str



   .. py:property:: rank
      :type: int



   .. py:property:: world_size
      :type: int



.. py:class:: DDPSequential(model: pytagi.nn.sequential.Sequential, config: DDPConfig, average: bool = True)

   Distributed training wrapper for Sequential models


   .. py:property:: output_z_buffer
      :type: pytagi.nn.data_struct.BaseHiddenStates


      Get the output hidden states


   .. py:property:: input_delta_z_buffer
      :type: pytagi.nn.data_struct.BaseDeltaStates


      Get the delta hidden states


   .. py:method:: __call__(mu_x: numpy.ndarray, var_x: numpy.ndarray = None) -> Tuple[numpy.ndarray, numpy.ndarray]

      Perform a forward pass



   .. py:method:: forward(mu_x: numpy.ndarray, var_x: numpy.ndarray = None) -> Tuple[numpy.ndarray, numpy.ndarray]

      Perform a forward pass



   .. py:method:: backward()

      Perform a backward pass



   .. py:method:: step()

      Perform a parameter update step



   .. py:method:: train()

      Set the model in training mode



   .. py:method:: eval()

      Set the model in evaluation mode



   .. py:method:: barrier()

      Synchronize all processes



   .. py:method:: get_outputs() -> Tuple[numpy.ndarray, numpy.ndarray]

      Get the outputs of the model



   .. py:method:: output_to_host()

      Copy the output to the host



   .. py:method:: get_device_with_index() -> str

      Get the device with index



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

   .. py:method:: update(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, var_obs: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)


   .. py:method:: update_using_indices(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, var_obs: numpy.ndarray, selected_idx: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)


   .. py:method:: update_heteros(output_states: pytagi.nn.data_struct.BaseHiddenStates, mu_obs: numpy.ndarray, delta_states: pytagi.nn.data_struct.BaseDeltaStates)


   .. py:property:: device
      :type: str



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

   Adding neural networks in a sequence mode


   .. py:method:: __call__(mu_x: numpy.ndarray, var_x: numpy.ndarray = None) -> Tuple[numpy.ndarray, numpy.ndarray]


   .. py:property:: layers
      :type: List[pytagi.nn.base_layer.BaseLayer]


      Get the layers of the model.


   .. py:property:: output_z_buffer
      :type: pytagi.nn.data_struct.BaseHiddenStates


      Get the output hidden states


   .. py:property:: input_delta_z_buffer
      :type: pytagi.nn.data_struct.BaseDeltaStates


      Get the delta hidden states


   .. py:property:: output_delta_z_buffer
      :type: pytagi.nn.data_struct.BaseDeltaStates


      Get the delta hidden states


   .. py:property:: z_buffer_size
      :type: int


      Get the z_buffer_size.


   .. py:property:: z_buffer_block_size
      :type: int


      Get the z_buffer_block_size.


   .. py:property:: device
      :type: str


      Get the device


   .. py:property:: input_state_update
      :type: bool


      Get the device


   .. py:property:: num_samples
      :type: int


      Get the num_samples.


   .. py:method:: to_device(device: str)

      Move the model to a specific device.



   .. py:method:: params_to_device()

      Move the model parameters to a specific cuda device.



   .. py:method:: params_to_host()

      Move the model parameters from cuda device to the host.



   .. py:method:: set_threads(num_threads: int)

      Set the number of threads to use.



   .. py:method:: train()

      Set the number of threads to use.



   .. py:method:: eval()

      Set the number of threads to use.



   .. py:method:: forward(mu_x: numpy.ndarray, var_x: numpy.ndarray = None) -> Tuple[numpy.ndarray, numpy.ndarray]

      Perform a forward pass.



   .. py:method:: backward()

      Perform a backward pass.



   .. py:method:: smoother()

      Perform a smoother pass.



   .. py:method:: step()

      Perform a step of inference.



   .. py:method:: reset_lstm_states()

      Reset lstm states



   .. py:method:: output_to_host() -> List[float]

      Copy the output data to the host.



   .. py:method:: delta_z_to_host() -> List[float]

      Copy the delta Z data to the host.



   .. py:method:: set_delta_z(delta_mu: numpy.ndarray, delta_var: numpy.ndarray)

      Send the delta Z to device



   .. py:method:: get_layer_stack_info() -> str

      Get information about the layer stack.



   .. py:method:: preinit_layer()

      Preinitialize the layer.



   .. py:method:: get_neg_var_w_counter() -> dict

      Get the number of negative variance weights.



   .. py:method:: save(filename: str)

      Save the model to a file.



   .. py:method:: load(filename: str)

      Load the model from a file.



   .. py:method:: save_csv(filename: str)

      Save the model parameters to a CSV file.



   .. py:method:: load_csv(filename: str)

      Load the model parameters from a CSV file.



   .. py:method:: parameters() -> List[numpy.ndarray]

      Get the model parameters. Stored tuple (mu_w, var_w, mu_b, var_b) in a list



   .. py:method:: load_state_dict(state_dict: dict)

      Load the model parameters from a state dict.



   .. py:method:: state_dict() -> dict

      Get the model parameters as a state dict where key is the layer name
      and value is a tuple of 4 arrays (mu_w, var_w, mu_b, var_b)



   .. py:method:: params_from(other: Sequential)

      Copy parameters from another model.



   .. py:method:: get_outputs() -> Tuple[numpy.ndarray, numpy.ndarray]


   .. py:method:: get_outputs_smoother() -> Tuple[numpy.ndarray, numpy.ndarray]


   .. py:method:: get_input_states() -> Tuple[numpy.ndarray, numpy.ndarray]

      Get the input states.



   .. py:method:: get_norm_mean_var() -> dict

      Get the mean and variance of the normalization layer.
      :returns: A dictionary containing the mean and variance of the normalization layer.
                each key is the layer name and the value is a tuple of 4 arrays:
                mu_batch: mean of the batch
                var_batch: variance of the batch
                mu_ema_batch: mean of the exponential moving average (ema) of the batch
                var_ema_batch: variance of the ema of the batch



   .. py:method:: get_lstm_states() -> dict

      Get the LSTM states for all LSTM layers as a dictionary.

      :returns:

                A dictionary where each key is the layer index (int) and each value is a 4-tuple
                    of numpy arrays (mu_h_prior, var_h_prior, mu_c_prior, var_c_prior).
      :rtype: dict



   .. py:method:: set_lstm_states(states: dict) -> None

      Set the LSTM states for all LSTM layers using a dictionary.

      :param states: A dictionary mapping layer indices (int) to a 4-tuple of numpy arrays:
                     (mu_h_prior, var_h_prior, mu_c_prior, var_c_prior).
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
