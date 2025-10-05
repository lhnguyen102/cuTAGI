pytagi.nn.sequential
====================

.. py:module:: pytagi.nn.sequential


Classes
-------

.. autoapisummary::

   pytagi.nn.sequential.Sequential


Module Contents
---------------

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
