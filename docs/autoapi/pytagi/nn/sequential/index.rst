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
