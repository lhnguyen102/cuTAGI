pytagi.nn.ddp
=============

.. py:module:: pytagi.nn.ddp


Classes
-------

.. autoapisummary::

   pytagi.nn.ddp.DDPConfig
   pytagi.nn.ddp.DDPSequential


Module Contents
---------------

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



