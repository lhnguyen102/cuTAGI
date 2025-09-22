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
