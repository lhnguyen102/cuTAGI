pytagi.nn.resnet_block
======================

.. py:module:: pytagi.nn.resnet_block


Classes
-------

.. autoapisummary::

   pytagi.nn.resnet_block.ResNetBlock


Module Contents
---------------

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
