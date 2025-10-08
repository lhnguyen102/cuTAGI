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


   A Residual Network (ResNet) block structure.

   This class implements the core structure of a ResNet block, consisting of a
   **main block** (which performs the main transformations) and an optional
   **shortcut** connection (which adds the input to the main block's output).
   It wraps the C++/CUDA backend `cutagi.ResNetBlock`.

   Initializes the ResNetBlock.

   :param main_block: The primary set of layers in the block (e.g., convolutional layers).
   :type main_block: Union[BaseLayer, LayerBlock]
   :param shortcut: The optional shortcut connection, often an identity mapping or a projection.
                    If None, an identity shortcut is implicitly assumed by the C++ backend.
   :type shortcut: Union[BaseLayer, LayerBlock], optional


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
