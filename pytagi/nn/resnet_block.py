from typing import Union

import cutagi

from pytagi.nn.base_layer import BaseLayer
from pytagi.nn.data_struct import BaseDeltaStates, BaseHiddenStates
from pytagi.nn.layer_block import LayerBlock


class ResNetBlock(BaseLayer):
    """A Residual Network (ResNet) block structure.

    This class implements the core structure of a ResNet block, consisting of a
    **main block** (which performs the main transformations) and an optional
    **shortcut** connection (which adds the input to the main block's output).
    It wraps the C++/CUDA backend `cutagi.ResNetBlock`.
    """

    def __init__(
        self,
        main_block: Union[BaseLayer, LayerBlock],
        shortcut: Union[BaseLayer, LayerBlock] = None,
    ):
        """Initializes the ResNetBlock.

        :param main_block: The primary set of layers in the block (e.g., convolutional layers).
        :type main_block: Union[BaseLayer, LayerBlock]
        :param shortcut: The optional shortcut connection, often an identity mapping or a projection.
                         If None, an identity shortcut is implicitly assumed by the C++ backend.
        :type shortcut: Union[BaseLayer, LayerBlock], optional
        """
        if shortcut is not None:
            self._cpp_backend = cutagi.ResNetBlock(
                main_block._cpp_backend, shortcut._cpp_backend
            )
        else:
            self._cpp_backend = cutagi.ResNetBlock(main_block._cpp_backend)

    def init_shortcut_state(self) -> None:
        """Initializes the hidden state buffers for the shortcut layer."""
        self._cpp_backend.init_shortcut_state()

    def init_shortcut_delta_state(self) -> None:
        """Initializes the delta state buffers (error signals) for the shortcut layer."""
        self._cpp_backend.init_shortcut_delta_state()

    def init_input_buffer(self) -> None:
        """Initializes the input state buffer used to hold the input for both the main block and the shortcut."""
        self._cpp_backend.init_input_buffer()

    @property
    def main_block(self) -> LayerBlock:
        """Gets the **main block** component of the ResNet block."""
        return self._cpp_backend.main_block

    @main_block.setter
    def main_block(self, value: LayerBlock):
        """Sets the **main block** component of the ResNet block.

        :param value: The new main block instance.
        :type value: LayerBlock
        """
        self._cpp_backend.main_block = value

    @property
    def shortcut(self) -> BaseLayer:
        """Gets the **shortcut** component of the ResNet block."""
        return self._cpp_backend.shortcut

    @shortcut.setter
    def shortcut(self, value: BaseLayer):
        """Sets the **shortcut** component of the ResNet block.

        :param value: The new shortcut instance.
        :type value: BaseLayer
        """
        self._cpp_backend.shortcut = value

    @property
    def input_z(self) -> BaseHiddenStates:
        """Gets the buffered input hidden states (mean and variance) for the block."""
        return self._cpp_backend.input_z

    @input_z.setter
    def input_z(self, value: BaseHiddenStates):
        """Sets the buffered input hidden states (mean and variance) for the block.

        :param value: The input hidden states.
        :type value: BaseHiddenStates
        """
        self._cpp_backend.input_z = value

    @property
    def input_delta_z(self) -> BaseDeltaStates:
        """Gets the delta states (error signals) associated with the block's input."""
        return self._cpp_backend.input_delta_z

    @input_delta_z.setter
    def input_delta_z(self, value: BaseDeltaStates):
        """Sets the delta states (error signals) associated with the block's input.

        :param value: The input delta states.
        :type value: BaseDeltaStates
        """
        self._cpp_backend.input_delta_z = value

    @property
    def shortcut_output_z(self) -> BaseHiddenStates:
        """Gets the output hidden states (mean and variance) from the shortcut layer."""
        return self._cpp_backend.shortcut_output_z

    @shortcut_output_z.setter
    def shortcut_output_z(self, value: BaseHiddenStates):
        """Sets the output hidden states (mean and variance) for the shortcut layer.

        :param value: The output hidden states for the shortcut.
        :type value: BaseHiddenStates
        """
        self._cpp_backend.shortcut_output_z = value

    @property
    def shortcut_output_delta_z(self) -> BaseDeltaStates:
        """Gets the delta states (error signals) associated with the shortcut layer's output."""
        return self._cpp_backend.shortcut_output_delta_z

    @shortcut_output_delta_z.setter
    def shortcut_output_delta_z(self, value: BaseDeltaStates):
        """Sets the delta states (error signals) associated with the shortcut layer's output.

        :param value: The output delta states for the shortcut.
        :type value: BaseDeltaStates
        """
        self._cpp_backend.shortcut_output_delta_z = value
