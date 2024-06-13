from typing import Union

import cutagi

from pytagi.nn.base_layer import BaseLayer
from pytagi.nn.data_struct import BaseDeltaStates, BaseHiddenStates
from pytagi.nn.layer_block import LayerBlock


class ResNetBlock(BaseLayer):
    """A residual architecture contains a main block and a shortcut layer"""

    def __init__(
        self,
        main_block: Union[BaseLayer, LayerBlock],
        shortcut: Union[BaseLayer, LayerBlock] = None,
    ):
        if shortcut is not None:
            self._cpp_backend = cutagi.ResNetBlock(
                main_block._cpp_backend, shortcut._cpp_backend
            )
        else:
            self._cpp_backend = cutagi.ResNetBlock(main_block._cpp_backend)

    def init_shortcut_state(self) -> None:
        """Initialize state buffer for shortcut"""
        self._cpp_backend.init_shortcut_state()

    def init_shortcut_delta_state(self) -> None:
        """Initialize update values for state buffer for the shortcut"""
        self._cpp_backend.init_shortcut_delta_state()

    def init_input_buffer(self) -> None:
        """Initialize input state buffer to hold temporary state"""
        self._cpp_backend.init_input_buffer()

    @property
    def main_block(self) -> LayerBlock:
        """Set main block"""
        return self._cpp_backend.main_block

    @main_block.setter
    def main_block(self, value: LayerBlock):
        """Set main block"""
        self._cpp_backend.main_block = value

    @property
    def shortcut(self) -> BaseLayer:
        """Set shortcut"""
        return self._cpp_backend.shortcut

    @shortcut.setter
    def shortcut(self, value: BaseLayer):
        """Set shortcut"""
        self._cpp_backend.shortcut = value

    @property
    def input_z(self) -> BaseHiddenStates:
        """Get output hidden states"""
        return self._cpp_backend.input_z

    @input_z.setter
    def input_z(self, value: BaseHiddenStates):
        """Set input hidden states."""
        self._cpp_backend.input_z = value

    @property
    def input_delta_z(self) -> BaseDeltaStates:
        """Get update values for input states"""
        return self._cpp_backend.input_delta_z

    @input_delta_z.setter
    def input_delta_z(self, value: BaseDeltaStates):
        """Set update values for input states"""
        self._cpp_backend.input_delta_z = value

    @property
    def shortcut_output_z(self) -> BaseHiddenStates:
        """Get output hidden states for shortcut"""
        return self._cpp_backend.shortcut_output_z

    @shortcut_output_z.setter
    def shortcut_output_z(self, value: BaseHiddenStates):
        """Set output hidden states for shortcut"""
        self._cpp_backend.shortcut_output_z = value

    @property
    def shortcut_output_delta_z(self) -> BaseDeltaStates:
        """Get update values for output hidden states for shortcut"""
        return self._cpp_backend.shortcut_output_delta_z

    @shortcut_output_delta_z.setter
    def shortcut_output_delta_z(self, value: BaseDeltaStates):
        """Set update values for output hidden states for shortcut"""
        self._cpp_backend.shortcut_output_delta_z = value
