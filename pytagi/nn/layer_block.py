import cutagi
from typing import List

from pytagi.nn.base_layer import BaseLayer


class LayerBlock(BaseLayer):
    """A stack of different layer derived from BaseLayer"""

    def __init__(self, *layers: BaseLayer):
        """
        Initialize the Sequential model with the given layers.
        Args:
            layers: A variable number of layers (instances of BaseLayer or derived classes).
        """
        backend_layers = [layer._cpp_backend for layer in layers]
        self._cpp_backend = cutagi.LayerBlock(backend_layers)

    def switch_to_cuda(self):
        """Convert all layers to cuda layer"""
        self._cpp_backend.switch_to_cuda()

    @property
    def layers(self) -> None:
        """Get layers"""
        return self._cpp_backend.layers

    @layers.setter
    def layers(self, value: List[BaseLayer]):
        """Set base layers"""
        self._cpp_backend.layers = value
