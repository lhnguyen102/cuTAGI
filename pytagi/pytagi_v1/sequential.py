# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

from typing import List

import cutagitest
import numpy as np
from base_layer import BaseLayer


class Sequential:
    """Adding neural networks in a sequence mode"""

    def __init__(self, *layers: BaseLayer):
        """
        Initialize the Sequential model with the given layers.
        Args:
            layers: A variable number of layers (instances of BaseLayer or derived classes).
        """
        self._sequential = cutagitest.Sequential(list(layers))

    @property
    def z_buffer_size(self) -> int:
        """Get the z_buffer_size."""
        return self._sequential.z_buffer_size

    @z_buffer_size.setter
    def z_buffer_size(self, value: int):
        """Set the z_buffer_size."""
        self._sequential.z_buffer_size = value

    @property
    def z_buffer_block_size(self) -> int:
        """Get the z_buffer_block_size."""
        return self._sequential.z_buffer_block_size

    @z_buffer_block_size.setter
    def z_buffer_block_size(self, value: int):
        """Set the z_buffer_block_size."""
        self._sequential.z_buffer_block_size = value

    def to_device(self, device: str):
        """Move the model to a specific device."""
        self._sequential.to_device(device)

    def set_threads(self, num_threads: int):
        """Set the number of threads to use."""
        self._sequential.set_threads(num_threads)

    def forward(self, mu_x: np.ndarray, var_x: np.ndarray = None) -> List[float]:
        """Perform a forward pass."""
        return self._sequential.forward_py(mu_x, var_x)

    def backward(self):
        """Perform a backward pass."""
        self._sequential.backward()

    def step(self):
        """Perform a step of inference."""
        self._sequential.step()

    def output_to_host(self) -> List[float]:
        """Copy the output data to the host."""
        return self._sequential.output_to_host()

    def delta_z_to_host(self) -> List[float]:
        """Copy the delta Z data to the host."""
        return self._sequential.delta_z_to_host()

    def get_layer_stack_info(self) -> str:
        """Get information about the layer stack."""
        return self._sequential.get_layer_stack_info()

    def save(self, filename: str):
        """Save the model to a file."""
        self._sequential.save(filename)

    def load(self, filename: str):
        """Load the model from a file."""
        self._sequential.load(filename)

    def save_csv(self, filename: str):
        """Save the model parameters to a CSV file."""
        self._sequential.save_csv(filename)

    def load_csv(self, filename: str):
        """Load the model parameters from a CSV file."""
        self._sequential.load_csv(filename)

    def params_from(self, other: "Sequential"):
        """Copy parameters from another model."""
        self._sequential.params_from(other)
