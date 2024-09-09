from typing import List, Tuple

import cutagi
import numpy as np

from pytagi.nn.base_layer import BaseLayer
from pytagi.nn.data_struct import BaseDeltaStates, BaseHiddenStates


class Sequential:
    """Adding neural networks in a sequence mode"""

    def __init__(self, *layers: BaseLayer):
        """
        Initialize the Sequential model with the given layers.
        Args:
            layers: A variable number of layers (instances of BaseLayer or derived classes).
        """
        backend_layers = [layer._cpp_backend for layer in layers]
        self._cpp_backend = cutagi.Sequential(backend_layers)

    def __call__(
        self, mu_x: np.ndarray, var_x: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(mu_x, var_x)

    @property
    def layers(self) -> List[BaseLayer]:
        """Get the layers of the model."""
        return self._cpp_backend.layers

    @property
    def output_z_buffer(self) -> BaseHiddenStates:
        """Get the output hidden states"""
        return self._cpp_backend.output_z_buffer

    @output_z_buffer.setter
    def output_z_buffer(self, value: BaseHiddenStates):
        """Set the output hidden states."""
        self._cpp_backend.output_z_buffer = value

    @property
    def input_delta_z_buffer(self) -> BaseDeltaStates:
        """Get the delta hidden states"""
        return self._cpp_backend.input_delta_z_buffer

    @input_delta_z_buffer.setter
    def input_delta_z_buffer(self, value: BaseDeltaStates):
        """Set the delta hidden states."""
        self._cpp_backend.input_delta_z_buffer = value

    @property
    def output_delta_z_buffer(self) -> BaseDeltaStates:
        """Get the delta hidden states"""
        return self._cpp_backend.output_delta_z_buffer

    @output_delta_z_buffer.setter
    def output_delta_z_buffer(self, value: BaseDeltaStates):
        """Set the delta hidden states."""
        self._cpp_backend.output_delta_z_buffer = value

    @property
    def z_buffer_size(self) -> int:
        """Get the z_buffer_size."""
        return self._cpp_backend.z_buffer_size

    @z_buffer_size.setter
    def z_buffer_size(self, value: int):
        """Set the z_buffer_size."""
        self._cpp_backend.z_buffer_size = value

    @property
    def z_buffer_block_size(self) -> int:
        """Get the z_buffer_block_size."""
        return self._cpp_backend.z_buffer_block_size

    @z_buffer_block_size.setter
    def z_buffer_block_size(self, value: int):
        """Set the z_buffer_block_size."""
        self._cpp_backend.z_buffer_block_size = value

    @property
    def device(self) -> str:
        """Get the device"""
        return self._cpp_backend.device

    @device.setter
    def device(self, value: str):
        """Set the sevice"""
        self._cpp_backend.device = value

    @property
    def input_state_update(self) -> bool:
        """Get the device"""
        return self._cpp_backend.input_state_update

    @input_state_update.setter
    def input_state_update(self, value: bool):
        """Set the sevice"""
        self._cpp_backend.input_state_update = value

    @property
    def num_samples(self) -> int:
        """Get the num_samples."""
        return self._cpp_backend.num_samples

    @num_samples.setter
    def num_samples(self, value: int):
        """Set the num_samples."""
        self._cpp_backend.num_samples = value

    def to_device(self, device: str):
        """Move the model to a specific device."""
        self._cpp_backend.to_device(device)

    def params_to_device(self):
        """Move the model parameters to a specific cuda device."""
        self._cpp_backend.params_to_device()

    def params_to_host(self):
        """Move the model parameters from cuda device to the host."""
        self._cpp_backend.params_to_host()

    def set_threads(self, num_threads: int):
        """Set the number of threads to use."""
        self._cpp_backend.set_threads(num_threads)

    def train(self):
        """Set the number of threads to use."""
        self._cpp_backend.train()

    def eval(self):
        """Set the number of threads to use."""
        self._cpp_backend.eval()

    def forward(
        self, mu_x: np.ndarray, var_x: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a forward pass."""
        self._cpp_backend.forward(mu_x, var_x)

        return self.get_outputs()

    def backward(self):
        """Perform a backward pass."""
        self._cpp_backend.backward()

    def smoother(self):
        """Perform a smoother pass."""
        self._cpp_backend.smoother()

        return self.get_outputs_smoother()

    def step(self):
        """Perform a step of inference."""
        self._cpp_backend.step()

    def output_to_host(self) -> List[float]:
        """Copy the output data to the host."""
        return self._cpp_backend.output_to_host()

    def delta_z_to_host(self) -> List[float]:
        """Copy the delta Z data to the host."""
        return self._cpp_backend.delta_z_to_host()

    def get_layer_stack_info(self) -> str:
        """Get information about the layer stack."""
        return self._cpp_backend.get_layer_stack_info()

    def preinit_layer(self):
        """Preinitialize the layer."""
        self._cpp_backend.preinit_layer()

    def save(self, filename: str):
        """Save the model to a file."""
        self._cpp_backend.save(filename)

    def load(self, filename: str):
        """Load the model from a file."""
        self._cpp_backend.load(filename)

    def save_csv(self, filename: str):
        """Save the model parameters to a CSV file."""
        self._cpp_backend.save_csv(filename)

    def load_csv(self, filename: str):
        """Load the model parameters from a CSV file."""
        self._cpp_backend.load_csv(filename)

    def parameters(self) -> List[np.ndarray]:
        """Get the model parameters. Stored mu_w, var_w, mu_b, var_b in list of
        numpy arrays. Example: A model of 5 layers leads to a params size of
        5 * 4 = 20
        """
        return self._cpp_backend.parameters()

    def load_state_dict(self, state_dict: dict):
        """Load the model parameters from a state dict."""
        self._cpp_backend.load_state_dict(state_dict)

    def get_state_dict(self) -> dict:
        """Get the model parameters as a state dict."""
        return self._cpp_backend.get_state_dict()

    def params_from(self, other: "Sequential"):
        """Copy parameters from another model."""
        self._cpp_backend.params_from(other)

    def get_outputs(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._cpp_backend.get_outputs()

    def get_outputs_smoother(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._cpp_backend.get_outputs_smoother()


