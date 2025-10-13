from typing import List, Tuple

import cutagi
import numpy as np

from pytagi.nn.base_layer import BaseLayer
from pytagi.nn.data_struct import BaseDeltaStates, BaseHiddenStates


class Sequential:
    """A sequential container for layers.

    Layers are added to the container in the order they are passed in the
    constructor. This class acts as a Python wrapper for the C++/CUDA
    backend `cutagi.Sequential`.

    Example:
        >>> import pytagi.nn as nn
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 5)
        ... )
        >>> mu_in = np.random.randn(1, 10)
        >>> var_in = np.abs(np.random.randn(1, 10))
        >>> mu_out, var_out = model(mu_in, var_in)
    """

    def __init__(self, *layers: BaseLayer):
        """Initializes the Sequential model with a sequence of layers.

        :param layers: A variable number of layer instances (e.g., Linear, ReLU)
                       that will be executed in sequence.
        :type layers: BaseLayer
        """
        backend_layers = [layer._cpp_backend for layer in layers]
        self._cpp_backend = cutagi.Sequential(backend_layers)

    def __call__(
        self, mu_x: np.ndarray, var_x: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """An alias for the forward pass.

        :param mu_x: The mean of the input data.
        :type mu_x: np.ndarray
        :param var_x: The variance of the input data. Defaults to None.
        :type var_x: np.ndarray, optional
        :return: A tuple containing the mean and variance of the output.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        return self.forward(mu_x, var_x)

    @property
    def layers(self) -> List[BaseLayer]:
        """The list of layers in the model."""
        return self._cpp_backend.layers

    @property
    def output_z_buffer(self) -> BaseHiddenStates:
        """The output hidden states buffer from the forward pass."""
        return self._cpp_backend.output_z_buffer

    @output_z_buffer.setter
    def output_z_buffer(self, value: BaseHiddenStates):
        """Sets the output hidden states buffer.

        :param value: The new output hidden states buffer.
        :type value: BaseHiddenStates
        """
        self._cpp_backend.output_z_buffer = value

    @property
    def input_delta_z_buffer(self) -> BaseDeltaStates:
        """The input delta states buffer used in the backward pass."""
        return self._cpp_backend.input_delta_z_buffer

    @input_delta_z_buffer.setter
    def input_delta_z_buffer(self, value: BaseDeltaStates):
        """Sets the input delta states buffer.

        :param value: The new input delta states buffer.
        :type value: BaseDeltaStates
        """
        self._cpp_backend.input_delta_z_buffer = value

    @property
    def output_delta_z_buffer(self) -> BaseDeltaStates:
        """The output delta states buffer from the backward pass."""
        return self._cpp_backend.output_delta_z_buffer

    @output_delta_z_buffer.setter
    def output_delta_z_buffer(self, value: BaseDeltaStates):
        """Sets the output delta states buffer.

        :param value: The new output delta states buffer.
        :type value: BaseDeltaStates
        """
        self._cpp_backend.output_delta_z_buffer = value

    @property
    def z_buffer_size(self) -> int:
        """The size of the hidden state (`z`) buffer."""
        return self._cpp_backend.z_buffer_size

    @z_buffer_size.setter
    def z_buffer_size(self, value: int):
        """Sets the size of the hidden state (`z`) buffer.

        :param value: The new buffer size.
        :type value: int
        """
        self._cpp_backend.z_buffer_size = value

    @property
    def z_buffer_block_size(self) -> int:
        """The block size of the hidden state (`z`) buffer."""
        return self._cpp_backend.z_buffer_block_size

    @z_buffer_block_size.setter
    def z_buffer_block_size(self, value: int):
        """Sets the block size of the hidden state (`z`) buffer.

        :param value: The new buffer block size.
        :type value: int
        """
        self._cpp_backend.z_buffer_block_size = value

    @property
    def device(self) -> str:
        """The computational device ('cpu' or 'cuda') the model is on."""
        return self._cpp_backend.device

    @device.setter
    def device(self, value: str):
        """Sets the computational device.

        :param value: The device to set, e.g., 'cpu' or 'cuda:0'.
        :type value: str
        """
        self._cpp_backend.device = value

    @property
    def input_state_update(self) -> bool:
        """Flag indicating if the input state should be updated."""
        return self._cpp_backend.input_state_update

    @input_state_update.setter
    def input_state_update(self, value: bool):
        """Sets the flag for updating the input state.

        :param value: The new boolean value.
        :type value: bool
        """
        self._cpp_backend.input_state_update = value

    @property
    def num_samples(self) -> int:
        """The number of samples used for Monte Carlo estimation. This is used
        for debugging purposes"""
        return self._cpp_backend.num_samples

    @num_samples.setter
    def num_samples(self, value: int):
        """Sets the number of samples for Monte Carlo estimation. This is used
        for debugging purposes

        :param value: The number of samples.
        :type value: int
        """
        self._cpp_backend.num_samples = value

    def to_device(self, device: str):
        """Moves the model and its parameters to a specified device.

        :param device: The target device, e.g., 'cpu' or 'cuda:0'.
        :type device: str
        """
        self._cpp_backend.to_device(device)

    def params_to_device(self):
        """Moves the model parameters to the currently configured CUDA device."""
        self._cpp_backend.params_to_device()

    def params_to_host(self):
        """Moves the model parameters from the CUDA device to the host (CPU)."""
        self._cpp_backend.params_to_host()

    def set_threads(self, num_threads: int):
        """Sets the number of CPU threads to use for computation.

        :param num_threads: The number of threads.
        :type num_threads: int
        """
        self._cpp_backend.set_threads(num_threads)

    def train(self):
        """Sets the model to training mode."""
        self._cpp_backend.train()

    def eval(self):
        """Sets the model to evaluation mode."""
        self._cpp_backend.eval()

    def forward(
        self, mu_x: np.ndarray, var_x: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs a forward pass through the network.

        :param mu_x: The mean of the input data.
        :type mu_x: np.ndarray
        :param var_x: The variance of the input data. Defaults to None.
        :type var_x: np.ndarray, optional
        :return: A tuple containing the mean and variance of the output.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        self._cpp_backend.forward(mu_x, var_x)
        return self.get_outputs()

    def backward(self):
        """Performs a backward pass to update the network parameters."""
        self._cpp_backend.backward()

    def smoother(self) -> Tuple[np.ndarray, np.ndarray]:
        """Performs a smoother pass (e.g., Rauch-Tung-Striebel smoother).

        This is used with the SLSTM to refine estimates by running backwards
        through time.

        :return: A tuple containing the mean and variance of the smoothed output.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        self._cpp_backend.smoother()
        return self.get_outputs_smoother()

    def step(self):
        """Performs a single step of inference to update the parameters."""
        self._cpp_backend.step()

    def reset_lstm_states(self):
        """Resets the hidden and cell states of all LSTM layers in the model."""
        self._cpp_backend.reset_lstm_states()

    def output_to_host(self) -> List[float]:
        """Copies the raw output data from the device to the host.

        :return: A list of floating-point values representing the flattened output.
        :rtype: List[float]
        """
        return self._cpp_backend.output_to_host()

    def delta_z_to_host(self) -> List[float]:
        """Copies the raw delta Z (error signal) data from the device to the host.

        :return: A list of floating-point values representing the flattened delta Z.
        :rtype: List[float]
        """
        return self._cpp_backend.delta_z_to_host()

    def set_delta_z(self, delta_mu: np.ndarray, delta_var: np.ndarray):
        """Sets the delta Z (error signal) on the device for the backward pass.

        :param delta_mu: The mean of the error signal.
        :type delta_mu: np.ndarray
        :param delta_var: The variance of the error signal.
        :type delta_var: np.ndarray
        """
        self._cpp_backend.set_delta_z(delta_mu, delta_var)

    def get_layer_stack_info(self) -> str:
        """Gets a string representation of the layer stack architecture.

        :return: A descriptive string of the model's layers.
        :rtype: str
        """
        return self._cpp_backend.get_layer_stack_info()

    def preinit_layer(self):
        """Pre-initializes the layers in the model."""
        self._cpp_backend.preinit_layer()

    def get_neg_var_w_counter(self) -> dict:
        """Counts the number of negative variance weights in each layer.

        :return: A dictionary where keys are layer names and values are the counts
                 of negative variances.
        :rtype: dict
        """
        return self._cpp_backend.get_neg_var_w_counter()

    def save(self, filename: str):
        """Saves the model's state to a binary file.

        :param filename: The path to the file where the model will be saved.
        :type filename: str
        """
        self._cpp_backend.save(filename)

    def load(self, filename: str):
        """Loads the model's state from a binary file.

        :param filename: The path to the file from which to load the model.
        :type filename: str
        """
        self._cpp_backend.load(filename)

    def save_csv(self, filename: str):
        """Saves the model parameters to a CSV file.

        :param filename: The base path for the CSV file(s).
        :type filename: str
        """
        self._cpp_backend.save_csv(filename)

    def load_csv(self, filename: str):
        """Loads the model parameters from a CSV file.

        :param filename: The base path of the CSV file(s).
        :type filename: str
        """
        self._cpp_backend.load_csv(filename)

    def parameters(
        self,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Gets all model parameters.

        :return: A list where each element is a tuple containing the parameters
                 for a layer: (mu_w, var_w, mu_b, var_b).
        :rtype: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        """
        return self._cpp_backend.parameters()

    def load_state_dict(self, state_dict: dict):
        """Loads the model's parameters from a state dictionary.

        :param state_dict: A dictionary containing the model's state.
        :type state_dict: dict
        """
        self._cpp_backend.load_state_dict(state_dict)

    def state_dict(self) -> dict:
        """Gets the model's parameters as a state dictionary.

        :return: A dictionary where each key is the layer name and the value is a
                 tuple of parameters: (mu_w, var_w, mu_b, var_b).
        :rtype: dict
        """
        return self._cpp_backend.state_dict()

    def params_from(self, other: "Sequential"):
        """Copies parameters from another Sequential model.

        :param other: The source model from which to copy parameters.
        :type other: Sequential
        """
        self._cpp_backend.params_from(other)

    def get_outputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the outputs from the last forward pass.

        :return: A tuple containing the mean and variance of the output.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        return self._cpp_backend.get_outputs()

    def get_outputs_smoother(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the outputs from the last smoother pass.

        :return: A tuple containing the mean and variance of the smoothed output.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        return self._cpp_backend.get_outputs_smoother()

    def get_input_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the input states of the model.

        :return: A tuple containing the mean and variance of the input states.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        return self._cpp_backend.get_input_states()

    def get_norm_mean_var(self) -> dict:
        """Gets the mean and variance from normalization layers.

        :return: A dictionary where each key is a normalization layer name and
                 the value is a tuple of four arrays:
                 (mu_batch, var_batch, mu_ema_batch, var_ema_batch).
        :rtype: dict
        """
        return self._cpp_backend.get_norm_mean_var()

    def get_lstm_states(self, time_step: int = -1) -> dict:
        """Get the LSTM states for all LSTM layers as a dictionary.

        :param time_step: The time step at which to retrieve the smoothed SLSTM states.
                          If not provided or -1, retrieves the unsmoothed current LSTM states.
        :type time_step: int, optional
        :return: A dictionary mapping layer indices to a 4-tuple of
                       numpy arrays: (mu_h_prior, var_h_prior, mu_c_prior, var_c_prior).
        :rtype: dict
        """
        return self._cpp_backend.get_lstm_states(time_step)

    def set_lstm_states(self, states: dict) -> None:
        """Sets the states for all LSTM layers.

        :param states: A dictionary mapping layer indices to a 4-tuple of
                       numpy arrays: (mu_h_prior, var_h_prior, mu_c_prior, var_c_prior).
        :type states: dict
        """
        self._cpp_backend.set_lstm_states(states)
