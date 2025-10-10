from typing import List, Tuple

import cutagi
import numpy as np

from pytagi.nn.data_struct import BaseDeltaStates, BaseHiddenStates
from pytagi.nn.sequential import Sequential


class DDPConfig:
    """Configuration for Distributed Data Parallel (DDP) training.

    This class holds all the necessary settings for initializing a distributed
    process group.
    """

    def __init__(
        self,
        device_ids: List[int],
        backend: str = "nccl",
        rank: int = 0,
        world_size: int = 1,
    ):
        """Initializes the DDP configuration.

        :param device_ids: A list of GPU device IDs to be used for training.
        :type device_ids: List[int]
        :param backend: The distributed backend to use. 'nccl' is recommended for GPUs.
                        Defaults to "nccl".
        :type backend: str, optional
        :param rank: The unique rank of the current process. Defaults to 0.
        :type rank: int, optional
        :param world_size: The total number of processes participating in the training.
                           Defaults to 1.
        :type world_size: int, optional
        """
        self._cpp_backend = cutagi.DDPConfig(
            device_ids, backend, rank, world_size
        )

    @property
    def device_ids(self) -> List[int]:
        """The list of GPU device IDs."""
        return self._cpp_backend.device_ids

    @device_ids.setter
    def device_ids(self, value: List[int]):
        """Sets the list of GPU device IDs.

        :param value: The new list of device IDs.
        :type value: List[int]
        """
        self._cpp_backend.device_ids = value

    @property
    def backend(self) -> str:
        """The distributed communication backend (e.g., 'nccl')."""
        return self._cpp_backend.backend

    @backend.setter
    def backend(self, value: str):
        """Sets the distributed backend.

        :param value: The new backend string.
        :type value: str
        """
        self._cpp_backend.backend = value

    @property
    def rank(self) -> int:
        """The rank of the current process in the distributed group."""
        return self._cpp_backend.rank

    @rank.setter
    def rank(self, value: int):
        """Sets the process rank.

        :param value: The new rank.
        :type value: int
        """
        self._cpp_backend.rank = value

    @property
    def world_size(self) -> int:
        """The total number of processes in the distributed group."""
        return self._cpp_backend.world_size

    @world_size.setter
    def world_size(self, value: int):
        """Sets the world size.

        :param value: The new world size.
        :type value: int
        """
        self._cpp_backend.world_size = value


class DDPSequential:
    """A wrapper for `Sequential` models to enable Distributed Data Parallel (DDP) training.

    This class handles gradient synchronization and parameter updates across multiple
    processes, allowing for scalable training on multiple GPUs.
    """

    def __init__(
        self,
        model: Sequential,
        config: DDPConfig,
        average: bool = True,
    ):
        """Initializes the DDPSequential wrapper.

        :param model: The `Sequential` model to be parallelized.
        :type model: Sequential
        :param config: The DDP configuration object.
        :type config: DDPConfig
        :param average: If True, gradients are averaged across processes. If False, they are summed.
                        Defaults to True.
        :type average: bool, optional
        """
        self._cpp_backend = cutagi.DDPSequential(
            model._cpp_backend, config._cpp_backend, average
        )
        self.model = model
        self.config = config

    @property
    def output_z_buffer(self) -> BaseHiddenStates:
        """The output hidden states buffer from the forward pass of the underlying model."""
        return self.model.output_z_buffer

    @output_z_buffer.setter
    def output_z_buffer(self, value: BaseHiddenStates):
        """Sets the output hidden states buffer on the underlying model.

        :param value: The new output hidden states buffer.
        :type value: BaseHiddenStates
        """
        self.model.output_z_buffer = value

    @property
    def input_delta_z_buffer(self) -> BaseDeltaStates:
        """The input delta states buffer for the backward pass of the underlying model."""
        return self.model.input_delta_z_buffer

    @input_delta_z_buffer.setter
    def input_delta_z_buffer(self, value: BaseDeltaStates):
        """Sets the input delta states buffer on the underlying model.

        :param value: The new input delta states buffer.
        :type value: BaseDeltaStates
        """
        self.model.input_delta_z_buffer = value

    def __call__(
        self, mu_x: np.ndarray, var_x: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """A convenient alias for the forward pass.

        :param mu_x: The mean of the input data for the current process.
        :type mu_x: np.ndarray
        :param var_x: The variance of the input data for the current process. Defaults to None.
        :type var_x: np.ndarray, optional
        :return: A tuple containing the mean and variance of the model's output.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        return self.forward(mu_x, var_x)

    def forward(
        self, mu_x: np.ndarray, var_x: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs a forward pass on the local model replica.

        :param mu_x: The mean of the input data.
        :type mu_x: np.ndarray
        :param var_x: The variance of the input data. Defaults to None.
        :type var_x: np.ndarray, optional
        :return: A tuple containing the mean and variance of the output.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        self._cpp_backend.forward(mu_x, var_x)
        return self._cpp_backend.get_outputs()

    def backward(self):
        """Performs a backward pass and synchronizes gradients across all processes."""
        self._cpp_backend.backward()

    def step(self):
        """Performs a single parameter update step based on the synchronized gradients."""
        self._cpp_backend.step()

    def train(self):
        """Sets the model to training mode."""
        self._cpp_backend.train()

    def eval(self):
        """Sets the model to evaluation mode."""
        self._cpp_backend.eval()

    def barrier(self):
        """Synchronizes all processes.

        Blocks until all processes in the distributed group have reached this point.
        """
        self._cpp_backend.barrier()

    def get_outputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the outputs from the last forward pass on the local replica.

        :return: A tuple containing the mean and variance of the output.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        return self._cpp_backend.get_outputs()

    def output_to_host(self):
        """Copies the output data from the device to the host (CPU memory)."""
        self._cpp_backend.output_to_host()

    def get_device_with_index(self) -> str:
        """Gets the device string for the current process, including its index.

        :return: The device string, e.g., 'cuda:0'.
        :rtype: str
        """
        return self._cpp_backend.get_device_with_index()
