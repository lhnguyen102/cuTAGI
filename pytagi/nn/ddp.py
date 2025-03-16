from typing import List, Tuple

import cutagi
import numpy as np

from pytagi.nn.sequential import Sequential


class DDPConfig:
    """Configuration for distributed training"""

    def __init__(
        self,
        device_ids: List[int],
        backend: str = "nccl",
        rank: int = 0,
        world_size: int = 1,
    ):
        self._cpp_backend = cutagi.DDPConfig(device_ids, backend, rank, world_size)

    @property
    def device_ids(self) -> List[int]:
        return self._cpp_backend.device_ids

    @device_ids.setter
    def device_ids(self, value: List[int]):
        self._cpp_backend.device_ids = value

    @property
    def backend(self) -> str:
        return self._cpp_backend.backend

    @backend.setter
    def backend(self, value: str):
        self._cpp_backend.backend = value

    @property
    def rank(self) -> int:
        return self._cpp_backend.rank

    @rank.setter
    def rank(self, value: int):
        self._cpp_backend.rank = value

    @property
    def world_size(self) -> int:
        return self._cpp_backend.world_size

    @world_size.setter
    def world_size(self, value: int):
        self._cpp_backend.world_size = value


class DDPSequential:
    """Distributed training wrapper for Sequential models"""

    def __init__(
        self,
        model: Sequential,
        config: DDPConfig,
        average: bool = True,
    ):
        self._cpp_backend = cutagi.DDPSequential(
            model._cpp_backend, config._cpp_backend, average
        )
        self.model = model
        self.config = config

    def __call__(
        self, mu_x: np.ndarray, var_x: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a forward pass"""
        return self.forward(mu_x, var_x)

    def forward(
        self, mu_x: np.ndarray, var_x: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a forward pass"""
        self._cpp_backend.forward(mu_x, var_x)
        return self._cpp_backend.get_outputs()

    def backward(self):
        """Perform a backward pass"""
        self._cpp_backend.backward()

    def step(self):
        """Perform a parameter update step"""
        self._cpp_backend.step()

    def train(self):
        """Set the model in training mode"""
        self._cpp_backend.train()

    def eval(self):
        """Set the model in evaluation mode"""
        self._cpp_backend.eval()

    def barrier(self):
        """Synchronize all processes"""
        self._cpp_backend.barrier()

    def get_outputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the outputs of the model"""
        return self._cpp_backend.get_outputs()
