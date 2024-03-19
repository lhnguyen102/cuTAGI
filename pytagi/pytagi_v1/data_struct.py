# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

from typing import List, Optional

import cutagitest


class BaseHiddenStates:
    def __init__(self, size: Optional[int] = None, block_size: Optional[int] = None):
        if size is not None and block_size is not None:
            self._cpp_backend = cutagitest.BaseHiddenStates(size, block_size)
        else:
            self._cpp_backend = cutagitest.BaseHiddenStates()

    @property
    def mu_a(self) -> List[float]:
        return self._cpp_backend.mu_a

    @mu_a.setter
    def mu_a(self, value: List[float]):
        self._cpp_backend.mu_a = value

    @property
    def var_a(self) -> List[float]:
        return self._cpp_backend.var_a

    @var_a.setter
    def var_a(self, value: List[float]):
        self._cpp_backend.var_a = value

    @property
    def jcb(self) -> List[float]:
        return self._cpp_backend.jcb

    @jcb.setter
    def jcb(self, value: List[float]):
        self._cpp_backend.jcb = value

    @property
    def size(self) -> int:
        return self._cpp_backend.size

    @property
    def block_size(self) -> int:
        return self._cpp_backend.block_size

    @property
    def actual_size(self) -> int:
        return self._cpp_backend.actual_size

    def set_input_x(self, mu_x: List[float], var_x: List[float], block_size: int):
        self._cpp_backend.set_input_x(mu_x, var_x, block_size)

    def get_name(self) -> str:
        return self._cpp_backend.get_name()


class BaseDeltaStates:
    def __init__(self, size: Optional[int] = None, block_size: Optional[int] = None):
        if size is not None and block_size is not None:
            self._cpp_backend = cutagitest.BaseDeltaStates(size, block_size)
        else:
            self._cpp_backend = cutagitest.BaseDeltaStates()

    @property
    def delta_mu(self) -> List[float]:
        return self._cpp_backend.delta_mu

    @delta_mu.setter
    def delta_mu(self, value: List[float]):
        self._cpp_backend.delta_mu = value

    @property
    def delta_var(self) -> List[float]:
        return self._cpp_backend.delta_var

    @delta_mu.setter
    def delta_var(self, value: List[float]):
        self._cpp_backend.delta_var = value

    @property
    def size(self) -> int:
        return self._cpp_backend.size

    @property
    def block_size(self) -> int:
        return self._cpp_backend.block_size

    @property
    def actual_size(self) -> int:
        return self._cpp_backend.actual_size

    def get_name(self) -> str:
        return self._cpp_backend.get_name()

    def reset_zeros(self) -> None:
        """Reset all delta_mu and delta_var to zeros"""
        self._cpp_backend.reset_zeros()

    def copy_from(self, source: "BaseDeltaStates", num_data: int = -1) -> None:
        """Copy values of delta_mu and delta_var from delta states"""
        self._cpp_backend.copy_from(source, num_data)
