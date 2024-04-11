from typing import List, Optional

import cutagi


class BaseHiddenStates:
    def __init__(self, size: Optional[int] = None, block_size: Optional[int] = None):
        if size is not None and block_size is not None:
            self._cpp_backend = cutagi.BaseHiddenStates(size, block_size)
        else:
            self._cpp_backend = cutagi.BaseHiddenStates()

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

    def set_size(self, new_size: int, new_block_size: int) -> str:
        self._cpp_backend.set_size(new_size, new_block_size)


class BaseDeltaStates:
    def __init__(self, size: Optional[int] = None, block_size: Optional[int] = None):
        if size is not None and block_size is not None:
            self._cpp_backend = cutagi.BaseDeltaStates(size, block_size)
        else:
            self._cpp_backend = cutagi.BaseDeltaStates()

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

    def set_size(self, new_size: int, new_block_size: int) -> str:
        self._cpp_backend.set_size(new_size, new_block_size)


class HRCSoftmax:
    """Hierarchical softmax wrapper from the CPP backend. Further details can be
    found here https://building-babylon.net/2017/08/01/hierarchical-softmax

    Attributes:
        obs: A fictive observation \in [-1, 1]
        idx: Indices assigned to each label
        n_obs: Number of indices for each label
        len: Length of an observation e.g 10 labels -> len(obs) = 11
    """

    def __init__(self) -> None:
        self._cpp_backend = cutagi.HRCSoftmax()

    @property
    def obs(self) -> List[float]:
        return self._cpp_backend.obs

    @obs.setter
    def obs(self, value: List[float]):
        self._cpp_backend.obs = value

    @property
    def idx(self) -> List[int]:
        return self._cpp_backend.idx

    @idx.setter
    def idx(self, value: List[int]):
        self._cpp_backend.idx = value

    @property
    def num_obs(self) -> int:
        return self._cpp_backend.num_obs

    @num_obs.setter
    def num_obs(self, value: int):
        self._cpp_backend.num_obs = value

    @property
    def len(self) -> int:
        return self._cpp_backend.len

    @len.setter
    def len(self, value: int):
        self._cpp_backend.len = value
