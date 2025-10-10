from typing import List, Optional

import cutagi


class BaseHiddenStates:
    """
    Represents the base hidden states, acting as a Python wrapper for the C++ backend.
    This class manages the mean (mu_a), variance (var_a), and Jacobian (jcb) of hidden states.
    """

    def __init__(
        self, size: Optional[int] = None, block_size: Optional[int] = None
    ):
        """
        Initializes the BaseHiddenStates.

        Args:
            size (Optional[int]): The size of the hidden states.
            block_size (Optional[int]): The block size for the hidden states.
        """
        if size is not None and block_size is not None:
            self._cpp_backend = cutagi.BaseHiddenStates(size, block_size)
        else:
            self._cpp_backend = cutagi.BaseHiddenStates()

    @property
    def mu_a(self) -> List[float]:
        """
        Gets or sets the mean of the hidden states (mu_a).
        """
        return self._cpp_backend.mu_a

    @mu_a.setter
    def mu_a(self, value: List[float]):
        self._cpp_backend.mu_a = value

    @property
    def var_a(self) -> List[float]:
        """
        Gets or sets the variance of the hidden states (var_a).
        """
        return self._cpp_backend.var_a

    @var_a.setter
    def var_a(self, value: List[float]):
        self._cpp_backend.var_a = value

    @property
    def jcb(self) -> List[float]:
        """
        Gets or sets the Jacobian of the hidden states (jcb).
        """
        return self._cpp_backend.jcb

    @jcb.setter
    def jcb(self, value: List[float]):
        self._cpp_backend.jcb = value

    @property
    def size(self) -> int:
        """
        Gets the size of the hidden states.
        """
        return self._cpp_backend.size

    @property
    def block_size(self) -> int:
        """
        Gets the block size of the hidden states.
        """
        return self._cpp_backend.block_size

    @property
    def actual_size(self) -> int:
        """
        Gets the actual size of the hidden states.
        """
        return self._cpp_backend.actual_size

    def set_input_x(
        self, mu_x: List[float], var_x: List[float], block_size: int
    ):
        """
        Sets the input for the hidden states.

        Args:
            mu_x (List[float]): The mean of the input x.
            var_x (List[float]): The variance of the input x.
            block_size (int): The block size for the input.
        """
        self._cpp_backend.set_input_x(mu_x, var_x, block_size)

    def get_name(self) -> str:
        """
        Gets the name of the hidden states type.

        Returns:
            str: The name of the hidden states type.
        """
        return self._cpp_backend.get_name()

    def set_size(self, new_size: int, new_block_size: int) -> str:
        """
        Sets a new size and block size for the hidden states.

        Args:
            new_size (int): The new size.
            new_block_size (int): The new block size.

        Returns:
            str: A message indicating the success or failure of the operation.
        """
        self._cpp_backend.set_size(new_size, new_block_size)


class BaseDeltaStates:
    """
    Represents the base delta states, acting as a Python wrapper for the C++ backend.
    This class manages the change in mean (delta_mu) and change in variance (delta_var)
    induced by the update step.
    """

    def __init__(
        self, size: Optional[int] = None, block_size: Optional[int] = None
    ):
        """
        Initializes the BaseDeltaStates.

        Args:
            size (Optional[int]): The size of the delta states.
            block_size (Optional[int]): The block size for the delta states.
        """
        if size is not None and block_size is not None:
            self._cpp_backend = cutagi.BaseDeltaStates(size, block_size)
        else:
            self._cpp_backend = cutagi.BaseDeltaStates()

    @property
    def delta_mu(self) -> List[float]:
        """
        Gets or sets the change in mean of the delta states (delta_mu).
        """
        return self._cpp_backend.delta_mu

    @delta_mu.setter
    def delta_mu(self, value: List[float]):
        self._cpp_backend.delta_mu = value

    @property
    def delta_var(self) -> List[float]:
        """
        Gets or sets the change in variance of the delta states (delta_var).
        """
        return self._cpp_backend.delta_var

    @delta_var.setter
    def delta_var(self, value: List[float]):
        self._cpp_backend.delta_var = value

    @property
    def size(self) -> int:
        """
        Gets the size of the delta states.
        """
        return self._cpp_backend.size

    @property
    def block_size(self) -> int:
        """
        Gets the block size of the delta states.
        """
        return self._cpp_backend.block_size

    @property
    def actual_size(self) -> int:
        """
        Gets the actual size of the delta states.
        """
        return self._cpp_backend.actual_size

    def get_name(self) -> str:
        """
        Gets the name of the delta states type.

        Returns:
            str: The name of the delta states type.
        """
        return self._cpp_backend.get_name()

    def reset_zeros(self) -> None:
        """Reset all delta_mu and delta_var to zeros."""
        self._cpp_backend.reset_zeros()

    def copy_from(self, source: "BaseDeltaStates", num_data: int = -1) -> None:
        """
        Copy values of delta_mu and delta_var from another delta states object.

        Args:
            source (BaseDeltaStates): The source delta states object to copy from.
            num_data (int): The number of data points to copy. Defaults to -1 (all).
        """
        self._cpp_backend.copy_from(source, num_data)

    def set_size(self, new_size: int, new_block_size: int) -> str:
        """
        Sets a new size and block size for the delta states.

        Args:
            new_size (int): The new size.
            new_block_size (int): The new block size.

        Returns:
            str: A message indicating the success or failure of the operation.
        """
        self._cpp_backend.set_size(new_size, new_block_size)


class HRCSoftmax:
    """
    Hierarchical softmax wrapper from the CPP backend.
    """

    def __init__(self) -> None:
        """Initializes the HRCSoftmax object."""
        self._cpp_backend = cutagi.HRCSoftmax()

    @property
    def obs(self) -> List[float]:
        """
        Gets or sets the fictive observation \in [-1, 1].
        """
        return self._cpp_backend.obs

    @obs.setter
    def obs(self, value: List[float]):
        self._cpp_backend.obs = value

    @property
    def idx(self) -> List[int]:
        """
        Gets or sets the indices assigned to each label.
        """
        return self._cpp_backend.idx

    @idx.setter
    def idx(self, value: List[int]):
        self._cpp_backend.idx = value

    @property
    def num_obs(self) -> int:
        """
        Gets or sets the number of indices for each label.
        """
        return self._cpp_backend.num_obs

    @num_obs.setter
    def num_obs(self, value: int):
        self._cpp_backend.num_obs = value

    @property
    def len(self) -> int:
        """
        Gets or sets the length of an observation (e.g., 10 labels -> len(obs) = 11).
        """
        return self._cpp_backend.len

    @len.setter
    def len(self, value: int):
        self._cpp_backend.len = value
