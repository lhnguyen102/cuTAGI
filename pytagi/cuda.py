from typing import Tuple

import cutagi


def is_available() -> bool:
    """Check if CUDA is available

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    return cutagi.is_cuda_available()


def is_nccl_available() -> bool:
    """Check if NCCL is available

    Returns:
        bool: True if NCCL is available, False otherwise
    """
    return cutagi.is_nccl_available()


def get_device_count() -> int:
    """Get the number of CUDA devices

    Returns:
        int: Number of CUDA devices
    """
    return cutagi.get_cuda_device_count()


def get_current_device() -> int:
    """Get the current CUDA device

    Returns:
        int: Current CUDA device
    """
    return cutagi.get_cuda_current_device()


def set_device(device_index: int) -> bool:
    """Set the current CUDA device

    Args:
        device_index: Device index to set
    """
    return cutagi.set_cuda_device(device_index)


def is_device_available(device_index: int) -> bool:
    """Check if a specific CUDA device is available

    Args:
        device_index: Device index to check
    """
    return cutagi.is_cuda_device_available(device_index)


def get_device_properties(device_index: int) -> str:
    """Get the properties of a specific CUDA device

    Args:
        device_index: Device index to get properties of
    """
    return cutagi.get_cuda_device_properties(device_index)


def get_device_memory(device_index: int) -> Tuple[int, int]:
    """Get the memory of a specific CUDA device

    Args:
        device_index: Device index to get memory of
    """
    return cutagi.get_cuda_device_memory(device_index)
