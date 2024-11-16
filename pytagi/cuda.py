import cutagi


def is_available() -> bool:
    """Check if CUDA is available

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    return cutagi.is_cuda_available()
