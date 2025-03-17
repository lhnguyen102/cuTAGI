import cutagi

from pytagi.nn.base_layer import BaseLayer


class MixtureLReLU(BaseLayer):
    """Muxture leaky ReLU layer"""

    def __init__(
        self,
        input_size: int,
        slope: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.slope = slope

        self._cpp_backend = cutagi.MixtureLReLU(
            input_size, slope
        )

    def get_layer_info(self) -> str:
        return self._cpp_backend.get_layer_info()

    def get_layer_name(self) -> str:
        return self._cpp_backend.get_layer_name()
