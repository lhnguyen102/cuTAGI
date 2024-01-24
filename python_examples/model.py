###############################################################################
# File:         model.py
# Description:  Diffrent example how to build a model in pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      Marche 12, 2023
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################
from pytagi import NetProp


class RegressionMLP(NetProp):
    """Multi-layer perceptron for regression task"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [1, 1, 1]
        self.nodes = [1, 50, 1]
        self.activations = [0, 4, 0]
        self.batch_size = 4
        self.sigma_v = 0.06
        self.sigma_v_min: float = 0.06
        self.device = "cpu"


class HeterosMLP(NetProp):
    """Multi-layer preceptron for regression task where the
    output's noise varies overtime"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [1, 1, 1, 1]
        self.nodes: list = [1, 100, 100, 2]  # output layer = [mean, std]
        self.activations: list = [0, 4, 4, 0]
        self.batch_size: int = 10
        self.sigma_v: float = 0
        self.sigma_v_min: float = 0
        self.noise_type: str = "heteros"
        self.noise_gain: float = 1.0
        self.init_method: str = "He"
        self.device: str = "cpu"


class DervMLP(NetProp):
    """Multi-layer perceptron for computing the derivative of a
    regression task"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [1, 1, 1, 1]
        self.nodes: list = [1, 64, 64, 1]
        self.activations: list = [0, 1, 4, 0]
        self.batch_size: int = 10
        self.sigma_v: float = 0.3
        self.sigma_v_min: float = 0.1
        self.decay_factor_sigma_v: float = 0.99
        self.collect_derivative: bool = True
        self.init_method: str = "He"


class FullCovMLP(NetProp):
    """Multi-layer perceptron for performing full-covariance prediction and
    inference"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [1, 1, 1, 1]
        self.nodes: list = [1, 30, 30, 1]
        self.activations: list = [0, 4, 4, 0]
        self.batch_size: int = 10
        self.sigma_v: float = 0.5
        self.sigma_v_min: float = 0.065
        self.decay_factor_sigma_v: float = 0.95
        self.sigma_x: float = 0.3485
        self.is_full_cov: bool = True
        self.multithreading: bool = True
        self.device: str = "cpu"


class MnistMLP(NetProp):
    """Multi-layer perceptron for mnist classificaiton.

    NOTE: The number of hidden states for last layer is 11 because
    TAGI use the hierarchical softmax for the classification task.
    Further details can be found in
    https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = [1, 1, 1, 1]
        self.nodes = [784, 100, 100, 11]
        self.activations = [0, 7, 7, 12]
        self.batch_size = 100
        self.sigma_v = 1
        self.is_idx_ud = True
        self.multithreading = True
        self.device = "cpu"


class SoftmaxMnistMLP(NetProp):
    """Multi-layer perceptron for mnist classificaiton."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [1, 1, 1, 1]
        self.nodes = [784, 100, 100, 10]
        self.activations = [0, 4, 4, 11]
        self.batch_size = 10
        self.sigma_v = 2
        self.is_idx_ud = False
        self.multithreading = True
        self.device = "cpu"


class TimeSeriesLSTM(NetProp):
    """LSTM for time series forecasting"""

    def __init__(
        self,
        input_seq_len: int,
        output_seq_len: int,
        seq_stride: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layers: list = [1, 7, 7, 1]
        self.nodes: list = [1, 5, 5, 1]
        self.activations: list = [0, 0, 0, 0]
        self.batch_size: int = 10
        self.input_seq_len: int = input_seq_len
        self.output_seq_len: int = output_seq_len
        self.seq_stride: int = seq_stride
        self.sigma_v: float = 2
        self.sigma_v_min: float = 0.3
        self.decay_factor_sigma_v: float = 0.95
        self.multithreading: bool = False
        self.device: str = "cpu"


class MnistEncoder(NetProp):
    """Encoder network for Mnist example"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [2, 2, 6, 4, 2, 6, 4, 1, 1]
        self.nodes: list = [784, 0, 0, 0, 0, 0, 0, 100, 10]
        self.kernels: list = [3, 1, 3, 3, 1, 3, 1, 1, 1]
        self.strides: list = [1, 0, 2, 1, 0, 2, 0, 0, 0]
        self.widths: list = [28, 0, 0, 0, 0, 0, 0, 0, 0]
        self.heights: list = [28, 0, 0, 0, 0, 0, 0, 0, 0]
        self.filters: list = [1, 16, 16, 16, 32, 32, 32, 1, 1]
        self.pads: list = [1, 0, 1, 1, 0, 1, 0, 0, 0]
        self.pad_types: list = [1, 0, 2, 1, 0, 2, 0, 0, 0]
        self.activations: list = [0, 4, 0, 0, 4, 0, 0, 4, 0]
        self.batch_size: int = 10
        self.is_output_ud: bool = False
        self.init_method: str = "He"
        self.device: str = "cuda"


class MnistDecoder(NetProp):
    """Decoder network for Mnist example"""

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = [1, 1, 21, 21, 21]
        self.nodes: list = [10, 1568, 0, 0, 784]
        self.kernels: list = [1, 3, 3, 3, 1]
        self.strides: list = [0, 2, 2, 1, 0]
        self.widths: list = [0, 7, 0, 0, 0]
        self.heights: list = [0, 7, 0, 0, 0]
        self.filters: list = [1, 32, 32, 16, 1]
        self.pads: list = [0, 1, 1, 1, 0]
        self.pad_types: list = [0, 2, 2, 1, 0]
        self.activations: list = [0, 4, 4, 4, 0]
        self.batch_size: int = 10
        self.sigma_v: float = 8
        self.sigma_v_min: float = 2
        self.is_idx_ud: bool = False
        self.last_backward_layer: int = 0
        self.decay_factor_sigma_v: float = 0.95
        self.init_method: str = "He"
        self.device: str = "cuda"
