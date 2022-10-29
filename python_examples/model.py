###############################################################################
# File:         model.py
# Description:  Diffrent example how to build a model in pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      October 29, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from python_src.tagi_network import NetProp


class RegressionMLP(NetProp):
    """Multi-layer perceptron for regression task"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [1, 1, 1]
        self.nodes = [1, 50, 1]
        self.activations = [0, 4, 0]
        self.batch_size = 4
        self.sigma_v = 0.06
        self.device = "cpu"


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
        self.activations = [0, 4, 4, 0]
        self.batch_size = 10
        self.sigma_v = 1
        self.is_idx_ud = True
        self.multithreading = True
        self.device = "cpu"


class TimeSeriesLSTM(NetProp):
    """LSTM for time series forecasting"""

    def __init__(self,
                 input_seq_len: int,
                 output_seq_len: int,
                 seq_stride: int = 1,
                 *args,
                 **kwargs) -> None:
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
