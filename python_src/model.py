###############################################################################
# File:         model.py
# Description:  Diffrent example how to build a model in pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      October 19, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from pytagi import Network


class NetProp(Network):
    """Base class for network properties"""

    def __init__(self) -> None:
        super().__init__()


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
    """Multi-layer perceptron for mnist classificaiton"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [1, 1, 1, 1]
        self.nodes = [1, 100, 100]
        self.activations = [0, 4, 4, 0]
        self.batch_size = 10
        self.sigma_v = 1
        self.device = "cpu"
