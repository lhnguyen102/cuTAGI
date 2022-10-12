###############################################################################
# File:         model.py
# Description:  Diffrent example how to build a model in pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      October 12, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
###############################################################################
from dataclasses import dataclass
from pytagi import Network


@dataclass
class RegressionMLP:
    """Multi-layer perceptron for regression task"""

    prop: Network = Network()

    def __post_init__(self) -> None:
        self.prop.layers = [1, 1, 1]
        self.prop.nodes = [1, 50, 1]
        self.prop.activations = [0, 4, 0]
        self.prop.batch_size = 4
        self.prop.sigma_v = 0.06