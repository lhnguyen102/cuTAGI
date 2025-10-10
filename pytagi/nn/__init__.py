"""
Neural Network module for pyTAGI.

This module provides various neural network layers and components,
including activation functions, base layers, convolutional layers,
recurrent layers, and utility modules. These components are designed
to work with probabilistic data structures and leverage a C++ backend
for performance.
"""

from pytagi.nn.activation import (
    ClosedFormSoftmax,
    EvenExp,
    LeakyReLU,
    MixtureReLU,
    MixtureSigmoid,
    MixtureTanh,
    ReLU,
    Remax,
    Sigmoid,
    Softmax,
    Softplus,
    Tanh,
)
from pytagi.nn.base_layer import BaseLayer
from pytagi.nn.batch_norm import BatchNorm2d
from pytagi.nn.conv2d import Conv2d
from pytagi.nn.convtranspose2d import ConvTranspose2d
from pytagi.nn.data_struct import BaseDeltaStates, BaseHiddenStates, HRCSoftmax
from pytagi.nn.ddp import DDPConfig, DDPSequential
from pytagi.nn.embedding import Embedding
from pytagi.nn.layer_block import LayerBlock
from pytagi.nn.layer_norm import LayerNorm
from pytagi.nn.linear import Linear
from pytagi.nn.lstm import LSTM
from pytagi.nn.output_updater import OutputUpdater
from pytagi.nn.pooling import AvgPool2d, MaxPool2d
from pytagi.nn.resnet_block import ResNetBlock
from pytagi.nn.sequential import Sequential
from pytagi.nn.slinear import SLinear
from pytagi.nn.slstm import SLSTM
