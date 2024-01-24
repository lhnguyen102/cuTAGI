import numpy as np


class TagiNN:
    """Tagi's neural network template"""

    def __init__(self):
        self.net = None

    def forward(self, input):
        return self.net.forward(input)

    def backward(self, grad_output):
        return self.net.backward(grad_output)

    def __call__(self, input):
        return self.forward(input)
