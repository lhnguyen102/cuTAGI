import numpy as np
from activation import Relu
from linear import Linear
from sequential import Sequential


class Model:
    """Tagi network"""

    def __init__(self):
        self.network = Sequential(
            Linear(784, 100), Relu(), Linear(100, 100), Relu(), Linear(100, 11)
        )

    def forward(self, mu_x: np.ndarray, var_x: np.ndarray = None):
        self.network.forward(mu_x, var_x)


if __name__ == "__main__":
    model = Model()
