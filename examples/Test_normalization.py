import fire
import numpy as np
from tqdm import tqdm

from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    LayerNorm,
    Linear,
    MixtureReLU,
    Sequential,
)
nb_x = 10
nb_z = 128
FNN_1 = Sequential(
    Linear(nb_x, nb_z, gain_weight=1, gain_bias=0.05),
    #MixtureReLU(),
)

FNN = Sequential(
    Linear(784, 128),
    MixtureReLU(),
    Linear(128, 128),
    MixtureReLU(),
    Linear(128, 11),
)

FNN_BATCHNORM = Sequential(
    Linear(784, 100),
    MixtureReLU(),
    BatchNorm2d(100),
    Linear(100, 100),
    MixtureReLU(),
    BatchNorm2d(100),
    Linear(100, 11),
)

FNN_LAYERNORM = Sequential(
    Linear(784, 100, bias=False),
    MixtureReLU(),
    LayerNorm((100,)),
    Linear(100, 100, bias=False),
    MixtureReLU(),
    LayerNorm((100,)),
    Linear(100, 11),
)

CNN = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, gain_weight=1, gain_bias=0.05),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, gain_weight=1, gain_bias=0.05),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100, gain_weight=1, gain_bias=0.05),
    MixtureReLU(),
    Linear(100, 11, gain_weight=1, gain_bias=0.05),
)

CNN_BATCHNORM = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False),
    MixtureReLU(),
    BatchNorm2d(16),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False),
    MixtureReLU(),
    BatchNorm2d(32),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    MixtureReLU(),
    Linear(100, 11),
)

CNN_LAYERNORM = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False),
    MixtureReLU(),
    LayerNorm((16, 27, 27)),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False),
    MixtureReLU(),
    LayerNorm((32, 9, 9)),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    MixtureReLU(),
    Linear(100, 11),
)


def main(num_epochs: int = 10, batch_size: int = 1, sigma_v: float = 0.1):
    # Network configuration
    net = CNN
    net.to_device("cpu")
    #net.set_threads(16)
    m_x = np.float32(np.random.normal(loc = 0.0, scale = 1.0, size = (nb_x)).flatten())
    v_x = np.ones(nb_x, dtype=np.float32).flatten()

    # Prior predictive - Feedforward
    m_pred, v_pred = net(m_x)
    print("Prior predictive -> E[v_pred] = ", np.average(v_pred), "+-", np.std(v_pred))


if __name__ == "__main__":
    fire.Fire(main)
