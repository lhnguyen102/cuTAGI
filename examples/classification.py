import fire
import numpy as np
from tqdm import tqdm

from examples.data_loader import MnistDataLoader
from pytagi import HRCSoftmaxMetric
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    LayerNorm,
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
)

FNN = Sequential(
    Linear(784, 100),
    ReLU(),
    Linear(100, 100),
    ReLU(),
    Linear(100, 11),
)

FNN_BATCHNORM = Sequential(
    Linear(784, 100),
    ReLU(),
    BatchNorm2d(100),
    Linear(100, 100),
    ReLU(),
    BatchNorm2d(100),
    Linear(100, 11),
)

FNN_LAYERNORM = Sequential(
    Linear(784, 100, bias=False),
    ReLU(),
    LayerNorm((100,)),
    Linear(100, 100, bias=False),
    ReLU(),
    LayerNorm((100,)),
    Linear(100, 11),
)

CNN = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28),
    ReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5),
    ReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    ReLU(),
    Linear(100, 11),
)

CNN_BATCHNORM = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False),
    ReLU(),
    BatchNorm2d(16),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False),
    ReLU(),
    BatchNorm2d(32),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    ReLU(),
    Linear(100, 11),
)

CNN_LAYERNORM = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False),
    LayerNorm((16, 27, 27)),
    ReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False),
    LayerNorm((32, 9, 9)),
    ReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    ReLU(),
    Linear(100, 11),
)


def main(num_epochs: int = 10, batch_size: int = 256, sigma_v: float = 2.0):
    """
    Run classification training on the MNIST dataset using a custom neural model.

    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    """
    # Load dataset
    train_dtl = MnistDataLoader(
        x_file="data/mnist/train-images-idx3-ubyte",
        y_file="data/mnist/train-labels-idx1-ubyte",
        num_images=60000,
    )
    test_dtl = MnistDataLoader(
        x_file="data/mnist/t10k-images-idx3-ubyte",
        y_file="data/mnist/t10k-labels-idx1-ubyte",
        num_images=10000,
    )

    # Hierachical Softmax
    metric = HRCSoftmaxMetric(num_classes=10)

    # Network configuration
    net = CNN
    net.to_device("cuda")
    net.set_threads(16)
    out_updater = OutputUpdater(net.device)

    # Training
    error_rates = []
    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
    )
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size=batch_size)
        for x, y, y_idx, label in batch_iter:
            # Feedforward and backward pass
            m_pred, v_pred = net(x)

            # Update output layers based on targets
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )

            # Update parameters
            net.backward()
            net.step()

            # Training metric
            error_rate = metric.error_rate(m_pred, v_pred, label)
            error_rates.append(error_rate)

        # Averaged error
        avg_error_rate = sum(error_rates[-100:])

        # Testing
        test_error_rates = []
        test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
        for x, _, _, label in test_batch_iter:
            m_pred, v_pred = net(x)

            # Training metric
            error_rate = metric.error_rate(m_pred, v_pred, label)
            test_error_rates.append(error_rate)

        test_error_rate = sum(test_error_rates) / len(test_error_rates)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | training error: {avg_error_rate:.2f}% | test error: {test_error_rate * 100:.2f}%",
            refresh=True,
        )
    print("Training complete.")


if __name__ == "__main__":
    fire.Fire(main)
