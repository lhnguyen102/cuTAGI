import fire
import numpy as np
from tqdm import tqdm
import sys

from examples.data_loader import MnistDataLoader
from pytagi import HRCSoftmaxMetric
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    LayerNorm,
    Linear,
    OutputUpdater,
    MixtureReLU,
    ReLU,
    Sequential,
)
FNN_1 = Sequential(
    #Linear(784, 128, gain_weight=1, gain_bias=0.1),
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False, gain_weight=1, gain_bias=0.1),
    #ReLU(),
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
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, gain_weight=1, gain_bias=1),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, gain_weight=1, gain_bias=1),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100, gain_weight=1, gain_bias=1),
    MixtureReLU(),
    Linear(100, 11, gain_weight=1, gain_bias=1),
)

CNN_VSCALED = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, gain_weight=3.2, gain_bias=1),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, gain_weight=1.6, gain_bias=1),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100, gain_weight=1.5, gain_bias=1),
    MixtureReLU(),
    Linear(100, 11, gain_weight=1, gain_bias=1),
)

mu_scale = 1.00
var_scale = 0.20
CNN_SCALED = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, gain_weight=var_scale, gain_bias=mu_scale),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, gain_weight=var_scale, gain_bias=mu_scale),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100, gain_weight=var_scale, gain_bias=mu_scale),
    MixtureReLU(),
    Linear(100, 11, gain_weight=var_scale, gain_bias=mu_scale),
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
    Linear(32 * 4 * 4, 100, bias=False),
    MixtureReLU(),
    Linear(100, 11, bias=False),
)

CNN_BATCHNORM_SCALED = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False, gain_weight=var_scale, gain_bias=mu_scale),
    MixtureReLU(),
    #ReLU(),
    BatchNorm2d(16),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False, gain_weight=var_scale, gain_bias=mu_scale),
    MixtureReLU(),
    #ReLU(),
    BatchNorm2d(32),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100, bias=False, gain_weight=var_scale, gain_bias=mu_scale),
    MixtureReLU(),
    #ReLU(),
    Linear(100, 11, bias=False, gain_weight=var_scale, gain_bias=mu_scale),
)

CNN_LAYERNORM = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False, gain_weight=1, gain_bias=1),
    MixtureReLU(),
    LayerNorm((16, 27, 27)),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False, gain_weight=1, gain_bias=1),
    MixtureReLU(),
    LayerNorm((32, 9, 9)),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100, gain_weight=1, gain_bias=1),
    MixtureReLU(),
    Linear(100, 11, gain_weight=1, gain_bias=1),
)

CNN_LAYERNORM_SCALED = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False, gain_weight=var_scale, gain_bias=mu_scale),
    #LayerNorm((16, 27, 27)),
    MixtureReLU(),
    LayerNorm((16, 27, 27)),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False, gain_weight=var_scale, gain_bias=mu_scale),
    MixtureReLU(),
    LayerNorm((32, 9, 9)),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100, gain_weight=var_scale, gain_bias=mu_scale),
    MixtureReLU(),
    Linear(100, 11, gain_weight=var_scale, gain_bias=mu_scale),
)

CNN_LAYERNORM_INV = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False, gain_weight=3.2, gain_bias=1),
    LayerNorm((16, 27, 27)),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False, gain_weight=0.35, gain_bias=1),
    LayerNorm((32, 9, 9)),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100, gain_weight=1.25, gain_bias=1),
    MixtureReLU(),
    Linear(100, 11, gain_weight=1, gain_bias=1),
)

CNN_LAYERNORM_INV_SCALED = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False, gain_weight=3.2, gain_bias=3),
    LayerNorm((16, 27, 27)),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False, gain_weight=0.9, gain_bias=0.95),
    LayerNorm((32, 9, 9)),
    MixtureReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100, gain_weight=1.25, gain_bias=1.25),
    MixtureReLU(),
    Linear(100, 11, gain_weight=0.8, gain_bias=1.5),
)


def main(num_epochs: int = 10, batch_size: int = 128, sigma_v: float = 0.1):
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
    net = CNN_SCALED
    net.to_device("cuda")
    #net.set_threads(16)
    out_updater = OutputUpdater(net.device)

    # Training
    error_rates = []
    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
    )
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    print_var = True
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size=batch_size)
        for x, y, y_idx, label in batch_iter:
            # Feedforward and backward pass
            m_pred, v_pred = net(x)

            if print_var: # Print prior predictive variance
                print("Prior predictive -> E[v_pred] = ", np.average(v_pred), " | E[s_pred]", np.average(np.sqrt(v_pred)))
                print("                 -> V[m_pred] = ", np.var(m_pred), " | s[m_pred]", np.std(m_pred))
                print_var = False
            #exit()

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
            f"Epoch {epoch + 1}/{num_epochs} | training error: {avg_error_rate:.2f}% | test error: {test_error_rate * 100:.2f}%\n",
            refresh=False,
        )
    print("Training complete.")


if __name__ == "__main__":
    fire.Fire(main)
