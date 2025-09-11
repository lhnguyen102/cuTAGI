# Temporary import. It will be removed in the final vserion
import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
import fire
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import pytagi
from pytagi.nn import (
    AGVI,
    CELU,
    AvgPool2d,
    BatchNorm2d,
    ClosedFormSoftmax,
    Conv2d,
    Exp,
    LayerNorm,
    Linear,
    MaxPool2d,
    MixtureReLU,
    MixtureTanh,
    OutputUpdater,
    ReLU,
    Remax,
    Sequential,
    Softmax,
    Softplus,
    SplitActivation,
)

FNN = Sequential(
    Linear(784, 128, gain_weight=1.0, gain_bias=1.0),
    ReLU(),
    Linear(128, 128, gain_weight=1.0, gain_bias=1.0),
    ReLU(),
    Linear(128, 10, gain_weight=1.0, gain_bias=1.0),
    # AGVI(Softplus(), overfit_mu=False),
    # SplitActivation(Exp(), Remax()),
    # Remax(),
    Softmax(),
)

FNN_BATCHNORM = Sequential(
    Linear(784, 100),
    MixtureReLU(),
    BatchNorm2d(100),
    Linear(100, 100),
    MixtureReLU(),
    BatchNorm2d(100),
    Linear(100, 10),
    Softmax(),
)

FNN_LAYERNORM = Sequential(
    Linear(784, 100, bias=False),
    MixtureReLU(),
    LayerNorm((100,)),
    Linear(100, 100, bias=False),
    MixtureReLU(),
    LayerNorm((100,)),
    Linear(100, 10),
    Softmax(),
)

CNN = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28),
    ReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5),
    ReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 128),
    ReLU(),
    Linear(128, 20),
    AGVI(Exp(), overfit_mu=True),
    # ClosedFormSoftmax(),
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
    Linear(100, 20),
    # AGVI(Softplus(), overfit_mu=False),
    # Remax(),
    # Softmax(),
    SplitActivation(Exp(), Softmax()),
)


def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    labels = labels.clone().detach()
    return F.one_hot(labels, num_classes=num_classes).numpy().flatten()


def main(num_epochs: int = 20, batch_size: int = 128, sigma_v: float = 0.0):
    """
    Run classification training on the MNIST dataset using PyTAGI.
    """

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (1.0,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # Initialize network
    net = CNN_BATCHNORM
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")

    # net.load(".examples/softmax_mnist_epoch20.bin")

    out_updater = OutputUpdater(net.device)

    # Training loop
    var_y = np.full((batch_size * 10,), sigma_v**2, dtype=np.float32)

    for epoch in range(num_epochs):
        net.train()
        train_error = 0
        num_train_samples = 0

        all_m_pred = []
        all_v_pred = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            # Prepare data
            x = data.numpy().flatten()  # Flatten the images
            y = one_hot_encode(target).flatten()  # Convert to one-hot encoding

            # y[y == 0] = -0.4
            # y[y == 1] = 16.5

            # Feedforward and backward pass
            m_pred, v_pred = net(x)
            v_pred = m_pred[1::2] + v_pred[::2]
            m_pred = m_pred[::2]
            print("m_pred: ", m_pred)
            print("v_pred: ", v_pred)
            all_m_pred.append(m_pred)
            all_v_pred.append(v_pred)

            # print("Probability statistics:")
            # print("E[mu]: ", np.mean(m_pred))
            # print("Std[mu]: ", np.std(m_pred))
            # print("E[var]: ", np.mean(v_pred))
            # print("Std[var]: ", np.std(v_pred))

            # Update output layers
            out_updater.update_heteros(
                output_states=net.output_z_buffer,
                mu_obs=y,
                # var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )

            # Update parameters
            net.backward()
            net.step()

            # Calculate error rate
            pred = np.reshape(m_pred, (batch_size, 10))
            label = np.argmax(pred, axis=1)
            train_error += np.sum(label != target.numpy())
            num_train_samples += len(target)

            # Update progress bar
            pbar.set_postfix(
                {"train_error": f"{train_error/num_train_samples:.2f}%"}
            )

        average_m_pred = np.mean(np.concatenate(all_m_pred))
        average_v_pred = np.mean(np.concatenate(all_v_pred))
        std_v_m_pred = np.std(np.concatenate(all_m_pred))
        std_v_pred = np.std(np.concatenate(all_v_pred))

        average_v_pred_positive_m_pred = np.mean(
            np.concatenate(all_v_pred)[np.concatenate(all_m_pred) > 0]
        )
        average_v_pred_negative_m_pred = np.mean(
            np.concatenate(all_v_pred)[np.concatenate(all_m_pred) < 0]
        )
        print("Average mu prediction: ", average_m_pred)
        print("Average var prediction: ", average_v_pred)
        print("Std mu prediction: ", std_v_m_pred)
        print("Std var prediction: ", std_v_pred)
        print(
            "Average var prediction (positive mu): ",
            average_v_pred_positive_m_pred,
        )
        print(
            "Average var prediction (negative mu): ",
            average_v_pred_negative_m_pred,
        )

        # Testing
        net.eval()
        test_error = 0
        num_test_samples = 0

        for data, target in test_loader:
            x = data.numpy().flatten()
            m_pred, v_pred = net(x)
            m_pred = m_pred[::2]

            # Calculate test error
            pred = np.reshape(m_pred, (batch_size, 10))
            label = np.argmax(pred, axis=1)
            test_error += np.sum(label != target.numpy())
            num_test_samples += len(target)

        test_error_rate = (test_error / num_test_samples) * 100
        print(
            f"\nEpoch {epoch+1}/{num_epochs}: "
            f"Train Error: {train_error/num_train_samples * 100:.2f}% | "
            f"Test Error: {test_error_rate:.2f}%"
        )

        # net.save(f".examples/softmax_mnist_epoch{epoch+1}.bin")


if __name__ == "__main__":
    main()
