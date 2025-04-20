# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
import fire
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytagi
from pytagi import HRCSoftmaxMetric
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    MaxPool2d,
    MixtureReLU,
    OutputUpdater,
    ReLU,
    Sequential,
    Softmax,
)

# Constants for dataset normalization
NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD = [0.2470, 0.2435, 0.2616]

# Define the CNN architecture
CNN = Sequential(
    # 32x32
    Conv2d(3, 32, 5, bias=False, padding=2, in_width=32, in_height=32),
    ReLU(),
    BatchNorm2d(32),
    MaxPool2d(2, 2),
    # 16x16
    Conv2d(32, 32, 5, bias=False, padding=2),
    ReLU(),
    BatchNorm2d(32),
    MaxPool2d(2, 2),
    # 8x8
    Conv2d(32, 64, 5, bias=False, padding=2),
    ReLU(),
    BatchNorm2d(64),
    MaxPool2d(2, 2),
    # 4x4
    Linear(64 * 4 * 4, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10),
    Softmax(),
)


def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    return F.one_hot(torch.tensor(labels), num_classes=num_classes).numpy()


def load_datasets(batch_size: int):
    """Load and transform CIFAR10 training and test datasets."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD
            ),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=True,
        download=True,
        transform=transform_train,
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )
    return train_loader, test_loader


def main(num_epochs: int = 10, batch_size: int = 128, sigma_v: float = 0.2):
    """
    Run classification training on the CIFAR10 dataset using PyTAGI.
    """

    # Load CIFAR10 dataset
    train_loader, test_loader = load_datasets(batch_size)

    # Initialize network
    net = CNN
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")
    out_updater = OutputUpdater(net.device)

    # Training loop
    var_y = np.full((batch_size * 10,), sigma_v**2, dtype=np.float32)

    for epoch in range(num_epochs):
        net.train()
        train_error = 0
        num_train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            # Prepare data
            x = data.numpy().flatten()  # Flatten the images
            y = one_hot_encode(target).flatten()  # Convert to one-hot encoding

            m_pred, v_pred = net(x)

            # Update output layers
            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
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

            pbar.set_postfix(
                {"train_error": f"{train_error/num_train_samples * 100:.2f}%"}
            )

        # Testing
        net.eval()
        test_error = 0
        num_test_samples = 0

        for data, target in test_loader:
            x = data.numpy().flatten()
            m_pred, v_pred = net(x)

            # Calculate test error
            pred = np.reshape(m_pred, (batch_size, 10))
            label = np.argmax(pred, axis=1)
            test_error += np.sum(label != target.numpy())
            num_test_samples += len(target)

        test_error_rate = (test_error / num_test_samples) * 100
        print(
            f"\nEpoch {epoch+1}/{num_epochs}: "
            f"Train Error: {train_error/num_train_samples:.2f}% | "
            f"Test Error: {test_error_rate:.2f}%"
        )


if __name__ == "__main__":
    main()
