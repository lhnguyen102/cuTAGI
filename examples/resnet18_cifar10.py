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
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
)
from examples.tagi_resnet_model import resnet18_cifar10

# Constants for dataset normalization
NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)

CNN_NET = Sequential(
    Conv2d(3, 32, 5, bias=False, padding=2, in_width=32, in_height=32),
    BatchNorm2d(32),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    BatchNorm2d(32),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    Conv2d(32, 64, 5, bias=False, padding=2),
    BatchNorm2d(64),
    ReLU(),
    AvgPool2d(3, 2, padding=1, padding_type=2),
    Linear(64 * 4 * 4, 256),
    ReLU(),
    Linear(256, 11),
)


def custom_collate_fn(batch):
    # batch is a list of tuples (image, label)
    batch_images, batch_labels = zip(*batch)

    # Convert to a single tensor
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)

    # Flatten images to shape (B*C*H*W,)
    batch_images = batch_images.reshape(-1)

    # Convert to numpy arrays
    batch_images = batch_images.numpy()
    batch_labels = batch_labels.numpy()

    return batch_images, batch_labels


def load_datasets(batch_size: int):
    """Load and transform CIFAR10 training and test datasets."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    return train_loader, test_loader


def main(num_epochs: int = 100, batch_size: int = 128, sigma_v: float = 1):
    """
    Run classification training on the Cifar dataset using a custom neural model.

    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    """
    utils = Utils()
    train_loader, test_loader = load_datasets(batch_size)

    # Hierachical Softmax
    metric = HRCSoftmaxMetric(num_classes=10)

    # Resnet18
    # net = resnet18_cifar10()
    net = CNN_NET
    net.to_device("cuda")
    # net.set_threads(16)
    out_updater = OutputUpdater(net.device)

    # Training
    error_rates = []
    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
    )
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        if epoch > 1:
            sigma_v = exponential_scheduler(
                curr_v=sigma_v, min_v=0.3, decaying_factor=0.95, curr_iter=epoch
            )
            var_y = np.full(
                (batch_size * metric.hrc_softmax.num_obs,),
                sigma_v**2,
                dtype=np.float32,
            )
        net.train()
        for x, labels in train_loader:
            # Feedforward and backward pass
            m_pred, v_pred = net(x)

            # Update output layers based on targets
            y, y_idx, _ = utils.label_to_obs(labels=labels, num_classes=10)
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
            error_rate = metric.error_rate(m_pred, v_pred, labels)
            error_rates.append(error_rate)

        # Averaged error
        avg_error_rate = sum(error_rates[-100:])

        # Testing
        test_error_rates = []
        net.eval()
        for x, labels in test_loader:
            m_pred, v_pred = net(x)

            # Training metric
            error_rate = metric.error_rate(m_pred, v_pred, labels)
            test_error_rates.append(error_rate)

        test_error_rate = sum(test_error_rates) / len(test_error_rates)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | training error: {avg_error_rate:.2f}% | test error: {test_error_rate * 100:.2f}%",
            refresh=True,
        )
    print("Training complete.")


if __name__ == "__main__":
    fire.Fire(main)
