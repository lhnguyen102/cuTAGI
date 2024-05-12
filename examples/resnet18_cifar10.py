# Temporary import. It will be removed in the final version
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
import fire
import numpy as np
from tqdm import tqdm

from examples.data_loader import MnistDataLoader
from pytagi import HRCSoftmaxMetric
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    OutputUpdater,
    ReLU,
    Sequential,
    LayerBlock,
    ResNetBlock,
)
from pytagi import Utils
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader


def make_layer_block(
    in_channels: int, out_channels: int, stride: int = 1, padding_type: int = 1
):
    """Create a layer block for resnet 18"""

    return LayerBlock(
        Conv2d(
            in_channels,
            out_channels,
            3,
            bias=False,
            stride=stride,
            padding=1,
            padding_type=padding_type,
        ),
        BatchNorm2d(out_channels),
        ReLU(),
        Conv2d(out_channels, out_channels, 3, bias=False, padding=1),
        BatchNorm2d(out_channels),
    )


def resnet18_cifar10() -> Sequential:
    """Resnet18 architecture for cifar10"""
    # Resnet Block
    # 32x32
    block_1 = make_layer_block(64, 64)
    resnet_block_1 = ResNetBlock(block_1)

    resnet_block_2 = ResNetBlock(make_layer_block(64, 64))

    # 16x16
    resnet_block_3 = ResNetBlock(
        make_layer_block(64, 128, 2, 2),
        LayerBlock(Conv2d(64, 128, 2, bias=False, stride=2), BatchNorm2d(128)),
    )
    resnet_block_4 = ResNetBlock(make_layer_block(128, 128))

    # 8x8
    resnet_block_5 = ResNetBlock(
        make_layer_block(128, 256, 2, 2),
        LayerBlock(Conv2d(128, 256, 2, bias=False, stride=2), BatchNorm2d(256)),
    )
    resnet_block_6 = ResNetBlock(make_layer_block(256, 256))

    # 4x4
    resnet_block_7 = ResNetBlock(
        make_layer_block(256, 512, 2, 2),
        LayerBlock(Conv2d(256, 512, 2, bias=False, stride=2), BatchNorm2d(512)),
    )
    resnet_block_8 = ResNetBlock(make_layer_block(512, 512))

    return Sequential(
        Conv2d(3, 64, 3, bias=False, padding=1, in_width=32, in_height=32),
        BatchNorm2d(64),
        ReLU(),
        resnet_block_1,
        ReLU(),
        resnet_block_2,
        ReLU(),
        resnet_block_3,
        ReLU(),
        resnet_block_4,
        ReLU(),
        resnet_block_5,
        ReLU(),
        resnet_block_6,
        ReLU(),
        resnet_block_7,
        ReLU(),
        resnet_block_8,
        ReLU(),
        AvgPool2d(4),
        Linear(512, 11),
    )


def custom_collate_fn(batch):
    # `batch` is a list of tuples (image, label)
    batch_images, batch_labels = zip(*batch)

    # Convert to a single tensor and then to numpy
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)

    # Flatten images and labels to 1D
    batch_images = batch_images.numpy().reshape(len(batch_images), -1)
    batch_labels = batch_labels.numpy().flatten()

    return batch_images, batch_labels


def main(num_epochs: int = 10, batch_size: int = 128, sigma_v: float = 1.0):
    """
    Run classification training on the Cifar dataset using a custom neural model.

    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    """
    utils = Utils()

    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data/cifar_torch", train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar_torch", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=100,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )

    # Hierachical Softmax
    metric = HRCSoftmaxMetric(num_classes=10)

    # Resnet18
    breakpoint()
    net = resnet18_cifar10()
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
