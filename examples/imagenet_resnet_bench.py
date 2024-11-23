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
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from examples.tagi_resnet_model import resnet18_imagenet
from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    MixtureReLU,
    OutputUpdater,
    ReLU,
    Sequential,
)

torch.manual_seed(42)


def load_datasets(batch_size: int, data_dir: str):
    """Load the ImageNet dataset."""
    # Data Transforms
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    train_dataset = datasets.ImageFolder(
        f"{data_dir}/train", transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_transforms)

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader


def torch_trainer(batch_size: int, num_epochs: int, device: str = "cuda"):
    """Train ResNet-18 on the ImageNet dataset."""
    # Hyperparameters
    learning_rate = 0.001
    data_dir = "data/imagenet/ILSVRC/Data/CLS-LOC"

    # Load ImageNet datasets
    train_loader, val_loader = load_datasets(batch_size, data_dir)

    # Initialize the model
    torch_device = torch.device(device)
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_loader.dataset.classes))
    model.to(torch_device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        model.train()
        error_rates = []

        # Training phase
        for _, (data, target) in enumerate(train_loader):
            data = data.to(torch_device)
            target = target.to(torch_device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            train_correct = pred.eq(target.view_as(pred)).sum().item()
            error_rates.append((1.0 - (train_correct / data.shape[0])))

        # Averaged error
        avg_error_rate = sum(error_rates[-100:])

        # Evaluation phase
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(torch_device)
                target = target.to(torch_device)
                output = model(data)

                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        val_error_rate = (1.0 - correct / len(val_loader.dataset)) * 100
        pbar.set_description(
            f"Epoch# {epoch + 1}/{num_epochs} | Training Error: {avg_error_rate:.2f}% | Validation Error: {val_error_rate:.2f}%\n",
            refresh=False,
        )

    print("Training complete!")


def main(
    framework: str = "torch",
    batch_size: int = 64,
    epochs: int = 50,
    device: str = "cuda",
    sigma_v: float = 0.1,
):
    if framework == "torch":
        torch_trainer(batch_size=batch_size, num_epochs=epochs, device=device)
    # elif framework == "tagi":
    #     tagi_trainer(
    #         batch_size=batch_size, num_epochs=epochs, device=device, sigma_v=sigma_v
    #     )
    else:
        raise RuntimeError(f"Invalid Framework: {framework}")


if __name__ == "__main__":
    fire.Fire(main)
