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
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from examples.tagi_resnet_model import resnet18_imagenet
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import OutputUpdater
import pytagi


torch.manual_seed(42)
pytagi.manual_seed(42)


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


def load_datasets(batch_size: int, framework: str = "torch"):
    """Load the ImageNet dataset."""
    # Data Transforms
    data_dir = "data/imagenet/ILSVRC/Data/CLS-LOC"
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )

    # Load datasets
    train_set = datasets.ImageFolder(f"{data_dir}/train", transform=train_transforms)
    val_set = datasets.ImageFolder(f"{data_dir}/val", transform=val_transforms)

    if framework == "torch":
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=4
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )

    return train_loader, val_loader


def tagi_trainer(
    num_epochs: int,
    batch_size: int,
    device: str,
    sigma_v: float,
):
    """
    Run classification training on the Cifar dataset using a custom neural model.

    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    """
    utils = Utils()
    train_loader, val_loader = load_datasets(batch_size, "tagi")

    # Hierachical Softmax
    metric = HRCSoftmaxMetric(num_classes=1000)

    # Resnet18
    net = resnet18_imagenet(gain_w=0.15, gain_b=0.15)
    device = "cpu" if not pytagi.cuda.is_available() else device
    net.to_device(device)

    # Training
    out_updater = OutputUpdater(net.device)
    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
    )
    with tqdm(range(num_epochs), desc="Epoch Progress") as epoch_pbar:
        for epoch in epoch_pbar:
            error_rates = []
            net.train()
            with tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Batch Progress"
            ) as batch_pbar:
                for i, (x, labels) in enumerate(batch_pbar):
                    m_pred, v_pred = net(x)

                    # Update output layers based on targets
                    y, y_idx, _ = utils.label_to_obs(labels=labels, num_classes=1000)
                    out_updater.update_using_indices(
                        output_states=net.output_z_buffer,
                        mu_obs=y,
                        var_obs=var_y,
                        selected_idx=y_idx,
                        delta_states=net.input_delta_z_buffer,
                    )
                    net.backward()
                    net.step()

                    # Training metric
                    error_rate = metric.error_rate(m_pred, v_pred, labels)
                    error_rates.append(error_rate)

                    if i > 0 and i % 100 == 0:
                        avg_error_rate = sum(error_rates[-100:])
                        batch_pbar.set_description(
                            f"Training error: {avg_error_rate:.2f}%",
                            refresh=True,
                        )
                    if i > 0 and i % 1000 == 0:
                        break

            # Averaged training error
            avg_error_rate = sum(error_rates[-100:])

            # Testing
            test_error_rates = []
            net.eval()
            for x, labels in val_loader:
                m_pred, v_pred = net(x)

                # Check if any m_pred and v_pred is nan
                if np.isnan(m_pred).any() or np.isnan(v_pred).any():
                    print("m_pred or v_pred is nan")
                    break

                # Training metric
                error_rate = metric.error_rate(m_pred, v_pred, labels)

                # check if error rate is nan
                if np.isnan(error_rate):
                    print("Error rate is nan")
                    break
                test_error_rates.append(error_rate)

            test_error_rate = sum(test_error_rates) / len(test_error_rates)
            epoch_pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs} | training error: {avg_error_rate:.2f}% | test error: {test_error_rate * 100:.2f}%",
                refresh=True,
            )
    print("Training complete.")


def torch_trainer(batch_size: int, num_epochs: int, device: str = "cuda"):
    """Train ResNet-18 on the ImageNet dataset."""
    # torch.set_float32_matmul_precision("high")

    # Hyperparameters
    learning_rate = 0.001

    # Load ImageNet datasets
    train_loader, val_loader = load_datasets(batch_size, "torch")

    # Initialize the model
    torch_device = torch.device(device)
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_loader.dataset.classes))
    model.to(torch_device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    with tqdm(range(num_epochs), desc="Epoch Progress") as epoch_pbar:
        for epoch in epoch_pbar:
            model.train()
            error_rates = []

            with tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Batch Progress"
            ) as batch_pbar:
                for i, (data, target) in enumerate(batch_pbar):
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

                    if i > 0 and i % 100 == 0:
                        avg_error_rate = sum(error_rates[-100:])
                        batch_pbar.set_description(
                            f"Training error: {avg_error_rate:.2f}%",
                            refresh=True,
                        )

            # Averaged training error
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
            epoch_pbar.set_description(
                f"Epoch# {epoch + 1}/{num_epochs} | Training Error: {avg_error_rate:.2f}% | Validation Error: {val_error_rate:.2f}%",
                refresh=True,
            )

    print("Training complete!")


def main(
    framework: str = "tagi",
    batch_size: int = 64,
    epochs: int = 1,
    device: str = "cuda",
    sigma_v: float = 0.1,
):
    if framework == "torch":
        torch_trainer(batch_size=batch_size, num_epochs=epochs, device=device)
    elif framework == "tagi":
        tagi_trainer(
            batch_size=batch_size, num_epochs=epochs, device=device, sigma_v=sigma_v
        )
    else:
        raise RuntimeError(f"Invalid Framework: {framework}")


if __name__ == "__main__":
    fire.Fire(main)
