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
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    OutputUpdater,
    ReLU,
    MixtureReLU,
    Sequential,
)
from examples.tagi_resnet_model import resnet18_cifar10
from examples.torch_resnet_model import ResNet18

torch.manual_seed(17)

# Constants for dataset normalization
NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD = [0.2470, 0.2435, 0.2616]

TAGI_CNN_NET = Sequential(
    # 32x32
    Conv2d(3, 32, 5, bias=False, padding=2, in_width=32, in_height=32),
    BatchNorm2d(32),
    MixtureReLU(),
    AvgPool2d(2, 2),
    # 16x16
    Conv2d(32, 32, 5, bias=False, padding=2),
    BatchNorm2d(32),
    MixtureReLU(),
    AvgPool2d(2, 2),
    # 8x8
    Conv2d(32, 64, 5, bias=False, padding=2),
    BatchNorm2d(64),
    MixtureReLU(),
    AvgPool2d(2, 2),
    # 4x4
    Linear(64 * 4 * 4, 256),
    MixtureReLU(),
    Linear(256, 128),
    MixtureReLU(),
    Linear(128, 11),
)

TAGI_FNN = Sequential(
    Linear(32 * 32 * 3, 4096),
    ReLU(),
    Linear(4096, 4096),
    ReLU(),
    Linear(4096, 11),
)


# TORCH
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class TorchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 32x32
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 16x16
            nn.Conv2d(32, 32, kernel_size=5, bias=False, padding=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 8x8
            nn.Conv2d(32, 64, kernel_size=5, bias=False, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 4x4
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        self.model.apply(initialize_weights)

    def forward(self, x):
        # for i, layer in enumerate(self.model):
        #     x = layer(x)
        #     print(f"Layer {i}: {layer.__class__.__name__}, Output shape {x.shape}")
        # return x
        return self.model(x)


class TorchFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(32 * 32 * 3, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )
        self.model.apply(initialize_weights)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


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


def load_datasets(batch_size: int, framework: str = "tagi"):
    """Load and transform CIFAR10 training and test datasets."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar", train=False, download=True, transform=transform_test
    )

    if framework == "torch":
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=1
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=1
        )
    else:
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
    train_loader, test_loader = load_datasets(batch_size, "tagi")

    # Hierachical Softmax
    metric = HRCSoftmaxMetric(num_classes=10)

    # Resnet18
    # net = TAGI_CNN_NET
    net = resnet18_cifar10()
    net.to_device(device)
    # net.set_threads(10)
    out_updater = OutputUpdater(net.device)

    # Training

    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
    )
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        error_rates = []
        if epoch > 0:
            sigma_v = exponential_scheduler(
                curr_v=sigma_v, min_v=0.2, decaying_factor=0.99, curr_iter=epoch
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
            f"Epoch {epoch + 1}/{num_epochs} | training error: {avg_error_rate:.2f}% | test error: {test_error_rate * 100:.2f}\n%",
            refresh=False,
        )
    print("Training complete.")


def torch_trainer(batch_size: int, num_epochs: int, device: str = "cuda"):
    # Hyperparameters
    learning_rate = 0.001

    # torch.set_float32_matmul_precision("high")
    train_loader, test_loader = load_datasets(batch_size, "torch")

    # Initialize the model, loss function, and optimizer
    torch_device = torch.device(device)
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )
    model = ResNet18()
    # model = TorchCNN()
    # model = torch.compile(model)
    model.to(torch_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        model.train()
        error_rates = []
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

        # Evaluate the model on the test set
        model.eval()
        test_loss = 0
        correct = 0
        sample_count = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(torch_device)
                target = target.to(torch_device)
                output = model(data)

                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                sample_count += data.shape[0]

        test_loss /= len(test_loader.dataset)
        test_error_rate = (1.0 - correct / len(test_loader.dataset)) * 100
        pbar.set_description(
            f"Epoch# {epoch +1}/{num_epochs}| training error: {avg_error_rate:.2f}% | Test error: {test_error_rate: .2f}%\n",
            refresh=False,
        )


def main(
    framework: str = "tagi",
    batch_size: int = 128,
    epochs: int = 50,
    device: str = "cuda",
    sigma_v: float = 1,
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
