import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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

TAGI_FNN = Sequential(
    Linear(784, 4096),
    ReLU(),
    Linear(4096, 4096),
    ReLU(),
    Linear(4096, 11),
)

TAGI_CNN = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28),
    ReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5),
    ReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 256),
    ReLU(),
    Linear(256, 11),
)

TAGI_CNN_BATCHNORM = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28, bias=False),
    ReLU(),
    BatchNorm2d(16),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5, bias=False),
    ReLU(),
    BatchNorm2d(32),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 256),
    ReLU(),
    Linear(256, 11),
)


DATA_FOLDER = "./data/mnist"


# TORCH
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class TorchFNN(nn.Module):
    def __init__(self):
        super(TorchFNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )
        self.model.apply(initialize_weights)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.model(x)
        return x


class TorchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        self.model.apply(initialize_weights)

    def forward(self, x):
        return self.model(x)


class TorchCNNBatchNorm(nn.Module):
    def __init__(self):
        super(TorchCNNBatchNorm, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.model(x)


def custom_collate_fn(batch):
    # batch is a list of tuples (image, label)
    batch_images, batch_labels = zip(*batch)

    # Convert to a single tensor and then to numpy
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)

    # Flatten images and labels to 1D
    batch_images = batch_images.numpy().reshape(len(batch_images), -1).flatten()
    batch_labels = batch_labels.numpy().flatten()

    return batch_images, batch_labels


def tagi_trainer(
    batch_size: int, num_epochs: int, device: str = "cpu", sigma_v: float = 2.0
):
    # Data loading and preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )

    utils = Utils()

    # Hierachical Softmax
    metric = HRCSoftmaxMetric(num_classes=10)
    net = TAGI_FNN
    net.to_device(device)
    # net.set_threads(16)
    out_updater = OutputUpdater(net.device)

    # Training
    error_rates = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in pbar:
        # count = 0

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.3, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full(
            (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
        )
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


def torch_trainer(batch_size: int, num_epochs: int, device: str = "cpu"):
    # Hyperparameters
    learning_rate = 0.01

    # torch.set_float32_matmul_precision("high")

    # Data loading and preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=DATA_FOLDER, train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    torch_device = torch.device(device)
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )
    model = TorchFNN().to(torch_device)
    # model = torch.compile(model, mode="reduce-overhead")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    error_rates = []
    for epoch in pbar:
        model.train()
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
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(torch_device)
                target = target.to(torch_device)
                output = model(data)

                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_error_rate = (1.0 - correct / len(test_loader.dataset)) * 100
        pbar.set_description(
            f"Epoch# {epoch/num_epochs}| training error: {avg_error_rate:.2f}% | Test error: {test_error_rate: .2f}%"
        )


def main(
    framework: str = "tagi",
    batch_size: int = 256,
    epochs: int = 20,
    device: str = "cuda",
):
    if framework == "torch":
        torch_trainer(batch_size=batch_size, num_epochs=epochs, device=device)
    elif framework == "tagi":
        tagi_trainer(batch_size=batch_size, num_epochs=epochs, device=device)
    else:
        raise RuntimeError(f"Invalid Framework: {framework}")


if __name__ == "__main__":
    fire.Fire(main)
