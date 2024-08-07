import fire
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
from pytagi.nn import OutputUpdater
from examples.tagi_resnet_model import resnet18_cifar10
from examples.torch_resnet_model import ResNet18

torch.manual_seed(42)

# Constants for dataset normalization
NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)


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
            transforms.RandomHorizontalFlip(),
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
    num_epochs: int = 10,
    batch_size: int = 128,
    device: str = "cuda",
    sigma_v: float = 1.0,
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
    net = resnet18_cifar10()
    net.to_device(device)
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
                curr_v=sigma_v, min_v=0.3, decaying_factor=0.99, curr_iter=epoch
            )
            var_y = np.full(
                (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
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


def torch_trainer(batch_size: int, num_epochs: int, device: str = "cuda"):
    # Hyperparameters
    learning_rate = 0.01

    torch.set_float32_matmul_precision("high")
    train_loader, test_loader = load_datasets(batch_size, "torch")

    # Initialize the model, loss function, and optimizer
    torch_device = torch.device(device)
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )
    model = ResNet18()
    # model = torch.compile(model)
    model.to(torch_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
            f"Epoch# {epoch +1}/{num_epochs}| training error: {avg_error_rate:.2f}% | Test error: {test_error_rate: .2f}%"
        )


def main(
    framework: str = "tagi",
    batch_size: int = 128,
    epochs: int = 40,
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
