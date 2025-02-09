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
from torch.utils.data.dataset import Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from examples.tagi_resnet_model import resnet18_imagenet
from examples.tagi_alexnet_model import create_alexnet
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import OutputUpdater
import pytagi
from examples.batchnorm_viz import BatchNormViz
from examples.param_viz import ParameterDistributionVisualizer
from examples.param_stat_table import ParamStatTable, WandBLogger


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


def load_datasets(batch_size: int, framework: str = "torch", nb_classes=1000):
    """Load the ImageNet dataset."""
    # Data Transforms
    data_dir = "./data/imagenet/ILSVRC/Data/CLS-LOC"
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
    train_set = datasets.ImageFolder(
        f"{data_dir}/train", transform=train_transforms
    )
    val_set = datasets.ImageFolder(f"{data_dir}/val", transform=val_transforms)

    ## Select a subset of classes
    targets = range(nb_classes - 1)
    indices = [
        i for i, label in enumerate(train_set.targets) if label in targets
    ]
    train_set = Subset(train_set, indices)
    indices = [i for i, label in enumerate(val_set.targets) if label in targets]
    val_set = Subset(val_set, indices)

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
            num_workers=1,
            collate_fn=custom_collate_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=custom_collate_fn,
        )

    return train_loader, val_loader


def tagi_trainer(
    num_epochs: int,
    batch_size: int,
    device: str,
    sigma_v: float,
    nb_classes: int = 1000,
    is_tracking: bool = False,
):
    """
    Run classification training on the Cifar dataset using a custom neural model.

    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    """
    # User data
    print_var = True
    viz_norm_stats = False  # print norm stats at last epoch
    viz_param = False  # visualize parameter distributions
    print_param_stat = True  # print mean and std of parameters
    is_tracking = (
        is_tracking if print_param_stat else False
    )  # track params with wandb

    # Load datasets
    utils = Utils()
    train_loader, test_loader = load_datasets(
        batch_size, "tagi", nb_classes=nb_classes
    )

    # Viz tools
    batch_norm_viz = BatchNormViz()
    param_viz = ParameterDistributionVisualizer()
    param_stat = ParamStatTable()

    if is_tracking:
        wandb_logger = WandBLogger(
            project_name="resnet",
            config={
                "sigma_v": sigma_v,
                "dataset": "imagenet",
                "nb_classes": nb_classes,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
            },
        )
        wandb_logger.init()

    # Hierachical Softmax
    metric = HRCSoftmaxMetric(num_classes=nb_classes)

    # Resnet18
    # net = resnet18_imagenet(gain_w=0.1, gain_b=0.1, nb_outputs=metric.hrc_softmax.len)
    net = create_alexnet(
        gain_w=0.1, gain_b=0.1, nb_outputs=metric.hrc_softmax.len
    )
    device = "cpu" if not pytagi.cuda.is_available() else device
    net.to_device(device)

    # Access parameters
    if viz_param:
        net.preinit_layer()
        state_dict = net.state_dict()
        param_viz.record_params(state_dict)

    if print_param_stat:
        net.preinit_layer()
        state_dict = net.state_dict()
        param_stat.record_params(state_dict)
        param_stat.print_current_parameter_distributions(topN=10)
        if is_tracking:
            log_data = param_stat.rows_to_dict()
            wandb_logger.log(log_data)

    # Training
    out_updater = OutputUpdater(net.device)
    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs), sigma_v**2, dtype=np.float32
    )
    with tqdm(range(num_epochs), desc="Epoch Progress") as epoch_pbar:
        for epoch in epoch_pbar:
            train_correct = 0
            net.train()
            with tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} - Batch Progress",
            ) as batch_pbar:
                for i, (x, labels) in enumerate(batch_pbar):
                    m_pred, v_pred = net(x)
                    if np.isnan(np.mean(m_pred)):
                        print("m_pred is nan")
                        break
                    if print_var:  # Print prior predictive variance
                        print(
                            "Prior predictive -> E[v_pred] = ",
                            np.average(v_pred),
                            " | E[s_pred]",
                            np.average(np.sqrt(v_pred)),
                        )
                        print(
                            "                 -> V[m_pred] = ",
                            np.var(m_pred),
                            " | s[m_pred]",
                            np.std(m_pred),
                        )
                        print_var = False

                    # Update output layers based on targets
                    y, y_idx, _ = utils.label_to_obs(
                        labels=labels, num_classes=nb_classes
                    )
                    out_updater.update_using_indices(
                        output_states=net.output_z_buffer,
                        mu_obs=y / 1,
                        var_obs=var_y,
                        selected_idx=y_idx,
                        delta_states=net.input_delta_z_buffer,
                    )
                    net.backward()
                    net.step()

                    # Training metric
                    pred = metric.get_predicted_labels(m_pred, v_pred)
                    train_correct += np.sum(pred == labels)

                    if i > 0 and i % 100 == 0:
                        avg_error_rate = (
                            1.0 - (train_correct / ((i + 1) * batch_size))
                        ) * 100
                        batch_pbar.set_description(
                            f"Training error: {avg_error_rate:.2f}%",
                            refresh=True,
                        )

            # Averaged training error
            avg_error_rate = (
                1.0 - train_correct / len(train_loader.dataset)
            ) * 100

            # Get batchnorm statistics
            if viz_norm_stats:
                train_norm_stats = net.get_norm_mean_var()
                batch_norm_viz.update(train_norm_stats, "train")

            # Param viz
            if viz_param:
                state_dict = net.state_dict()
                param_viz.record_params(state_dict)
                param_viz.plot_distributions(
                    output_dir="saved_results/param_viz"
                )
                param_viz.plot_initial_vs_final_differences(
                    output_dir="saved_results/param_viz"
                )

            # Testing
            correct = 0
            net.eval()
            for x, labels in test_loader:
                m_pred, v_pred = net(x)

                # Training metric
                pred = metric.get_predicted_labels(m_pred, v_pred)
                correct += np.sum(pred == labels)

                if viz_norm_stats and epoch == num_epochs - 1:
                    test_norm_stats = net.get_norm_mean_var()
                    batch_norm_viz.update(test_norm_stats, "test")
                    batch_norm_viz.plot_all_layers(
                        folder_name="saved_results/batchnorm"
                    )

            test_error_rate = (1.0 - correct / len(test_loader.dataset)) * 100

            if print_param_stat:
                param_stat.record_params(net.state_dict())
                param_stat.print_current_parameter_distributions(topN=10)
                if is_tracking:
                    log_data = param_stat.rows_to_dict()
                    log_data["train_error"] = avg_error_rate
                    log_data["test_error"] = test_error_rate
                    wandb_logger.log(log_data)

            epoch_pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs} | training error: {avg_error_rate:.2f}% | test error: {test_error_rate:.2f}%",
                refresh=True,
            )
    if is_tracking:
        wandb_logger.finish()
    print("Training complete.")


def torch_trainer(
    batch_size: int,
    num_epochs: int,
    device: str = "cuda",
    nb_classes: int = 1000,
):
    """Train ResNet-18 on the ImageNet dataset."""
    # torch.set_float32_matmul_precision("high")

    # Hyperparameters
    learning_rate = 0.0003

    # Load ImageNet datasets
    train_loader, val_loader = load_datasets(
        batch_size, "torch", nb_classes=nb_classes
    )

    # Initialize the model
    torch_device = torch.device(device)
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your CUDA installation."
        )
    # model = models.resnet18(weights=None)
    # num_ftrs = model.fc.in_features
    # # model.fc = nn.Linear(num_ftrs, len(train_loader.dataset.classes))
    # model.fc = nn.Linear(num_ftrs, nb_classes)
    model = models.AlexNet(num_classes=nb_classes)
    model.to(torch_device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    with tqdm(range(num_epochs), desc="Epoch Progress") as epoch_pbar:
        for epoch in epoch_pbar:
            model.train()
            train_correct = 0

            with tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} - Batch Progress",
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
                    train_correct += pred.eq(target.view_as(pred)).sum().item()

                    if i > 0 and i % 100 == 0:
                        avg_error_rate = (
                            1.0 - (train_correct / ((i + 1) * batch_size))
                        ) * 100
                        batch_pbar.set_description(
                            f"Training error: {avg_error_rate:.2f}%",
                            refresh=True,
                        )

            # Averaged training error
            avg_error_rate = (
                1.0 - train_correct / len(train_loader.dataset)
            ) * 100

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
    batch_size: int = 128,
    epochs: int = 20,
    device: str = "cuda",
    sigma_v: float = 0.1,
    nb_classes: int = 8,
):
    if framework == "torch":
        torch_trainer(
            batch_size=batch_size,
            num_epochs=epochs,
            device=device,
            nb_classes=nb_classes,
        )
    elif framework == "tagi":
        tagi_trainer(
            batch_size=batch_size,
            num_epochs=epochs,
            device=device,
            sigma_v=sigma_v,
            nb_classes=nb_classes,
        )
    else:
        raise RuntimeError(f"Invalid Framework: {framework}")


if __name__ == "__main__":
    fire.Fire(main)
