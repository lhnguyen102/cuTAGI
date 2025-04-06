import os
import signal
import sys

# Add the 'build' directory to sys.path
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
import fire
import mpi4py.MPI as MPI
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import pytagi
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    DDPConfig,
    DDPSequential,
    Linear,
    MixtureReLU,
    OutputUpdater,
    ReLU,
    Sequential,
)

# Define a simple CNN model
CNN = Sequential(
    Conv2d(1, 16, 4, padding=1, in_width=28, in_height=28),
    ReLU(),
    AvgPool2d(3, 2),
    Conv2d(16, 32, 5),
    ReLU(),
    AvgPool2d(3, 2),
    Linear(32 * 4 * 4, 100),
    ReLU(),
    Linear(100, 11),
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
    Linear(100, 11),
)

FNN = Sequential(
    Linear(784, 128),
    MixtureReLU(),
    Linear(128, 128),
    MixtureReLU(),
    Linear(128, 11),
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def signal_handler(signum, frame):
    # Optionally print or log something
    print(f"Rank {rank} caught signal {signum}. Preparing to shut down.")
    comm.Abort()


# Install the handler on every rank
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def custom_collate_fn(batch):
    """
    Custom collate function for PyTorch DataLoader to convert data to
    the format expected by TAGI (flattened numpy arrays).
    """
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


pytagi.manual_seed(0)


def main(
    num_epochs: int = 10,
    batch_size: int = 128,
    sigma_v: float = 0.1,
    num_workers: int = 2,
    data_dir: str = "./data",
):
    """
    Run distributed classification training on the MNIST dataset using PyTorch DataLoader.
    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    - sigma_v: float, variance for the output
    - data_dir: str, directory to store/load the MNIST dataset
    """
    if not pytagi.cuda.is_available():
        raise ValueError("CUDA is not available")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    utils = Utils()

    print(f"Process {rank} of {world_size} starting distributed training")

    # Set seed (add rank to the seed to ensure different processes have different seeds)
    seed = 42 + rank
    pytagi.manual_seed(seed)
    torch.manual_seed(seed)

    # Set up distributed configuration
    device_ids = [i % 2 for i in range(world_size)]
    config = DDPConfig(
        device_ids=device_ids, backend="nccl", rank=rank, world_size=world_size
    )

    ddp_model = DDPSequential(CNN, config, average=True)

    # Data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Training set
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test set
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Hierarchical Softmax
    metric = HRCSoftmaxMetric(num_classes=10)

    # Create output updater
    device = "cuda:" + str(device_ids[rank])
    out_updater = OutputUpdater(device)

    # Training
    error_rates = []

    # Progress bar (only on rank 0)
    pbar = None
    if rank == 0:
        pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        ddp_model.train()
        total_train_error = 0
        total_train_samples = 0

        for x_batch, labels in train_loader:
            y, y_idx, _ = utils.label_to_obs(labels=labels, num_classes=10)

            # Set variance for observations
            var_y = np.full(
                (len(labels) * metric.hrc_softmax.num_obs,),
                sigma_v**2,
                dtype=np.float32,
            )

            m_pred, v_pred = ddp_model(x_batch)

            # Update output layers based on targets
            out_updater.update_using_indices(
                output_states=ddp_model.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=ddp_model.input_delta_z_buffer,
            )

            # Update parameters
            ddp_model.backward()
            ddp_model.step()

            # Training metric
            error_rate = metric.error_rate(m_pred, v_pred, labels)
            error_rates.append(error_rate)

            # Calculate errors per batch
            pred = metric.get_predicted_labels(m_pred, v_pred)
            batch_errors = np.sum(pred != labels)
            total_train_error += batch_errors
            total_train_samples += len(labels)

        # Gather training errors from all processes
        all_train_errors = comm.gather(total_train_error, root=0)
        all_train_samples = comm.gather(total_train_samples, root=0)

        if rank == 0 and all_train_errors and all_train_samples:
            global_train_error = sum(all_train_errors)
            global_train_samples = sum(all_train_samples)
            global_train_error_rate = (
                global_train_error / global_train_samples
                if global_train_samples > 0
                else 0.0
            )

        ############################################################################
        # Testing (distributed across all processes)
        ############################################################################
        ddp_model.eval()

        local_test_error = 0
        local_test_samples = 0

        for x_batch, labels in test_loader:
            y, y_idx, _ = utils.label_to_obs(labels=labels, num_classes=10)

            # Forward pass
            m_pred, v_pred = ddp_model(x_batch)

            # Calculate errors
            pred = metric.get_predicted_labels(m_pred, v_pred)
            batch_errors = np.sum(pred != labels)

            local_test_error += batch_errors
            local_test_samples += len(labels)

        # Use MPI to gather and reduce the test error statistics
        total_test_error = comm.reduce(local_test_error, op=MPI.SUM, root=0)
        total_test_samples = comm.reduce(local_test_samples, op=MPI.SUM, root=0)

        # Calculate and print final test error rate on rank 0
        if rank == 0:
            global_test_error_rate = (
                total_test_error / total_test_samples
                if total_test_samples > 0
                else 0.0
            )

            if pbar:
                pbar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs} | train error: {global_train_error_rate * 100:.2f}% | test error: {global_test_error_rate * 100:.2f}%",
                    refresh=True,
                )
                pbar.update(1)

    if rank == 0:
        print("Training complete.")
        if pbar:
            pbar.close()


if __name__ == "__main__":
    # Command: `mpirun -n <num_processes> python -m examples.ddp_classification_v2`
    try:
        fire.Fire(main)
    except KeyboardInterrupt:
        pass
    finally:
        if not MPI.Is_finalized():
            MPI.COMM_WORLD.Barrier()
            MPI.Finalize()
