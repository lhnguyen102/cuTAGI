import os
import sys
import signal

# Add the 'build' directory to sys.path
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
import fire
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import mpi4py.MPI as MPI

from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
import pytagi
from pytagi.nn import (
    BatchNorm2d,
    Conv2d,
    DDPConfig,
    DDPSequential,
    Linear,
    MixtureReLU,
    OutputUpdater,
    ResNetBlock,
    LayerBlock,
    Sequential,
)
from examples.tagi_resnet_model import resnet18_cifar10

# Initialize MPI communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Constants for dataset normalization
NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD = [0.2470, 0.2435, 0.2616]


def signal_handler(signum, frame):
    # Optionally print or log something
    print(f"Rank {rank} caught signal {signum}. Preparing to shut down.")
    comm.Abort()


# Install the handler on every rank
signal.signal(signal.SIGINT, signal_handler)
# signal.signal(signal.SIGTERM, signal_handler)


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


def load_datasets(
    batch_size: int,
    data_dir: str,
    world_size: int,
    rank: int,
    seed: int,
    num_workers: int = 2,
):
    """Load and transform CIFAR10 training and test datasets with distributed samplers."""
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

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test,
    )

    # Training set with distributed sampler
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test set with distributed sampler
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_sampler


def main(
    num_epochs: int = 2,
    batch_size: int = 256,
    sigma_v: float = 0.1,
    gain_w: float = 0.10,
    gain_b: float = 0.10,
    data_dir: str = "./data/cifar",
    num_workers: int = 2,
):
    """
    Run distributed data parallel classification training on the CIFAR-10 dataset with ResNet.

    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    - sigma_v: float, variance for the output
    - gain_w: float, weight gain for initialization
    - gain_b: float, bias gain for initialization
    - data_dir: str, directory to store/load the CIFAR-10 dataset
    - num_workers: int, number of workers for data loading
    """
    if not pytagi.cuda.is_available():
        raise ValueError("CUDA is not available")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    utils = Utils()

    print(f"Process {rank} of {world_size} starting distributed training")

    # Set seed
    seed = 42 + rank
    pytagi.manual_seed(seed)
    torch.manual_seed(seed)

    # Set up distributed configuration
    # Distribute processes across available GPUs
    device_ids = [i % torch.cuda.device_count() for i in range(world_size)]
    config = DDPConfig(
        device_ids=device_ids, backend="nccl", rank=rank, world_size=world_size
    )

    # Create ResNet model and wrap it with DDPSequential
    tagi_model = resnet18_cifar10(gain_w=gain_w, gain_b=gain_b)
    ddp_model = DDPSequential(tagi_model, config, average=True)

    # Load datasets with distributed samplers
    train_loader, test_loader, train_sampler = load_datasets(
        batch_size, data_dir, world_size, rank, seed, num_workers
    )

    # Hierarchical Softmax metric
    metric = HRCSoftmaxMetric(num_classes=10)

    # Create output updater
    device = "cuda:" + str(device_ids[rank])
    out_updater = OutputUpdater(device)

    # Initial variance for observations
    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
    )

    # Progress bar (only on rank 0)
    pbar = None
    if rank == 0:
        pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in range(num_epochs):
        # Update sampler's epoch
        train_sampler.set_epoch(epoch)

        # Decay sigma_v over time
        if epoch > 0:
            sigma_v = exponential_scheduler(
                curr_v=sigma_v, min_v=0, decaying_factor=1, curr_iter=epoch
            )
            var_y = np.full(
                (batch_size * metric.hrc_softmax.num_obs,),
                sigma_v**2,
                dtype=np.float32,
            )

        # Training
        ddp_model.train()
        total_train_error = 0
        total_train_samples = 0

        for x_batch, labels in train_loader:
            y, y_idx, _ = utils.label_to_obs(labels=labels, num_classes=10)
            m_pred, v_pred = ddp_model(x_batch)

            # Update output layers based on targets
            out_updater.update_using_indices(
                output_states=ddp_model.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=ddp_model.input_delta_z_buffer,
            )

            # Backward pass and parameter update
            ddp_model.backward()
            ddp_model.step()

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

        # Testing (distributed across all processes)
        ddp_model.eval()
        local_test_error = 0
        local_test_samples = 0

        for x_batch, labels in test_loader:
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
                total_test_error / total_test_samples if total_test_samples > 0 else 0.0
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
    # Command: `mpirun -n <num_processes> python -m examples.ddp_cifar_resnet`
    try:
        fire.Fire(main)
    except KeyboardInterrupt:
        pass
    finally:
        if not MPI.Is_finalized():
            MPI.COMM_WORLD.Barrier()
            MPI.Finalize()
