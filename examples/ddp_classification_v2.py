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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import mpi4py.MPI as MPI

from pytagi import HRCSoftmaxMetric, Utils
import pytagi
from pytagi.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    DDPConfig,
    DDPSequential,
    Linear,
    MixtureReLU,
    OutputUpdater,
    Sequential,
    ReLU,
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


def ignore_sigint_in_worker(_):
    """
    Called once for each DataLoader worker. We ignore the SIGINT signal
    in the workers so the main process can handle Ctrl+C gracefully.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def load_datasets(batch_size: int, rank: int, world_size: int):
    """
    Load MNIST datasets using PyTorch's DataLoader with DistributedSampler for training,
    and a regular DataLoader for testing (only on rank 0).

    Args:
        batch_size: Size of each batch
        rank: Current process rank
        world_size: Total number of processes
    """
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Normalize with MNIST mean and std
            transforms.Normalize((0.1307,), (1.0,)),
        ]
    )

    # Load datasets
    train_set = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    # Calculate sizes for distributed training
    train_size = len(train_set)
    samples_per_process = train_size // world_size

    # Create distributed sampler for training
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )

    # Create training DataLoader with distributed sampler
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
        num_workers=2,
        worker_init_fn=ignore_sigint_in_worker,  # <-- Add worker_init_fn
        drop_last=False,
    )

    # Create test loader without distributed sampler (only for rank 0)
    test_loader = None
    if rank == 0:
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=2,
            worker_init_fn=ignore_sigint_in_worker,  # <-- Add worker_init_fn
        )

    return (
        train_loader,
        test_loader,
        train_sampler,
        None,  # No test sampler needed
        samples_per_process,  # Return samples per process instead of total size
        len(test_set),
    )


def main(num_epochs: int = 10, batch_size: int = 128, sigma_v: float = 0.05):
    """
    Run distributed classification training on the MNIST dataset using PyTorch DataLoader.
    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    - sigma_v: float, variance for the output
    """

    # Set up signal handling in the main process
    def signal_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not pytagi.cuda.is_available():
        raise ValueError("CUDA is not available")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Set up distributed configuration
    device_ids = [i % 2 for i in range(world_size)]
    config = DDPConfig(
        device_ids=device_ids, backend="nccl", rank=rank, world_size=world_size
    )

    ddp_model = None
    train_loader = None
    test_loader = None

    try:
        # Build and wrap the model in DDP
        ddp_model = DDPSequential(CNN, config, average=False)

        # Load dataset using PyTorch DataLoader with DistributedSampler
        train_loader, test_loader, train_sampler, _, _, _ = load_datasets(
            batch_size, rank, world_size
        )
        utils = Utils()

        # Hierarchical Softmax
        metric = HRCSoftmaxMetric(num_classes=10)

        # Create output updater
        device = "cuda:" + str(device_ids[rank])
        out_updater = OutputUpdater(device)

        # Training
        error_rates = []
        var_y = np.full(
            (batch_size * metric.hrc_softmax.num_obs,),
            sigma_v**2,
            dtype=np.float32,
        )

        pbar = None
        if rank == 0:
            pbar = tqdm(range(num_epochs), desc="Training Progress")

        for epoch in range(num_epochs):
            # Set epoch for the sampler (important for shuffling)
            train_sampler.set_epoch(epoch)

            ddp_model.train()
            if rank == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")

            # Process batches assigned to this process by the sampler
            epoch_error_rates = []

            for x, labels in train_loader:
                # Convert labels to observations using Utils
                y, y_idx, _ = utils.label_to_obs(labels=labels, num_classes=10)

                # Forward pass
                m_pred, v_pred = ddp_model(x)

                # Update output layers based on targets
                out_updater.update_using_indices(
                    output_states=ddp_model.model.output_z_buffer,
                    mu_obs=y,
                    var_obs=var_y,
                    selected_idx=y_idx,
                    delta_states=ddp_model.model.input_delta_z_buffer,
                )

                # Update parameters
                ddp_model.backward()
                ddp_model.step()

                # Training metric
                error_rate = metric.error_rate(m_pred, v_pred, labels)
                error_rates.append(error_rate)
                epoch_error_rates.append(error_rate)

            # Synchronize at the end of each epoch
            ddp_model.barrier()
            comm.Barrier()

            # Gather error rates from all processes
            all_error_rates = comm.gather(epoch_error_rates, root=0)

            # Testing (only on rank 0)
            ddp_model.eval()
            test_error_rate = None
            if rank == 0 and test_loader is not None:
                correct = 0
                num_samples = 0

                for x, labels in test_loader:
                    m_pred, v_pred = ddp_model(x)
                    # Get predictions
                    pred = metric.get_predicted_labels(m_pred, v_pred)
                    correct += np.sum(pred == labels)
                    num_samples += len(labels)

                test_error_rate = (1.0 - correct / num_samples) * 100

            # Only rank 0 computes and displays metrics
            if rank == 0:
                # Combine error rates from all processes
                flat_error_rates = [
                    rate for process_rates in all_error_rates for rate in process_rates
                ]
                avg_error_rate = sum(flat_error_rates) / len(flat_error_rates) * 100

                if pbar:
                    pbar.set_description(
                        f"Epoch {epoch + 1}/{num_epochs} | training error: {avg_error_rate:.2f}% | test error: {test_error_rate:.2f}%",
                        refresh=True,
                    )
                    pbar.update(1)

    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by Ctrl+C. Cleaning up...")
    except Exception as e:
        if rank == 0:
            print(f"\nError occurred: {e}")
    finally:
        # Attempt final synchronization and cleanup
        if ddp_model is not None:
            try:
                ddp_model.barrier()
            except:
                pass

        try:
            comm.Barrier()
        except:
            pass

        if rank == 0:
            if pbar:
                pbar.close()
            print("Cleanup complete.")


if __name__ == "__main__":
    # Command: `mpirun -n 2 python -m examples.ddp_classification_v2`
    fire.Fire(main)
    # Ensure MPI is properly finalized
    if MPI.Is_initialized() and not MPI.Is_finalized():
        try:
            MPI.COMM_WORLD.Barrier()
        except:
            pass
        MPI.Finalize()
