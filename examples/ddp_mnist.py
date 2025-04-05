import os
import sys

# Add the 'build' directory to sys.path
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
import fire
import mpi4py.MPI as MPI
import numpy as np
from tqdm import tqdm

import pytagi
from examples.data_loader import MnistDataLoader
from pytagi import HRCSoftmaxMetric
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
    ReLU(),
    Linear(128, 128),
    ReLU(),
    Linear(128, 11),
)

pytagi.manual_seed(0)


def main(num_epochs: int = 10, batch_size: int = 256, sigma_v: float = 0.1):
    """
    Run distributed classification training on the MNIST dataset.
    Parameters:
    - num_epochs: int, number of epochs for training
    - batch_size: int, size of the batch for training
    - sigma_v: float, variance for the output
    """
    if not pytagi.cuda.is_available():
        raise ValueError("CUDA is not available")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    print(f"Process {rank} of {world_size} starting distributed training")

    # Set seed (add rank to the seed to ensure different processes have different seeds)
    seed = 42 + rank
    pytagi.manual_seed(seed)

    # Set up distributed configuration
    device_ids = [i % 2 for i in range(world_size)]
    config = DDPConfig(
        device_ids=device_ids, backend="nccl", rank=rank, world_size=world_size
    )

    ddp_model = DDPSequential(CNN, config, average=True)

    # Load dataset - each process loads the full dataset, but will process only its subset
    train_dtl = MnistDataLoader(
        x_file="data/mnist/train-images-idx3-ubyte",
        y_file="data/mnist/train-labels-idx1-ubyte",
        num_images=60000,
    )
    test_dtl = MnistDataLoader(
        x_file="data/mnist/t10k-images-idx3-ubyte",
        y_file="data/mnist/t10k-labels-idx1-ubyte",
        num_images=10000,
    )

    # Hierarchical Softmax
    metric = HRCSoftmaxMetric(num_classes=10)

    # Create output updater
    device = "cuda:" + str(device_ids[rank])
    out_updater = OutputUpdater(device)

    # Training
    error_rates = []
    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
    )

    # Progress bar (only on rank 0)
    pbar = None
    if rank == 0:
        pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in range(num_epochs):
        batch_iter = train_dtl.create_data_loader(batch_size=batch_size)
        ddp_model.train()

        # Calculate total number of batches and ensure all processes handle the same number
        total_batches = train_dtl.dataset["value"][0].shape[0] // batch_size
        batches_per_process = total_batches // world_size

        # Each process processes a subset of batches
        batch_count = 0
        process_batch_count = 0

        total_train_error = 0
        total_train_samples = 0

        for x, y, y_idx, label in batch_iter:
            # Skip batches not assigned to this process
            if batch_count % world_size != rank:
                batch_count += 1
                continue

            # Stop if this process has handled its share of batches
            if process_batch_count >= batches_per_process:
                break

            m_pred, v_pred = ddp_model(x)

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
            error_rate = metric.error_rate(m_pred, v_pred, label)
            error_rates.append(error_rate)

            # Calculate errors per batch
            pred = metric.get_predicted_labels(m_pred, v_pred)
            batch_errors = np.sum(pred != label)
            total_train_error += batch_errors
            total_train_samples += len(label)

            batch_count += 1
            process_batch_count += 1

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
        test_batch_per_process = (
            test_dtl.dataset["value"][0].shape[0] // world_size
        )

        test_batch_iter = test_dtl.create_data_loader(batch_size=batch_size)

        test_batch_count = 0
        test_process_batch_count = 0

        for x_batch, _, _, label in test_batch_iter:
            # Skip batches not assigned to this process
            if batch_count % world_size != rank:
                batch_count += 1
                continue

            # Stop if this process has handled its share of batches
            if test_process_batch_count >= test_batch_per_process:
                break

            # Forward pass
            m_pred, v_pred = ddp_model(x_batch)

            # Calculate errors
            pred = metric.get_predicted_labels(m_pred, v_pred)
            batch_errors = np.sum(pred != label)

            local_test_error += batch_errors
            local_test_samples += len(label)

            test_batch_count += 1
            test_process_batch_count += 1

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
    # Command: `mpirun -n <num_processes> python -m examples.ddp_classification`
    try:
        fire.Fire(main)
    finally:
        if not MPI.Is_finalized():
            MPI.COMM_WORLD.Barrier()
            MPI.Finalize()
