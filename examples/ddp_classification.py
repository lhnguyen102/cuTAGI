import os
import sys

# Add the 'build' directory to sys.path
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
import fire
import numpy as np
from tqdm import tqdm
import mpi4py.MPI as MPI

from examples.data_loader import MnistDataLoader
from pytagi import HRCSoftmaxMetric
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
    ReLU(),
    Linear(128, 128),
    ReLU(),
    Linear(128, 11),
)


def main(num_epochs: int = 2, batch_size: int = 64, sigma_v: float = 0.2):
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

    # Set up distributed configuration
    device_ids = [i % 2 for i in range(world_size)]
    config = DDPConfig(
        device_ids=device_ids, backend="nccl", rank=rank, world_size=world_size
    )

    ddp_model = DDPSequential(CNN, config, average=True)

    # Load dataset - each process loads a subset of the data
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

    pbar = None
    if rank == 0:
        pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in range(num_epochs):
        batch_iter = train_dtl.create_data_loader(batch_size=batch_size)
        ddp_model.train()
        # print epoch
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Calculate total number of batches and ensure all processes handle the same number
        total_batches = train_dtl.dataset["value"][0].shape[0] // batch_size
        batches_per_process = total_batches // world_size
        if rank == 0:
            print(
                f"Total batches: {total_batches}, Batches per process: {batches_per_process}"
            )

        # Each process processes a subset of batches
        batch_count = 0
        process_batch_count = 0
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
            error_rate = metric.error_rate(m_pred, v_pred, label)
            error_rates.append(error_rate)

            batch_count += 1
            process_batch_count += 1

        # Synchronize at the end of each epoch
        ddp_model.barrier()
        comm.Barrier()

        # Testing (only on rank 0)
        if rank == 0:
            correct = 0
            num_samples = 0
            test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
            ddp_model.eval()

            for x, _, _, label in test_batch_iter:
                m_pred, v_pred = ddp_model(x)

                # Training metric
                pred = metric.get_predicted_labels(m_pred, v_pred)
                correct += np.sum(pred == label)
                num_samples += len(label)

            test_error_rate = (1.0 - correct / num_samples) * 100
            avg_error_rate = sum(error_rates[-100:]) / min(100, len(error_rates))

            if pbar:
                pbar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs} | training error: {avg_error_rate:.2f}% | test error: {test_error_rate:.2f}%",
                    refresh=True,
                )
                pbar.update(1)
    if rank == 0:
        print("Training complete.")
        if pbar:
            pbar.close()


if __name__ == "__main__":
    # Command: `mpirun -n 2 python -m examples.ddp_classification`
    try:
        fire.Fire(main)
    finally:
        if not MPI.Is_finalized():
            MPI.COMM_WORLD.Barrier()
            MPI.Finalize()
