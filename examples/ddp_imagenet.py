import os
import signal
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)
import fire
import mpi4py.MPI as MPI
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

import pytagi
from examples.tagi_alexnet_model import create_alexnet
from examples.tagi_resnet_model import resnet18_imagenet
from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
from pytagi.nn import (
    BatchNorm2d,
    Conv2d,
    DDPConfig,
    DDPSequential,
    LayerBlock,
    Linear,
    MixtureReLU,
    OutputUpdater,
    ResNetBlock,
    Sequential,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]


def signal_handler(signum, frame):
    print(f"Rank {rank} caught signal {signum}. Preparing to shut down.")
    comm.Abort()


signal.signal(signal.SIGINT, signal_handler)


def custom_collate_fn(batch):
    batch_images, batch_labels = zip(*batch)
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)
    batch_images = batch_images.reshape(-1)
    batch_images = batch_images.numpy()
    batch_labels = batch_labels.numpy()
    return batch_images, batch_labels


def load_datasets(
    batch_size: int,
    data_dir: str,
    world_size: int,
    rank: int,
    seed: int,
    num_workers: int = 4,
    nb_classes: int = 1000,
):
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD
            ),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        f"{data_dir}/train", transform=train_transforms
    )
    val_dataset = torchvision.datasets.ImageFolder(
        f"{data_dir}/val", transform=val_transforms
    )

    targets = range(nb_classes - 1)
    train_indices = [
        i for i, label in enumerate(train_dataset.targets) if label in targets
    ]
    val_indices = [
        i for i, label in enumerate(val_dataset.targets) if label in targets
    ]
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


def main(
    num_epochs: int = 2,
    batch_size: int = 128,
    sigma_v: float = 0.1,
    gain_w: float = 0.10,
    gain_b: float = 0.10,
    data_dir: str = "./data/imagenet/ILSVRC/Data/CLS-LOC",
    num_workers: int = 4,
    nb_classes: int = 1000,
):
    if not pytagi.cuda.is_available():
        raise ValueError("CUDA is not available")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    utils = Utils()
    print(f"Process {rank} of {world_size} starting distributed training")

    seed = 42 + rank
    pytagi.manual_seed(seed)
    torch.manual_seed(seed)

    device_ids = [i % torch.cuda.device_count() for i in range(world_size)]
    config = DDPConfig(
        device_ids=device_ids, backend="nccl", rank=rank, world_size=world_size
    )

    tagi_model = create_alexnet(
        gain_w=gain_w, gain_b=gain_b, nb_outputs=nb_classes
    )
    ddp_model = DDPSequential(tagi_model, config, average=True)

    train_loader, val_loader, train_sampler = load_datasets(
        batch_size, data_dir, world_size, rank, seed, num_workers, nb_classes
    )

    metric = HRCSoftmaxMetric(num_classes=nb_classes)
    device = ddp_model.get_device_with_index()
    out_updater = OutputUpdater(device)

    var_y = np.full(
        (batch_size * metric.hrc_softmax.num_obs,), sigma_v**2, dtype=np.float32
    )

    pbar = None
    if rank == 0:
        pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        if epoch > 0:
            sigma_v = exponential_scheduler(
                curr_v=sigma_v, min_v=0, decaying_factor=1, curr_iter=epoch
            )
            var_y = np.full(
                (batch_size * metric.hrc_softmax.num_obs,),
                sigma_v**2,
                dtype=np.float32,
            )

        ddp_model.train()
        total_train_error = 0
        total_train_samples = 0

        for x_batch, labels in train_loader:
            y, y_idx, _ = utils.label_to_obs(
                labels=labels, num_classes=nb_classes
            )
            m_pred, v_pred = ddp_model(x_batch)

            out_updater.update_using_indices(
                output_states=ddp_model.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=ddp_model.input_delta_z_buffer,
            )

            ddp_model.backward()
            ddp_model.step()

            pred = metric.get_predicted_labels(m_pred, v_pred)
            batch_errors = np.sum(pred != labels)
            total_train_error += batch_errors
            total_train_samples += len(labels)

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

        ddp_model.eval()
        local_val_error = 0
        local_val_samples = 0

        for x_batch, labels in val_loader:
            m_pred, v_pred = ddp_model(x_batch)
            pred = metric.get_predicted_labels(m_pred, v_pred)
            batch_errors = np.sum(pred != labels)
            local_val_error += batch_errors
            local_val_samples += len(labels)

        total_val_error = comm.reduce(local_val_error, op=MPI.SUM, root=0)
        total_val_samples = comm.reduce(local_val_samples, op=MPI.SUM, root=0)

        if rank == 0:
            global_val_error_rate = (
                total_val_error / total_val_samples
                if total_val_samples > 0
                else 0.0
            )

            if pbar:
                pbar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs} | train error: {global_train_error_rate * 100:.2f}% | val error: {global_val_error_rate * 100:.2f}%",
                    refresh=True,
                )
                pbar.update(1)

    if rank == 0:
        print("Training complete.")
        if pbar:
            pbar.close()


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except KeyboardInterrupt:
        pass
    finally:
        if not MPI.Is_finalized():
            MPI.COMM_WORLD.Barrier()
            MPI.Finalize()
