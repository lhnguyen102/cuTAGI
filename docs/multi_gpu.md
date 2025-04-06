# Distributed Data Parallel (DDP) with TAGI

This guide explains how to install dependencies and run DDP training using TAGI, which distributes batches across multiple GPUs and synchronizes updating values for mean and variances (`delta_mu_`, `delta_var_`) for parameter updates. Implementation details can be found in `src/ddp`.


```shell
                    +----------+
                    |    MPI   |
                    +-----------
                   /            \
+---------------------+     +---------------------+
|       GPU 0         |     |       GPU 1         |
|  Batch A            |     |  Batch B            |
|  Forward + Backward |     |  Forward + Backward |
|  Compute Updates    |     |  Compute Updates    |
+----------+----------+     +----------+----------+
             \                       /
              \   All-Reduce (NCCL) /
                +-----------------+
                | Aggregated      |
                | delta_mu_,      |
                | delta_var_      |
                +--------+--------+
                        |
                +---------+-------+
                |  Update Weights |
                +-----------------+
```

`All-Reduce` aggregates `deltas` (updates) from all GPUs, typically summing or averaging (`average=True`) them before updating model weights and biases. Here is an example to select either summing or averaging.

```python
from pytagi.nn import DDPSequential, DDPConfig, Linear, Sequential, ReLU

config = DDPConfig(device_ids=..., backend="nccl", rank=r..., world_size=...)
model = Sequential(Linear(1, 16), ReLU(), Linear(16, 2))
ddp_model = DDPSequential(model, config, average=True)
```

## Requirements
- Ubuntu 22.04 or later
- Compatible CUDA version for NCCL

## Installation Steps

### 1. Install MPI

MPI is required to run and manage multiple parallel processes.

```bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
```

Verify if MPU is installed on you machine by running this command

```shell
mpirun --version
```

### 2. Install NCCL

NCCL handles communication between GPUs.To install it, follow the [official NCCL guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html). Below is an example for CUDA 12.2. For other CUDA versions, refer to the guide for appropriate instructions.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install libnccl2=2.25.1-1+cuda12.2 libnccl-dev=2.25.1-1+cuda12.2
```

Verify if NCCL is installed on your machine by running this command

```shell
dpkg -l | grep nccl
```

### 3. Install MPI4PY
`mpi4py` is the Python package, it is required for Python script running on multiple-gpu

```shell
conda install mpi4py
```


## How to Use

### Python Example

Run CIFAR-10 training with ResNet18 on 2 GPUs:

```bash
mpirun -np 2 python -m examples.ddp_cifar_resnet
```

### C++ Example

Run the ResNet18 test in C++:

```bash
mpirun -np 2 build/run_tests --gtest_filter=ResNetDDPTest.ResNet_NCCL
```

## Troubleshooting

### 1. MPI4PY instalation issue?

If you cannot install `mpi4py` using `pip install mpi4py`, a workaround is to install it using coda

```shell
conda install mpi4py
```

### 2. Pytorch data loader

PyTorch's `DataLoader` uses multiprocessing. If you stop the script using `ctrl+c`, press it only once to avoid leaving zombie processes. To manually kill them:

```bash
ps aux | grep ddp_cifar_resnet
```

Find the process ID (PID), then:

```bash
kill -9 <PID>
```
