```markdown
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
           \                      /
            \ All-Reduce (NCCL)  /
             +-----------------+
             | Aggregated      |
             | delta_mu_,      |
             | delta_var_      |
             +--------+--------+
                      |
            +---------v---------+
            |  Update Weights   |
            +-------------------+
```

## Requirements

- Ubuntu 22.04 or later
- Compatible CUDA version for NCCL
- Python or C++ build of TAGI

## Installation Steps

### 1. Install MPI

MPI is required to run and manage multiple parallel processes.

```bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
```

### 2. Install NCCL

NCCL handles communication between GPUs. Follow the [official NCCL guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html) or use the commands below:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install libnccl2=2.25.1-1+cuda12.2 libnccl-dev=2.25.1-1+cuda12.2
```

## How to Use

### Python Example

Run CIFAR-10 training with ResNet18 on 2 GPUs:

```bash
mpirun -np 2 python examples/ddp_cifar_resnet.py
```

### C++ Example

Run the ResNet18 test in C++:

```bash
mpirun -np 2 build/run_tests --gtest_filter=ResNetDDPTest.ResNet_NCCL
```

## Troubleshooting

PyTorch's `DataLoader` uses multiprocessing. If you stop the script using `ctrl+c`, press it only once to avoid leaving zombie processes. To manually kill them:

```bash
ps aux | grep ddp_cifar_resnet
```

Find the process ID (PID), then:

```bash
kill -9 <PID>
```
