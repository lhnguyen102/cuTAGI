#include "../include/ddp.h"

#include <stdexcept>

#include "../include/custom_logger.h"

#if defined(USE_NCCL) && defined(USE_CUDA) && defined(USE_MPI)
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include "../include/cuda_error_checking.cuh"
#ifdef USE_CUDA
#include "../include/base_layer_cuda.cuh"
#endif
#endif

DDPConfig::DDPConfig(const std::vector<int> &device_ids,
                     const std::string &backend, int rank, int world_size)
    : device_ids(device_ids),
      backend(backend),
      rank(rank),
      world_size(world_size) {}

#if defined(DISTRIBUTED_AVAILABLE)
// NCCLCommunicator implementation when dependencies are available
NCCLCommunicator::NCCLCommunicator(int rank,
                                   const std::vector<int> &device_ids) {
    this->rank = rank;
    this->world_size = device_ids.size();

    // Set device for this process
    cudaSetDevice(device_ids[rank]);

    // Initialize NCCL
    ncclUniqueId id;
    // Rank 0 creates the unique id. It communicates the id to all processes.
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    // All processes get the id, i.e., secret key to communicate with each other
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclCommInitRank(&comm, world_size, id, rank);
    cudaStreamCreate(&stream);
}

NCCLCommunicator::~NCCLCommunicator() {
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);
}

void NCCLCommunicator::all_reduce(float *data, size_t count, bool average) {
    // Use averaging if requested
    ncclRedOp_t op = average ? ncclAvg : ncclSum;
    ncclAllReduce(data, data, count, ncclFloat32, op, comm, stream);
    cudaStreamSynchronize(stream);
}

void NCCLCommunicator::barrier() { cudaStreamSynchronize(stream); }

void NCCLCommunicator::check_async_error() { CHECK_CUDA_NCCL_ASYNC(comm); }
#endif

// DDPSequential implementation
DDPSequential::DDPSequential(std::shared_ptr<Sequential> model,
                             const DDPConfig &config, bool average)
    : model(model), config(config), average(average) {
#if defined(DISTRIBUTED_AVAILABLE)

    if (config.device_ids.size() < 2) {
        LOG(LogLevel::ERROR, "DDP requires at least 2 devices.");
    }

    // Create appropriate communicator based on backend. The model that trained
    // on different batches to communicate with each other.
    if (config.backend == "nccl") {
        communicator =
            std::make_unique<NCCLCommunicator>(config.rank, config.device_ids);
    } else {
        LOG(LogLevel::ERROR, "Unsupported backend: " + config.backend);
    }

    // Move model to appropriate device
    if (config.backend == "nccl") {
        std::string device =
            "cuda:" + std::to_string(config.device_ids[config.rank]);
        model->to_device(device);
    }
#else
    LOG(LogLevel::WARNING,
        "Distributed training is not available. Make sure NCCL, CUDA, and MPI "
        "are enabled.");
    // Create a stub communicator
    communicator =
        std::make_unique<NCCLCommunicator>(config.rank, config.device_ids);
#endif
}

void DDPSequential::sync_parameters() {
#if defined(DISTRIBUTED_AVAILABLE)
    // Synchronize delta parameters across processes
    for (auto &layer : model->layers) {
        if (layer->get_layer_type() != LayerType::Activation &&
            layer->get_layer_type() != LayerType::Pool2d) {
            // Check if this is a CUDA layer
            auto cuda_layer = dynamic_cast<BaseLayerCuda *>(layer.get());
            if (cuda_layer) {
                // For CUDA layers, use device pointers directly
                // Synchronize weights and biases deltas
                communicator->all_reduce(cuda_layer->d_delta_mu_w,
                                         layer->delta_mu_w.size(), average);
                communicator->all_reduce(cuda_layer->d_delta_var_w,
                                         layer->delta_var_w.size(), average);

                if (layer->bias) {
                    communicator->all_reduce(cuda_layer->d_delta_mu_b,
                                             layer->delta_mu_b.size(), average);
                    communicator->all_reduce(cuda_layer->d_delta_var_b,
                                             layer->delta_var_b.size());
                }
            } else {
                LOG(LogLevel::ERROR, "CUDA layer not found");
            }
        }
    }
#endif
}

void DDPSequential::forward(const std::vector<float> &mu_a,
                            const std::vector<float> &var_a) {
    model->forward(mu_a, var_a);
}

void DDPSequential::backward() { model->backward(); }

void DDPSequential::step() {
#if defined(DISTRIBUTED_AVAILABLE)
    sync_parameters();
    model->step();
    communicator->check_async_error();
#else
    model->step();
#endif
}

void DDPSequential::train() { model->train(); }

void DDPSequential::eval() { model->eval(); }

void DDPSequential::barrier() { communicator->barrier(); }