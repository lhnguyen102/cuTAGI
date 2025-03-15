#include "../include/distributed.h"

#include <stdexcept>

#include "../include/custom_logger.h"

#if defined(USE_NCCL) && defined(USE_CUDA) && defined(USE_MPI)
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include "../include/cuda_error_checking.cuh"
#endif

// DistributedConfig implementation is always available
DistributedConfig::DistributedConfig(const std::vector<int> &device_ids,
                                     const std::string &backend, int rank,
                                     int world_size)
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

void NCCLCommunicator::all_reduce(float *data, size_t count) {
    // Perform all-reduce operation i.e., sum of all the data across all
    // processes
    ncclAllReduce(data, data, count, ncclFloat32, ncclSum, comm, stream);
    cudaStreamSynchronize(stream);
}

void NCCLCommunicator::barrier() { cudaStreamSynchronize(stream); }

void NCCLCommunicator::check_async_error() { CHECK_CUDA_NCCL_ASYNC(comm); }
#endif

// DistributedSequential implementation
DistributedSequential::DistributedSequential(std::shared_ptr<Sequential> model,
                                             const DistributedConfig &config,
                                             bool average)
    : model(model), config(config), average(average) {
#if defined(DISTRIBUTED_AVAILABLE)
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

void DistributedSequential::sync_parameters() {
#if defined(DISTRIBUTED_AVAILABLE)
    // Synchronize delta parameters across processes
    for (auto &layer : model->layers) {
        if (layer->get_layer_type() != LayerType::Activation &&
            layer->get_layer_type() != LayerType::Pool2d) {
            // Synchronize weights and biases deltas
            communicator->all_reduce(layer->delta_mu_w.data(),
                                     layer->delta_mu_w.size());
            communicator->all_reduce(layer->delta_var_w.data(),
                                     layer->delta_var_w.size());

            if (layer->bias) {
                communicator->all_reduce(layer->delta_mu_b.data(),
                                         layer->delta_mu_b.size());
                communicator->all_reduce(layer->delta_var_b.data(),
                                         layer->delta_var_b.size());
            }

            // Scale by world size to get average
            float scale =
                this->average ? 1.0f / communicator->get_world_size() : 1.0f;

            for (auto &val : layer->delta_mu_w) val *= scale;
            for (auto &val : layer->delta_var_w) val *= scale;

            if (layer->bias) {
                for (auto &val : layer->delta_mu_b) val *= scale;
                for (auto &val : layer->delta_var_b) val *= scale;
            }
        }
    }
#endif
}

void DistributedSequential::forward(const std::vector<float> &mu_a,
                                    const std::vector<float> &var_a) {
    model->forward(mu_a, var_a);
}

void DistributedSequential::backward() { model->backward(); }

void DistributedSequential::step() {
#if defined(DISTRIBUTED_AVAILABLE)
    sync_parameters();
    model->step();
    communicator->check_async_error();
#else
    model->step();
#endif
}

void DistributedSequential::train() { model->train(); }

void DistributedSequential::eval() { model->eval(); }

void DistributedSequential::barrier() { communicator->barrier(); }