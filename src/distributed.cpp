#include "../include/distributed.h"

#include <cuda_runtime.h>

#include <stdexcept>

#include "../include/custom_logger.h"
#ifdef USE_MPI
#include <mpi.h>
#endif

DistributedConfig::DistributedConfig(const std::vector<int> &device_ids,
                                     const std::string &backend, int rank,
                                     int world_size)
    : device_ids(device_ids),
      backend(backend),
      rank(rank),
      world_size(world_size) {}

NCCLCommunicator::NCCLCommunicator(int rank,
                                   const std::vector<int> &device_ids) {
    this->rank = rank;
    this->world_size = device_ids.size();

    // Set device for this process
    cudaSetDevice(device_ids[rank]);

    // Initialize NCCL
    ncclUniqueId id;
    // Rank 0 creates the unique id. It communicates the id to all processes.
#ifdef USE_MPI
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    // All processes get the id, i.e., secret key to communicate with each other
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
#else
    LOG(LogLevel::ERROR, "MPI is required to broadcast the NCCL unique id");
#endif

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

#ifdef USE_MPI
MPICommunicator::MPICommunicator() {
    // Check if MPI is already initialized
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        LOG(LogLevel::ERROR,
            "MPI must be initialized before creating MPICommunicator");
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
}

MPICommunicator::~MPICommunicator() {
    // Main program handle MPI cleanup
}

void MPICommunicator::all_reduce(float *data, size_t count) {
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);
}

void MPICommunicator::barrier() { MPI_Barrier(MPI_COMM_WORLD); }
#endif

DistributedSequential::DistributedSequential(std::shared_ptr<Sequential> model,
                                             const DistributedConfig &config,
                                             bool average)
    : model(model), config(config), average(average) {
    // Create appropriate communicator based on backend. The model that trained
    // on different batches to communicate with each other.
    if (config.backend == "nccl") {
        communicator =
            std::make_unique<NCCLCommunicator>(config.rank, config.device_ids);
    }
#ifdef USE_MPI
    else if (config.backend == "mpi") {
        communicator = std::make_unique<MPICommunicator>();
    }
#endif
    else {
        LOG(LogLevel::ERROR, "Unsupported backend: " + config.backend);
    }

    // Move model to appropriate device
    if (config.backend == "nccl") {
        std::string device =
            "cuda:" + std::to_string(config.device_ids[config.rank]);
        model->to_device(device);
    }
}

void DistributedSequential::sync_parameters() {
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
}

void DistributedSequential::forward(const std::vector<float> &mu_a,
                                    const std::vector<float> &var_a) {
    model->forward(mu_a, var_a);
}

void DistributedSequential::backward() { model->backward(); }

void DistributedSequential::step() {
    sync_parameters();
    model->step();
}