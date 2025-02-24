#include "../include/distributed.h"

#include <cuda_runtime.h>

#include <stdexcept>
#ifdef USE_MPI
#include <mpi.h>
#endif

DistributedConfig::DistributedConfig(const std::vector<int>& device_ids,
                                     const std::string& backend, int rank,
                                     int world_size)
    : device_ids(device_ids),
      backend(backend),
      rank(rank),
      world_size(world_size) {}

NCCLCommunicator::NCCLCommunicator(int rank,
                                   const std::vector<int>& device_ids) {
    this->rank = rank;
    this->world_size = device_ids.size();

    // Set device for this process
    cudaSetDevice(device_ids[rank]);

    // Initialize NCCL
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    // Broadcast id to all processes using MPI
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL communicator
    ncclCommInitRank(&comm, world_size, id, rank);

    // Create CUDA stream
    cudaStreamCreate(&stream);
}

NCCLCommunicator::~NCCLCommunicator() {
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);
}

void NCCLCommunicator::all_reduce(float* data, size_t count) {
    ncclAllReduce(data, data, count, ncclFloat32, ncclSum, comm, stream);
    cudaStreamSynchronize(stream);
}

void NCCLCommunicator::barrier() { cudaStreamSynchronize(stream); }

#ifdef USE_MPI
MPICommunicator::MPICommunicator(int* argc, char*** argv) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
}

MPICommunicator::~MPICommunicator() { MPI_Finalize(); }

void MPICommunicator::all_reduce(float* data, size_t count) {
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);
}

void MPICommunicator::barrier() { MPI_Barrier(MPI_COMM_WORLD); }
#endif

DistributedSequential::DistributedSequential(std::shared_ptr<Sequential> model,
                                             const DistributedConfig& config)
    : model(model), config(config) {
    // Create appropriate communicator based on backend
    if (config.backend == "nccl") {
        communicator =
            std::make_unique<NCCLCommunicator>(config.rank, config.device_ids);
    }
#ifdef USE_MPI
    else if (config.backend == "mpi") {
        // Note: argc and argv need to be passed from main
        communicator = std::make_unique<MPICommunicator>(&argc, &argv);
    }
#endif
    else {
        throw std::runtime_error("Unsupported backend: " + config.backend);
    }

    // Move model to appropriate device
    if (config.backend == "nccl") {
        model->to_device("cuda");
        cudaSetDevice(config.device_ids[config.rank]);
    }
}

void DistributedSequential::sync_parameters() {
    // Synchronize delta parameters across processes
    for (auto& layer : model->layers) {
        if (layer->get_layer_type() != LayerType::Activation &&
            layer->get_layer_type() != LayerType::Pool2d) {
            // Synchronize weight deltas
            communicator->all_reduce(layer->delta_mu_w.data(),
                                     layer->delta_mu_w.size());
            communicator->all_reduce(layer->delta_var_w.data(),
                                     layer->delta_var_w.size());

            // Synchronize bias deltas if the layer uses bias
            if (layer->bias) {
                communicator->all_reduce(layer->delta_mu_b.data(),
                                         layer->delta_mu_b.size());
                communicator->all_reduce(layer->delta_var_b.data(),
                                         layer->delta_var_b.size());
            }

            // Scale by world size to get average
            float scale = 1.0f / communicator->get_world_size();

            // Scale deltas
            for (auto& val : layer->delta_mu_w) val *= scale;
            for (auto& val : layer->delta_var_w) val *= scale;

            if (layer->bias) {
                for (auto& val : layer->delta_mu_b) val *= scale;
                for (auto& val : layer->delta_var_b) val *= scale;
            }
        }
    }
}

void DistributedSequential::forward(const std::vector<float>& mu_a,
                                    const std::vector<float>& var_a) {
    model->forward(mu_a, var_a);
}

void DistributedSequential::backward() { model->backward(); }

void DistributedSequential::step() {
    // Synchronize delta parameters before updating
    sync_parameters();
    model->step();
}