#pragma once

#include <nccl.h>

#include <memory>
#include <vector>

#include "sequential.h"

// Forward declarations
class DistributedConfig;
class Communicator;
class NCCLCommunicator;
class MPICommunicator;

// Abstract communicator interface for different backends (NCCL, MPI, etc.)
class Communicator {
   public:
    virtual ~Communicator() = default;
    virtual void all_reduce(float *data, size_t count) = 0;
    virtual void barrier() = 0;
    virtual int get_rank() const = 0;
    virtual int get_world_size() const = 0;
};

// NCCL-specific communicator implementation
class NCCLCommunicator : public Communicator {
   private:
    ncclComm_t comm;
    cudaStream_t stream;
    int rank;
    int world_size;

   public:
    NCCLCommunicator(int rank, const std::vector<int> &device_ids);
    ~NCCLCommunicator();

    void all_reduce(float *data, size_t count) override;
    void barrier() override;
    int get_rank() const override { return rank; }
    int get_world_size() const override { return world_size; }
};

// MPI-specific communicator implementation
class MPICommunicator : public Communicator {
   private:
    int rank;
    int world_size;

   public:
    MPICommunicator();
    ~MPICommunicator();

    void all_reduce(float *data, size_t count) override;
    void barrier() override;
    int get_rank() const override { return rank; }
    int get_world_size() const override { return world_size; }
};

// Configuration for distributed training
class DistributedConfig {
   public:
    std::vector<int> device_ids;
    std::string backend = "nccl";
    int rank = 0;
    int world_size = 1;

    DistributedConfig(const std::vector<int> &device_ids,
                      const std::string &backend = "nccl", int rank = 0,
                      int world_size = 1);
};

// Main DDP wrapper class
class DistributedSequential {
   private:
    std::shared_ptr<Sequential> model;
    std::unique_ptr<Communicator> communicator;
    DistributedConfig config;

    void sync_parameters();

   public:
    DistributedSequential(std::shared_ptr<Sequential> model,
                          const DistributedConfig &config);

    void forward(const std::vector<float> &mu_a,
                 const std::vector<float> &var_a = std::vector<float>());
    void backward();
    void step();

    std::shared_ptr<Sequential> get_model() { return model; }
    const DistributedConfig &get_config() const { return config; }
};