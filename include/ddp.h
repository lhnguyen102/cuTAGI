#pragma once

// Define a macro to check if all required dependencies are available
#if defined(USE_NCCL) && defined(USE_CUDA) && defined(USE_MPI)
#define DISTRIBUTED_AVAILABLE 1
#include <nccl.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "sequential.h"

// Forward declarations
class DDPConfig;
class Communicator;
class NCCLCommunicator;
class MPICommunicator;

// Abstract communicator interface for different backends (NCCL, MPI, etc.)
class Communicator {
   public:
    virtual ~Communicator() = default;
    virtual void all_reduce(float *data, size_t count,
                            bool average = false) = 0;
    virtual void barrier() = 0;
    virtual int get_rank() const = 0;
    virtual int get_world_size() const = 0;
    virtual void check_async_error() = 0;
    virtual void broadcast(float *data, size_t count, int root) = 0;
};

#ifdef DISTRIBUTED_AVAILABLE
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

    void all_reduce(float *data, size_t count, bool average = false) override;
    void barrier() override;
    int get_rank() const override { return rank; }
    int get_world_size() const override { return world_size; }
    void check_async_error() override;
    void broadcast(float *data, size_t count, int root) override;

    // Add methods to get NCCL communicator and stream
    ncclComm_t get_comm() const { return comm; }
    cudaStream_t get_stream() const { return stream; }
};
#else
// Stub implementation when NCCL is not available
class NCCLCommunicator : public Communicator {
   private:
    int rank;
    int world_size;

   public:
    NCCLCommunicator(int rank, const std::vector<int> &device_ids)
        : rank(rank), world_size(1) {}
    ~NCCLCommunicator() {}

    void all_reduce(float *data, size_t count, bool average = false) override {}
    void barrier() override {}
    int get_rank() const override { return rank; }
    int get_world_size() const override { return world_size; }
    void check_async_error() override {}
    void broadcast(float *data, size_t count, int root) override {}
};
#endif

// Configuration for distributed training
class DDPConfig {
   public:
    std::vector<int> device_ids;
    std::string backend = "nccl";
    int rank = 0;
    int world_size = 1;

    DDPConfig(const std::vector<int> &device_ids,
              const std::string &backend = "nccl", int rank = 0,
              int world_size = 1);
};

// Main DDP wrapper class
class DDPSequential {
   private:
    std::shared_ptr<Sequential> model;
    std::unique_ptr<Communicator> communicator;
    DDPConfig config;
    bool average = true;

   public:
    DDPSequential(std::shared_ptr<Sequential> model, const DDPConfig &config,
                  bool average = true);

    void forward(const std::vector<float> &mu_a,
                 const std::vector<float> &var_a = std::vector<float>());

    void forward(BaseHiddenStates &input_states) {
        this->model->forward(input_states);
    }

    // Python Wrapper
    void forward_py(
        pybind11::array_t<float> mu_a_np,
        pybind11::array_t<float> var_a_np = pybind11::array_t<float>()) {
        this->model->forward_py(mu_a_np, var_a_np);
    }
    void backward();
    void step();
    void train();
    void eval();
    void barrier();
    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    get_outputs() {
        return this->model->get_outputs();
    }

    void output_to_host() { this->model->output_to_host(); }

    BaseHiddenStates *output_z_buffer() {
        return this->model->output_z_buffer.get();
    }
    BaseDeltaStates *input_delta_z_buffer() {
        return this->model->input_delta_z_buffer.get();
    }

    std::string get_device_with_index() const {
        return this->model->get_device_with_index();
    }
    void sync_parameters();
    void sync_base_parameters();

    std::shared_ptr<Sequential> get_model() { return model; }
    const DDPConfig &get_config() const { return config; }
};