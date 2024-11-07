#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "base_layer_cuda.cuh"

__global__ void relu_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a);

__global__ void relu_mean_var_cuda_vectorized(float const *mu_z,
                                              float const *var_z,
                                              int num_states, float *mu_a,
                                              float *jcb, float *var_a);

__global__ void sigmoid_mean_var_cuda(float const *mu_z, float const *var_z,
                                      int num_states, float *mu_a, float *jcb,
                                      float *var_a);

__global__ void tanh_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a);

__global__ void mixture_relu_mean_var_cuda(float const *mu_z,
                                           float const *var_z, int num_states,
                                           float *mu_a, float *jcb,
                                           float *var_a);

__global__ void mixture_sigmoid_mean_var_cuda(float const *mu_z,
                                              float const *var_z,
                                              int num_states, float *mu_a,
                                              float *jcb, float *var_a);

__global__ void mixture_tanh_mean_var_cuda(float const *mu_z,
                                           float const *var_z, int num_states,
                                           float *mu_a, float *jcb,
                                           float *var_a);

__global__ void softplus_mean_var_cuda(float const *mu_z, float const *var_z,
                                       int num_states, float *mu_a, float *jcb,
                                       float *var_a);

__global__ void leakyrelu_mean_var_cuda(float const *mu_z, float const *var_z,
                                        float alpha, int num_states,
                                        float *mu_a, float *jcb, float *var_a);

__global__ void softmax_mean_var_cuda(float const *mu_z, float *var_z,
                                      size_t output_size, int batch_size,
                                      float *mu_a, float *jcb, float *var_a);

__global__ void even_exp_mean_var_cuda(float const *mu_z, float const *var_z,
                                       float const *jcb_z, int num_states,
                                       float *mu_a, float *var_a, float *jcb_a);

////////////////////////////////////////////////////////////////////////////////
/// Relu
////////////////////////////////////////////////////////////////////////////////
class ReLUCuda : public BaseLayerCuda {
   public:
    ReLUCuda();
    ~ReLUCuda();

    // Delete copy constructor and copy assignment
    ReLUCuda(const ReLUCuda &) = delete;
    ReLUCuda &operator=(const ReLUCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    ReLUCuda(ReLUCuda &&) = default;
    ReLUCuda &operator=(ReLUCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class SigmoidCuda : public BaseLayerCuda {
   public:
    SigmoidCuda();
    ~SigmoidCuda();

    // Delete copy constructor and copy assignment
    SigmoidCuda(const SigmoidCuda &) = delete;
    SigmoidCuda &operator=(const SigmoidCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    SigmoidCuda(SigmoidCuda &&) = default;
    SigmoidCuda &operator=(SigmoidCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class TanhCuda : public BaseLayerCuda {
   public:
    TanhCuda();

    ~TanhCuda();

    // Delete copy constructor and copy assignment
    TanhCuda(const TanhCuda &) = delete;
    TanhCuda &operator=(const TanhCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    TanhCuda(TanhCuda &&) = default;
    TanhCuda &operator=(TanhCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class MixtureReLUCuda : public BaseLayerCuda {
   public:
    MixtureReLUCuda();
    ~MixtureReLUCuda();

    // Delete copy constructor and copy assignment
    MixtureReLUCuda(const MixtureReLUCuda &) = delete;
    MixtureReLUCuda &operator=(const MixtureReLUCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureReLUCuda(MixtureReLUCuda &&) = default;
    MixtureReLUCuda &operator=(MixtureReLUCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class MixtureSigmoidCuda : public BaseLayerCuda {
   public:
    MixtureSigmoidCuda();
    ~MixtureSigmoidCuda();

    // Delete copy constructor and copy assignment
    MixtureSigmoidCuda(const MixtureSigmoidCuda &) = delete;
    MixtureSigmoidCuda &operator=(const MixtureSigmoidCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureSigmoidCuda(MixtureSigmoidCuda &&) = default;
    MixtureSigmoidCuda &operator=(MixtureSigmoidCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class MixtureTanhCuda : public BaseLayerCuda {
   public:
    MixtureTanhCuda();
    ~MixtureTanhCuda();

    // Delete copy constructor and copy assignment
    MixtureTanhCuda(const MixtureTanhCuda &) = delete;
    MixtureTanhCuda &operator=(const MixtureTanhCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureTanhCuda(MixtureTanhCuda &&) = default;
    MixtureTanhCuda &operator=(MixtureTanhCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class SoftplusCuda : public BaseLayerCuda {
   public:
    SoftplusCuda();
    ~SoftplusCuda();

    // Delete copy constructor and copy assignment
    SoftplusCuda(const SoftplusCuda &) = delete;
    SoftplusCuda &operator=(const SoftplusCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    SoftplusCuda(SoftplusCuda &&) = default;
    SoftplusCuda &operator=(SoftplusCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class LeakyReLUCuda : public BaseLayerCuda {
   public:
    float alpha = 0.1f;
    LeakyReLUCuda();
    ~LeakyReLUCuda();

    // Delete copy constructor and copy assignment
    LeakyReLUCuda(const LeakyReLUCuda &) = delete;
    LeakyReLUCuda &operator=(const LeakyReLUCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    LeakyReLUCuda(LeakyReLUCuda &&) = default;
    LeakyReLUCuda &operator=(LeakyReLUCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class SoftmaxCuda : public BaseLayerCuda {
   public:
    SoftmaxCuda();
    ~SoftmaxCuda();

    // Delete copy constructor and copy assignment
    SoftmaxCuda(const SoftmaxCuda &) = delete;
    SoftmaxCuda &operator=(const SoftmaxCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    SoftmaxCuda(SoftmaxCuda &&) = default;
    SoftmaxCuda &operator=(SoftmaxCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class EvenExpCuda : public BaseLayerCuda {
   public:
    EvenExpCuda();
    ~EvenExpCuda();

    unsigned int num_cuda_threads = 16;

    // Delete copy constructor and copy assignment
    EvenExpCuda(const EvenExpCuda &) = delete;
    EvenExpCuda &operator=(const EvenExpCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    EvenExpCuda(EvenExpCuda &&) = default;
    EvenExpCuda &operator=(EvenExpCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};