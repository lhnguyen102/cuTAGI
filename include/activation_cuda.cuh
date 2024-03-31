///////////////////////////////////////////////////////////////////////////////
// File:         activation_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 04, 2023
// Updated:      March 31, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "base_layer_cuda.cuh"
#include "struct_var.h"

__global__ void relu_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a);

__global__ void sigmoid_mean_var_cuda(float const *mu_z, float const *var_z,
                                      int num_states, float *mu_a, float *jcb,
                                      float *var_a);

__global__ void tanh_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a);

__global__ void mixture_relu_mean_var_cuda(float const *mu_z,
                                           float const *var_z, float omega_tol,
                                           int num_states, float *mu_a,
                                           float *jcb, float *var_a);

__global__ void mixture_sigmoid_mean_var_cuda(float const *mu_z,
                                              float const *var_z,
                                              float omega_tol, int num_states,
                                              float *mu_a, float *jcb,
                                              float *var_a);

__global__ void mixture_tanh_mean_var_cuda(float const *mu_z,
                                           float const *var_z, float omega_tol,
                                           int num_states, float *mu_a,
                                           float *jcb, float *var_a);

__global__ void softplus_mean_var_cuda(float const *mu_z, float const *var_z,
                                       int num_states, float *mu_a, float *jcb,
                                       float *var_a);

__global__ void leakyrelu_mean_var_cuda(float const *mu_z, float const *var_z,
                                        float alpha, int num_states,
                                        float *mu_a, float *jcb, float *var_a);

__global__ void softmax_mean_var_cuda(float const *mu_z, float *var_z,
                                      size_t output_size, int batch_size,
                                      float *mu_a, float *jcb, float *var_a);

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

    void allocate_param_delta() override{};

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

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

    void allocate_param_delta() override{};

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

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

    void allocate_param_delta() override{};

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

    std::unique_ptr<BaseLayer> to_host() override;
};

class MixtureReluCuda : public BaseLayerCuda {
   public:
    float omega_tol = 0.0000001f;
    MixtureReluCuda();
    ~MixtureReluCuda();

    // Delete copy constructor and copy assignment
    MixtureReluCuda(const MixtureReluCuda &) = delete;
    MixtureReluCuda &operator=(const MixtureReluCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureReluCuda(MixtureReluCuda &&) = default;
    MixtureReluCuda &operator=(MixtureReluCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void allocate_param_delta() override{};

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

    std::unique_ptr<BaseLayer> to_host() override;
};

class MixtureSigmoidCuda : public BaseLayerCuda {
   public:
    float omega_tol = 0.0000001f;
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

    void allocate_param_delta() override{};

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

    std::unique_ptr<BaseLayer> to_host() override;
};

class MixtureTanhCuda : public BaseLayerCuda {
   public:
    float omega_tol = 0.0000001f;
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

    void allocate_param_delta() override{};

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

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

    void allocate_param_delta() override{};

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

    std::unique_ptr<BaseLayer> to_host() override;
};

class LeakyReluCuda : public BaseLayerCuda {
   public:
    float alpha = 0.1f;
    LeakyReluCuda();
    ~LeakyReluCuda();

    // Delete copy constructor and copy assignment
    LeakyReluCuda(const LeakyReluCuda &) = delete;
    LeakyReluCuda &operator=(const LeakyReluCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    LeakyReluCuda(LeakyReluCuda &&) = default;
    LeakyReluCuda &operator=(LeakyReluCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void allocate_param_delta() override{};

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

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

    void allocate_param_delta() override{};

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

    std::unique_ptr<BaseLayer> to_host() override;
};