///////////////////////////////////////////////////////////////////////////////
// File:         activation_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 04, 2023
// Updated:      December 04, 2023
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

__global__ void relu_mean_var(float const *mu_z, float const *var_z,
                              int num_states, float *mu_a, float *jcb,
                              float *var_a);

__global__ void sigmoid_mean_var(float const *mu_z, float const *var_z,
                                 int num_states, float *mu_a, float *jcb,
                                 float *var_a);

__global__ void tanh_mean_var(float const *mu_z, float const *var_z,
                              int num_states, float *mu_a, float *jcb,
                              float *var_a);

__global__ void mixture_relu(float const *mu_z, float const *var_z,
                             float omega_tol, int num_states, float *mu_a,
                             float *jcb, float *var_a);

__global__ void mixture_sigmoid(float const *mu_z, float const *var_z,
                                float omega_tol, int num_states, float *mu_a,
                                float *jcb, float *var_a);

__global__ void mixture_tanh(float const *mu_z, float const *var_z,
                             float omega_tol, int num_states, float *mu_a,
                             float *jcb, float *var_a);

__global__ void softplus(float const *mu_z, float const *var_z, int num_states,
                         float *mu_a, float *jcb, float *var_a);

__global__ void leakyrelu(float const *mu_z, float const *var_z, float alpha,
                          int num_states, float *mu_a, float *jcb,
                          float *var_a);

__global__ void softmax(float const *mu_z, float *var_z, size_t output_size,
                        int batch_size, float *mu_a, float *jcb, float *var_a);

////////////////////////////////////////////////////////////////////////////////
/// Relu
////////////////////////////////////////////////////////////////////////////////
class ReluCuda : public BaseLayerCuda {
   public:
    ReluCuda();
    ~ReluCuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(HiddenStateCuda &input_states, HiddenStateCuda &output_states,
                 TempStateCuda &temp_states) override;

    // Overloaded function from base layer
    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    // Cuda base layer
    using BaseLayerCuda::param_backward;
    using BaseLayerCuda::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};

class SigmoidCuda : public BaseLayerCuda {
   public:
    SigmoidCuda();
    ~SigmoidCuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(HiddenStateCuda &input_states, HiddenStateCuda &output_states,
                 TempStateCuda &temp_states) override;

    // Overloaded function from base layer
    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    // Cuda base layer
    using BaseLayerCuda::param_backward;
    using BaseLayerCuda::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};

class TanhCuda : public BaseLayerCuda {
   public:
    TanhCuda();
    ~TanhCuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(HiddenStateCuda &input_states, HiddenStateCuda &output_states,
                 TempStateCuda &temp_states) override;

    // Overloaded function from base layer
    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    // Cuda base layer
    using BaseLayerCuda::param_backward;
    using BaseLayerCuda::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};

class MixtureReluCuda : public BaseLayerCuda {
   public:
    float omega_tol = 0.0000001f;
    MixtureReluCuda();
    ~MixtureReluCuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(HiddenStateCuda &input_states, HiddenStateCuda &output_states,
                 TempStateCuda &temp_states) override;

    // Overloaded function from base layer
    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    // Cuda base layer
    using BaseLayerCuda::param_backward;
    using BaseLayerCuda::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};

class MixtureSigmoidCuda : public BaseLayerCuda {
   public:
    float omega_tol = 0.0000001f;
    MixtureSigmoidCuda();
    ~MixtureSigmoidCuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(HiddenStateCuda &input_states, HiddenStateCuda &output_states,
                 TempStateCuda &temp_states) override;

    // Overloaded function from base layer
    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    // Cuda base layer
    using BaseLayerCuda::param_backward;
    using BaseLayerCuda::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};

class MixtureTanhCuda : public BaseLayerCuda {
   public:
    float omega_tol = 0.0000001f;
    MixtureTanhCuda();
    ~MixtureTanhCuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(HiddenStateCuda &input_states, HiddenStateCuda &output_states,
                 TempStateCuda &temp_states) override;

    // Overloaded function from base layer
    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    // Cuda base layer
    using BaseLayerCuda::param_backward;
    using BaseLayerCuda::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};

class SoftplusCuda : public BaseLayerCuda {
   public:
    SoftplusCuda();
    ~SoftplusCuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(HiddenStateCuda &input_states, HiddenStateCuda &output_states,
                 TempStateCuda &temp_states) override;

    // Overloaded function from base layer
    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    // Cuda base layer
    using BaseLayerCuda::param_backward;
    using BaseLayerCuda::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};

class LeakyReluCuda : public BaseLayerCuda {
   public:
    float alpha = 0.1f;
    LeakyReluCuda();
    ~LeakyReluCuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(HiddenStateCuda &input_states, HiddenStateCuda &output_states,
                 TempStateCuda &temp_states) override;

    // Overloaded function from base layer
    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    // Cuda base layer
    using BaseLayerCuda::param_backward;
    using BaseLayerCuda::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};

class SoftmaxCuda : public BaseLayerCuda {
   public:
    SoftmaxCuda();
    ~SoftmaxCuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(HiddenStateCuda &input_states, HiddenStateCuda &output_states,
                 TempStateCuda &temp_states) override;

    // Overloaded function from base layer
    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    // Cuda base layer
    using BaseLayerCuda::param_backward;
    using BaseLayerCuda::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};