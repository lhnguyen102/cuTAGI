///////////////////////////////////////////////////////////////////////////////
// File:         activation_layer_cpu.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      December 16, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <thread>
#include <vector>

#include "base_layer.h"
#include "common.h"
#include "data_struct.h"
#ifdef USE_CUDA
#include "activation_cuda.cuh"
#endif

////////////////////////////////////////////////////////////////////////////////
/// Relu
////////////////////////////////////////////////////////////////////////////////
class Relu : public BaseLayer {
   public:
    Relu();
    ~Relu();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    static void relu_mean_var(std::vector<float> const &mu_z,
                              std::vector<float> const &var_z, int start_chunk,
                              int end_chunk, std::vector<float> &mu_a,
                              std::vector<float> &jcb,
                              std::vector<float> &var_a);

    static void relu_mean_var_mp(std::vector<float> const &mu_z,
                                 std::vector<float> const &var_z, int n,
                                 unsigned int num_threads,
                                 std::vector<float> &mu_a,
                                 std::vector<float> &jcb,
                                 std::vector<float> &var_a);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

#ifdef USE_CUDA
    ReluCuda to_cuda();
#else
    void to_device() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
class Sigmoid : public BaseLayer {
   public:
    Sigmoid();
    ~Sigmoid();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    static void sigmoid_mean_var(std::vector<float> &mu_z,
                                 std::vector<float> &var_z, int start_chunk,
                                 int end_chunk, std::vector<float> &mu_a,
                                 std::vector<float> &jcb,
                                 std::vector<float> &var_a);

    static void sigmoid_mean_var_mp(std::vector<float> &mu_z,
                                    std::vector<float> &var_z, int n,
                                    unsigned int num_threads,
                                    std::vector<float> &mu_a,
                                    std::vector<float> &jcb,
                                    std::vector<float> &var_a);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

#ifdef USE_CUDA
    SigmoidCuda to_cuda();
#else
    void to_device() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
class Tanh : public BaseLayer {
   public:
    Tanh();
    ~Tanh();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    static void tanh_mean_var(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int start_chunk,
                              int end_chunk, std::vector<float> &mu_a,
                              std::vector<float> &jcb,
                              std::vector<float> &var_a);

    void tanh_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                          int n, unsigned int num_threads,
                          std::vector<float> &mu_a, std::vector<float> &jcb,
                          std::vector<float> &var_a);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

#ifdef USE_CUDA
    TanhCuda to_cuda();
#else
    void to_device() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Relu
////////////////////////////////////////////////////////////////////////////////
class MixtureRelu : public BaseLayer {
   public:
    float omega_tol = 0.0000001f;
    MixtureRelu();
    ~MixtureRelu();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    static void mixture_relu_mean_var(std::vector<float> &mu_z,
                                      std::vector<float> &var_z,
                                      float omega_tol, int start_chunk,
                                      int end_chunk, std::vector<float> &mu_a,
                                      std::vector<float> &jcb,
                                      std::vector<float> &var_a);
    static void mixture_relu_mean_var_mp(
        std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol,
        int n, unsigned int num_threads, std::vector<float> &mu_a,
        std::vector<float> &jcb, std::vector<float> &var_a);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

#ifdef USE_CUDA
    MixtureReluCuda to_cuda();
#else
    void to_device() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
class MixtureSigmoid : public BaseLayer {
   public:
    float omega_tol = 0.0000001f;
    MixtureSigmoid();
    ~MixtureSigmoid();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    static void mixture_sigmoid_mean_var(
        std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol,
        int start_chunk, int end_chunk, std::vector<float> &mu_a,
        std::vector<float> &jcb, std::vector<float> &var_a);

    static void mixture_sigmoid_mean_var_mp(
        std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol,
        int n, unsigned int num_threads, std::vector<float> &mu_a,
        std::vector<float> &jcb, std::vector<float> &var_a);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

#ifdef USE_CUDA
    MixtureSigmoid to_cuda();
#else
    void to_device() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
class MixtureTanh : public BaseLayer {
   public:
    float omega_tol = 0.0000001f;
    MixtureTanh();
    ~MixtureTanh();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    static void mixture_tanh_mean_var(std::vector<float> &mu_z,
                                      std::vector<float> &var_z,
                                      float omega_tol, int start_chunk,
                                      int end_chunk, std::vector<float> &mu_a,
                                      std::vector<float> &jcb,
                                      std::vector<float> &var_a);

    static void mixture_tanh_mean_var_mp(
        std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol,
        int n, unsigned int num_threads, std::vector<float> &mu_a,
        std::vector<float> &jcb, std::vector<float> &var_a);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

#ifdef USE_CUDA
    MixtureTanhCuda to_cuda();
#else
    void to_device() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
class Softplus : public BaseLayer {
   public:
    Softplus();
    ~Softplus();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    static void softplus_mean_var(std::vector<float> &mu_z,
                                  std::vector<float> &var_z, int start_chunk,
                                  int end_chunk, std::vector<float> &mu_a,
                                  std::vector<float> &jcb,
                                  std::vector<float> &var_a);

    static void softplus_mean_var_mp(std::vector<float> &mu_z,
                                     std::vector<float> &var_z, int n,
                                     unsigned int num_threads,
                                     std::vector<float> &mu_a,
                                     std::vector<float> &jcb,
                                     std::vector<float> &var_a);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

#ifdef USE_CUDA
    SoftplusCuda to_cuda();
#else
    void to_device() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Leaky ReLU
////////////////////////////////////////////////////////////////////////////////
class LeakyRelu : public BaseLayer {
   public:
    float alpha = 0.1f;
    LeakyRelu();
    ~LeakyRelu();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    static void leaky_relu_mean_var(std::vector<float> &mu_z,
                                    std::vector<float> &var_z, float alpha,
                                    int start_chunk, int end_chunk,
                                    std::vector<float> &mu_a,
                                    std::vector<float> &jcb,
                                    std::vector<float> &var_a);

    static void leaky_relu_mean_var_mp(std::vector<float> &mu_z,
                                       std::vector<float> &var_z, float alpha,
                                       int n, unsigned int num_threads,
                                       std::vector<float> &mu_a,
                                       std::vector<float> &jcb,
                                       std::vector<float> &var_a);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

#ifdef USE_CUDA
    LeakyReluCuda to_cuda();
#else
    void to_device() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Stable Softmax
////////////////////////////////////////////////////////////////////////////////
class Softmax : public BaseLayer {
   public:
    float alpha = 0.1f;
    Softmax();
    ~Softmax();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    static void softmax_mean_var(std::vector<float> &mu_z,
                                 std::vector<float> &var_z, int no,
                                 int batch_size, std::vector<float> &mu_a,
                                 std::vector<float> &jcb,
                                 std::vector<float> &var_a);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};

#ifdef USE_CUDA
    SoftmaxCuda to_cuda();
#else
    void to_device() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Remax
////////////////////////////////////////////////////////////////////////////////
class RemaxA : public BaseLayer {
   public:
    float alpha = 0.1f;
    RemaxA();
    ~RemaxA();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    // TODO: How to add mixture relu
    void to_log(std::vector<float> &mu_m, std::vector<float> &var_m, int no,
                int B, std::vector<float> &mu_log, std::vector<float> &var_log);

    void sum_class_hidden_states(std::vector<float> &mu_m,
                                 std::vector<float> &var_m, int no, int B,
                                 std::vector<float> &mu_sum,
                                 std::vector<float> &var_sum);

    void compute_cov_log_logsum(std::vector<float> &mu_m,
                                std::vector<float> &var_m,
                                std::vector<float> &mu_sum, int no, int B,
                                std::vector<float> &cov_log_logsum);

    void compute_remax_prob(std::vector<float> &mu_log,
                            std::vector<float> &var_log,
                            std::vector<float> &mu_logsum,
                            std::vector<float> &var_logsum,
                            std::vector<float> &cov_log_logsum, int no, int B,
                            std::vector<float> &mu_a,
                            std::vector<float> &var_a);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override{};

    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};