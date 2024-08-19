///////////////////////////////////////////////////////////////////////////////
// File:         activation.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      August 19, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <memory>
#include <thread>
#include <vector>

#include "base_layer.h"
#include "common.h"
#include "data_struct.h"

void relu_mean_var(std::vector<float> const &mu_z,
                   std::vector<float> const &var_z, int start_chunk,
                   int end_chunk, std::vector<float> &mu_a,
                   std::vector<float> &jcb, std::vector<float> &var_a);

void relu_mean_var_mp(std::vector<float> const &mu_z,
                      std::vector<float> const &var_z, int n,
                      unsigned int num_threads, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a);

void sigmoid_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int start_chunk, int end_chunk, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a);

void sigmoid_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                         int n, unsigned int num_threads,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
                         std::vector<float> &var_a);

void tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                   int start_chunk, int end_chunk, std::vector<float> &mu_a,
                   std::vector<float> &jcb, std::vector<float> &var_a);

void tanh_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int n, unsigned int num_threads, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a);

void mixture_relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
                           std::vector<float> &var_a);

void mixture_relu_mean_var_mp(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int n,
                              unsigned int num_threads,
                              std::vector<float> &mu_a, std::vector<float> &jcb,
                              std::vector<float> &var_a);

void mixture_sigmoid_mean_var(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int start_chunk,
                              int end_chunk, std::vector<float> &mu_a,
                              std::vector<float> &jcb,
                              std::vector<float> &var_a);

void mixture_sigmoid_mean_var_mp(std::vector<float> &mu_z,
                                 std::vector<float> &var_z, int n,
                                 unsigned int num_threads,
                                 std::vector<float> &mu_a,
                                 std::vector<float> &jcb,
                                 std::vector<float> &var_a);

void mixture_tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
                           std::vector<float> &var_a);

void mixture_tanh_mean_var_mp(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int n,
                              unsigned int num_threads,
                              std::vector<float> &mu_a, std::vector<float> &jcb,
                              std::vector<float> &var_a);

void softplus_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                       int start_chunk, int end_chunk, std::vector<float> &mu_a,
                       std::vector<float> &jcb, std::vector<float> &var_a);

void softplus_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                          int n, unsigned int num_threads,
                          std::vector<float> &mu_a, std::vector<float> &jcb,
                          std::vector<float> &var_a);

void leaky_relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                         float alpha, int start_chunk, int end_chunk,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
                         std::vector<float> &var_a);

void leaky_relu_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                            float alpha, int n, unsigned int num_threads,
                            std::vector<float> &mu_a, std::vector<float> &jcb,
                            std::vector<float> &var_a);

void softmax_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int no, int batch_size, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a);

void even_exp_mean_var(std::vector<float> const &mu_z,
                       std::vector<float> const &var_z,
                       std::vector<float> &jcb_z, int start_chunk,
                       int end_chunk, std::vector<float> &mu_a,
                       std::vector<float> &var_a, std::vector<float> &jcb_a);

void even_exp_mean_var_mp(std::vector<float> const &mu_z,
                          std::vector<float> const &var_z,
                          std::vector<float> const &jcb_z, int n,
                          unsigned int num_threads, std::vector<float> &mu_a,
                          std::vector<float> &var_a, std::vector<float> &jcb_a);

////////////////////////////////////////////////////////////////////////////////
/// ReLU
////////////////////////////////////////////////////////////////////////////////
class ReLU : public BaseLayer {
   public:
    ReLU();
    ~ReLU();

    // Delete copy constructor and copy assignment
    ReLU(const ReLU &) = delete;
    ReLU &operator=(const ReLU &) = delete;

    // Optionally implement move constructor and move assignment
    ReLU(ReLU &&) = default;
    ReLU &operator=(ReLU &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
class Sigmoid : public BaseLayer {
   public:
    Sigmoid();
    ~Sigmoid();

    // Delete copy constructor and copy assignment
    Sigmoid(const Sigmoid &) = delete;
    Sigmoid &operator=(const Sigmoid &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    Sigmoid(Sigmoid &&) = default;
    Sigmoid &operator=(Sigmoid &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
class Tanh : public BaseLayer {
   public:
    Tanh();
    ~Tanh();

    // Delete copy constructor and copy assignment
    Tanh(const Tanh &) = delete;
    Tanh &operator=(const Tanh &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    Tanh(Tanh &&) = default;
    Tanh &operator=(Tanh &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void allocate_param_delta() override {};

    using BaseLayer::backward;

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture ReLU
////////////////////////////////////////////////////////////////////////////////
class MixtureReLU : public BaseLayer {
   public:
    MixtureReLU();
    ~MixtureReLU();

    // Delete copy constructor and copy assignment
    MixtureReLU(const MixtureReLU &) = delete;
    MixtureReLU &operator=(const MixtureReLU &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureReLU(MixtureReLU &&) = default;
    MixtureReLU &operator=(MixtureReLU &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
class MixtureSigmoid : public BaseLayer {
   public:
    MixtureSigmoid();
    ~MixtureSigmoid();

    // Delete copy constructor and copy assignment
    MixtureSigmoid(const MixtureSigmoid &) = delete;
    MixtureSigmoid &operator=(const MixtureSigmoid &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureSigmoid(MixtureSigmoid &&) = default;
    MixtureSigmoid &operator=(MixtureSigmoid &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
class MixtureTanh : public BaseLayer {
   public:
    MixtureTanh();
    ~MixtureTanh();

    // Delete copy constructor and copy assignment
    MixtureTanh(const MixtureTanh &) = delete;
    MixtureTanh &operator=(const MixtureTanh &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureTanh(MixtureTanh &&) = default;
    MixtureTanh &operator=(MixtureTanh &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
class Softplus : public BaseLayer {
   public:
    Softplus();
    ~Softplus();

    // Delete copy constructor and copy assignment
    Softplus(const Softplus &) = delete;
    Softplus &operator=(const Softplus &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    Softplus(Softplus &&) = default;
    Softplus &operator=(Softplus &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Leaky ReLU
////////////////////////////////////////////////////////////////////////////////
class LeakyReLU : public BaseLayer {
   public:
    float alpha = 0.1f;
    LeakyReLU();
    ~LeakyReLU();

    // Delete copy constructor and copy assignment
    LeakyReLU(const LeakyReLU &) = delete;
    LeakyReLU &operator=(const LeakyReLU &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    LeakyReLU(LeakyReLU &&) = default;
    LeakyReLU &operator=(LeakyReLU &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
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

    // Delete copy constructor and copy assignment
    Softmax(const Softmax &) = delete;
    Softmax &operator=(const Softmax &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    Softmax(Softmax &&) = default;
    Softmax &operator=(Softmax &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
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
                 BaseTempStates &temp_states) override {};

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};
};

////////////////////////////////////////////////////////////////////////////////
/// EvenExp
////////////////////////////////////////////////////////////////////////////////
class EvenExp : public BaseLayer {
   public:
    EvenExp();
    ~EvenExp();

    // Delete copy constructor and copy assignment
    EvenExp(const EvenExp &) = delete;
    EvenExp &operator=(const EvenExp &) = delete;

    // Optionally implement move constructor and move assignment
    EvenExp(EvenExp &&) = default;
    EvenExp &operator=(EvenExp &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};
