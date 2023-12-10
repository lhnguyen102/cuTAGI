///////////////////////////////////////////////////////////////////////////////
// File:         activation_layer_cpu.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      December 10, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <thread>
#include <vector>

#include "base_layer.h"
#include "common.h"
#include "data_struct.h"

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

    void forward(HiddenStateBase &input_states, HiddenStateBase &output_states,
                 TempStateBase &temp_states) override;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
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

    void forward(HiddenStateBase &input_states, HiddenStateBase &output_states,
                 TempStateBase &temp_states) override;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
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

    void forward(HiddenStateBase &input_states, HiddenStateBase &output_states,
                 TempStateBase &temp_states) override;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
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

    void forward(HiddenStateBase &input_states, HiddenStateBase &output_states,
                 TempStateBase &temp_states) override;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
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

    void forward(HiddenStateBase &input_states, HiddenStateBase &output_states,
                 TempStateBase &temp_states) override;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
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

    void forward(HiddenStateBase &input_states, HiddenStateBase &output_states,
                 TempStateBase &temp_states) override;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
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

    void forward(HiddenStateBase &input_states, HiddenStateBase &output_states,
                 TempStateBase &temp_states) override;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
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

    void forward(HiddenStateBase &input_states, HiddenStateBase &output_states,
                 TempStateBase &temp_states) override;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
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

    void forward(HiddenStateBase &input_states, HiddenStateBase &output_states,
                 TempStateBase &temp_states) override;

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
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

    void forward(HiddenStateBase &input_states, HiddenStateBase &output_states,
                 TempStateBase &temp_states) override{};

    void update_weights() override{};

    void update_biases() override{};

    void save(std::ofstream &file) override{};

    void load(std::ifstream &file) override{};
};