///////////////////////////////////////////////////////////////////////////////
// File:         activation_layer_cpu.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      November 15, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <thread>
#include <vector>

#include "base_layer.h"
#include "common.h"
////////////////////////////////////////////////////////////////////////////////
/// Relu
////////////////////////////////////////////////////////////////////////////////
class Relu : public BaseLayer {
   public:
    Relu();
    ~Relu();

    static void relu_mean_var(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int start_chunk,
                              int end_chunk, std::vector<float> &mu_a,
                              std::vector<float> &jcb,
                              std::vector<float> &var_a);

    static void relu_mean_var_mp(std::vector<float> &mu_z,
                                 std::vector<float> &var_z, int n,
                                 unsigned int num_threads,
                                 std::vector<float> &mu_a,
                                 std::vector<float> &jcb,
                                 std::vector<float> &var_a);

    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override;
    //  void state_backward(std::vector<float> &jcb,
    //                      DeltaStates &input_delta_states,
    //                      DeltaStates &output_hidden_states,
    //                      TempStates &temp_states) override{};
    //  void param_backward(std::vector<float> &mu_a, DeltaStates &delta_states,
    //                      TempStates &temp_states) override{};
};

////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
class Sigmoid : public BaseLayer {
   public:
    Sigmoid();
    ~Sigmoid();
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

    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override;
    void state_backward(std::vector<float> &jcb,
                        DeltaStates &input_delta_states,
                        DeltaStates &output_hidden_states,
                        TempStates &temp_states) override{};
    void param_backward(std::vector<float> &mu_a, DeltaStates &delta_states,
                        TempStates &temp_states) override{};
};
////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
class Tanh : public BaseLayer {
   public:
    Tanh();
    ~Tanh();
    static void tanh_mean_var(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int start_chunk,
                              int end_chunk, std::vector<float> &mu_a,
                              std::vector<float> &jcb,
                              std::vector<float> &var_a);

    void tanh_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                          int n, unsigned int num_threads,
                          std::vector<float> &mu_a, std::vector<float> &jcb,
                          std::vector<float> &var_a);

    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override;
    void state_backward(std::vector<float> &jcb,
                        DeltaStates &input_delta_states,
                        DeltaStates &output_hidden_states,
                        TempStates &temp_states) override{};
    void param_backward(std::vector<float> &mu_a, DeltaStates &delta_states,
                        TempStates &temp_states) override{};
};
////////////////////////////////////////////////////////////////////////////////
/// Mixture Relu
////////////////////////////////////////////////////////////////////////////////
class MixtureRelu : public BaseLayer {
   public:
    float omega_tol = 0.0000001f;
    MixtureRelu();
    ~MixtureRelu();
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

    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override;
    void state_backward(std::vector<float> &jcb,
                        DeltaStates &input_delta_states,
                        DeltaStates &output_hidden_states,
                        TempStates &temp_states) override{};
    void param_backward(std::vector<float> &mu_a, DeltaStates &delta_states,
                        TempStates &temp_states) override{};
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
class MixtureSigmoid : public BaseLayer {
   public:
    float omega_tol = 0.0000001f;
    MixtureSigmoid();
    ~MixtureSigmoid();
    static void mixture_sigmoid_mean_var(
        std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol,
        int start_chunk, int end_chunk, std::vector<float> &mu_a,
        std::vector<float> &jcb, std::vector<float> &var_a);

    static void mixture_sigmoid_mean_var_mp(
        std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol,
        int n, unsigned int num_threads, std::vector<float> &mu_a,
        std::vector<float> &jcb, std::vector<float> &var_a);

    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override;
    void state_backward(std::vector<float> &jcb,
                        DeltaStates &input_delta_states,
                        DeltaStates &output_hidden_states,
                        TempStates &temp_states) override{};
    void param_backward(std::vector<float> &mu_a, DeltaStates &delta_states,
                        TempStates &temp_states) override{};
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
class MixtureTanh : public BaseLayer {
   public:
    float omega_tol = 0.0000001f;
    MixtureTanh();
    ~MixtureTanh();
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

    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override;
    void state_backward(std::vector<float> &jcb,
                        DeltaStates &input_delta_states,
                        DeltaStates &output_hidden_states,
                        TempStates &temp_states) override{};
    void param_backward(std::vector<float> &mu_a, DeltaStates &delta_states,
                        TempStates &temp_states) override{};
};
////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
class Softplus : public BaseLayer {
   public:
    Softplus();
    ~Softplus();
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

    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override;
    void state_backward(std::vector<float> &jcb,
                        DeltaStates &input_delta_states,
                        DeltaStates &output_hidden_states,
                        TempStates &temp_states) override{};
    void param_backward(std::vector<float> &mu_a, DeltaStates &delta_states,
                        TempStates &temp_states) override{};
};
////////////////////////////////////////////////////////////////////////////////
/// Leaky ReLU
////////////////////////////////////////////////////////////////////////////////
class LeakyRelu : public BaseLayer {
   public:
    float alpha = 0.1f;
    LeakyRelu();
    ~LeakyRelu();
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

    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override;
    void state_backward(std::vector<float> &jcb,
                        DeltaStates &input_delta_states,
                        DeltaStates &output_hidden_states,
                        TempStates &temp_states) override{};
    void param_backward(std::vector<float> &mu_a, DeltaStates &delta_states,
                        TempStates &temp_states) override{};
};
////////////////////////////////////////////////////////////////////////////////
/// Stable Softmax
////////////////////////////////////////////////////////////////////////////////
class Softmax : public BaseLayer {
   public:
    float alpha = 0.1f;
    Softmax();
    ~Softmax();
    static void softmax_mean_var(std::vector<float> &mu_z,
                                 std::vector<float> &var_z, int no,
                                 int batch_size, std::vector<float> &mu_a,
                                 std::vector<float> &jcb,
                                 std::vector<float> &var_a);

    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override;
    void state_backward(std::vector<float> &jcb,
                        DeltaStates &input_delta_states,
                        DeltaStates &output_hidden_states,
                        TempStates &temp_states) override{};
    void param_backward(std::vector<float> &mu_a, DeltaStates &delta_states,
                        TempStates &temp_states) override{};
};

////////////////////////////////////////////////////////////////////////////////
/// Remax
////////////////////////////////////////////////////////////////////////////////
class RemaxA : public BaseLayer {
   public:
    float alpha = 0.1f;
    RemaxA();
    ~RemaxA();
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

    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override{};

    void state_backward(std::vector<float> &jcb,
                        DeltaStates &input_delta_states,
                        DeltaStates &output_hidden_states,
                        TempStates &temp_states) override{};

    void param_backward(std::vector<float> &mu_a, DeltaStates &delta_states,
                        TempStates &temp_states) override{};
};