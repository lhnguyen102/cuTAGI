///////////////////////////////////////////////////////////////////////////////
// File:         activation_layer_cpu.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      October 15, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <thread>

#include "base_layer.h"
#include "common.h"
////////////////////////////////////////////////////////////////////////////////
/// Relu
////////////////////////////////////////////////////////////////////////////////
class Relu : public BaseLayer {
   public:
    Relu();
    ~Relu();
    void relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                       int start_chunk, int end_chunk, std::vector<float> &mu_a,
                       std::vector<float> &jcb, std::vector<float> &var_a);
    void relu_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                          int n, unsigned int num_threads,
                          std::vector<float> &mu_a, std::vector<float> &jcb,
                          std::vector<float> &var_a);

    void forward(HiddenStates &input_states,
                 HiddenStates &output_states) override;
    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var) override;
    void param_backward() override;
};
////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
class Sigmoid : public BaseLayer {
   public:
    Sigmoid();
    ~Sigmoid();
    void sigmoid_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                          int start_chunk, int end_chunk,
                          std::vector<float> &mu_a, std::vector<float> &jcb,
                          std::vector<float> &var_a);
    void sigmoid_mean_var_mp(std::vector<float> &mu_z,
                             std::vector<float> &var_z, int n,
                             unsigned int num_threads, std::vector<float> &mu_a,
                             std::vector<float> &jcb,
                             std::vector<float> &var_a);

    void forward(HiddenStates &input_states,
                 HiddenStates &output_states) override;
    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var) override;
    void param_backward() override;
};
////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
class Tanh : public BaseLayer {
   public:
    Tanh();
    ~Tanh();
    void tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                       int start_chunk, int end_chunk, std::vector<float> &mu_a,
                       std::vector<float> &jcb, std::vector<float> &var_a);
    void tanh_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                          int n, unsigned int num_threads,
                          std::vector<float> &mu_a, std::vector<float> &jcb,
                          std::vector<float> &var_a);

    void forward(HiddenStates &input_states,
                 HiddenStates &output_states) override;
    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var) override;
    void param_backward() override;
};
////////////////////////////////////////////////////////////////////////////////
/// Mixture Relu
////////////////////////////////////////////////////////////////////////////////
class MixtureRelu : public BaseLayer {
   public:
    float omega_tol = 0.0000001f;
    MixtureRelu();
    ~MixtureRelu();
    void mixture_relu_mean_var(std::vector<float> &mu_z,
                               std::vector<float> &var_z, float omega_tol,
                               int start_chunk, int end_chunk,
                               std::vector<float> &mu_a,
                               std::vector<float> &jcb,
                               std::vector<float> &var_a);
    void mixture_relu_mean_var_mp(std::vector<float> &mu_z,
                                  std::vector<float> &var_z, float omega_tol,
                                  int n, unsigned int num_threads,
                                  std::vector<float> &mu_a,
                                  std::vector<float> &jcb,
                                  std::vector<float> &var_a);

    void forward(HiddenStates &input_states,
                 HiddenStates &output_states) override;
    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var) override;
    void param_backward() override;
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
class MixtureSigmoid : public BaseLayer {
   public:
    float omega_tol = 0.0000001f;
    MixtureSigmoid();
    ~MixtureSigmoid();
    void mixture_sigmoid_mean_var(std::vector<float> &mu_z,
                                  std::vector<float> &var_z, float omega_tol,
                                  int start_chunk, int end_chunk,
                                  std::vector<float> &mu_a,
                                  std::vector<float> &jcb,
                                  std::vector<float> &var_a);
    void mixture_sigmoid_mean_var_mp(std::vector<float> &mu_z,
                                     std::vector<float> &var_z, float omega_tol,
                                     int n, unsigned int num_threads,
                                     std::vector<float> &mu_a,
                                     std::vector<float> &jcb,
                                     std::vector<float> &var_a);

    void forward(HiddenStates &input_states,
                 HiddenStates &output_states) override;
    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var) override;
    void param_backward() override;
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
class MixtureTanh : public BaseLayer {
   public:
    float omega_tol = 0.0000001f;
    MixtureTanh();
    ~MixtureTanh();
    void mixture_tanh_mean_var(std::vector<float> &mu_z,
                               std::vector<float> &var_z, float omega_tol,
                               int start_chunk, int end_chunk,
                               std::vector<float> &mu_a,
                               std::vector<float> &jcb,
                               std::vector<float> &var_a);
    void mixture_tanh_mean_var_mp(std::vector<float> &mu_z,
                                  std::vector<float> &var_z, float omega_tol,
                                  int n, unsigned int num_threads,
                                  std::vector<float> &mu_a,
                                  std::vector<float> &jcb,
                                  std::vector<float> &var_a);

    void forward(HiddenStates &input_states,
                 HiddenStates &output_states) override;
    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var) override;
    void param_backward() override;
};
////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
class Softplus : public BaseLayer {
   public:
    Softplus();
    ~Softplus();
    void softplus_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
                           std::vector<float> &var_a);
    void softplus_mean_var_mp(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int n,
                              unsigned int num_threads,
                              std::vector<float> &mu_a, std::vector<float> &jcb,
                              std::vector<float> &var_a);

    void forward(HiddenStates &input_states,
                 HiddenStates &output_states) override;
    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var) override;
    void param_backward() override;
};
////////////////////////////////////////////////////////////////////////////////
/// Leaky ReLU
////////////////////////////////////////////////////////////////////////////////
class LeakyRelu : public BaseLayer {
   public:
    float alpha = 0.1f;
    LeakyRelu();
    ~LeakyRelu();
    void leaky_relu_mean_var(std::vector<float> &mu_z,
                             std::vector<float> &var_z, float alpha,
                             int start_chunk, int end_chunk,
                             std::vector<float> &mu_a, std::vector<float> &jcb,
                             std::vector<float> &var_a);
    void leaky_relu_mean_var_mp(std::vector<float> &mu_z,
                                std::vector<float> &var_z, float alpha, int n,
                                unsigned int num_threads,
                                std::vector<float> &mu_a,
                                std::vector<float> &jcb,
                                std::vector<float> &var_a);

    void forward(HiddenStates &input_states,
                 HiddenStates &output_states) override;
    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var) override;
    void param_backward() override;
};