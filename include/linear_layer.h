///////////////////////////////////////////////////////////////////////////////
// File:         linear_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 20, 2023
// Updated:      April 18, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base_layer.h"
#include "data_struct.h"
#include "param_init.h"

void linear_fwd_mean_var(std::vector<float> &mu_w, std::vector<float> &var_w,
                         std::vector<float> &mu_b, std::vector<float> &var_b,
                         std::vector<float> &mu_a, std::vector<float> &var_a,
                         int start_chunk, int end_chunk, size_t input_size,
                         size_t output_size, int batch_size, bool bias,
                         std::vector<float> &mu_z, std::vector<float> &var_z);

void linear_fwd_mean_var_mp(std::vector<float> &mu_w, std::vector<float> &var_w,
                            std::vector<float> &mu_b, std::vector<float> &var_b,
                            std::vector<float> &mu_a, std::vector<float> &var_a,
                            size_t input_size, size_t output_size,
                            int batch_size, bool bias, unsigned int num_threads,
                            std::vector<float> &mu_z,
                            std::vector<float> &var_z);

void linear_fwd_full_cov(std::vector<float> &mu_w, std::vector<float> &var_a_f,
                         size_t input_size, size_t output_size, int B,
                         int start_chunk, int end_chunk,
                         std::vector<float> &var_z_fp);

void linear_fwd_full_cov_mp(std::vector<float> &mu_w,
                            std::vector<float> &var_a_f, size_t input_size,
                            size_t output_size, int batch_size,
                            unsigned int num_threads,
                            std::vector<float> &var_z_fp);

void linear_fwd_fc_full_var(std::vector<float> &var_w,
                            std::vector<float> &var_b, std::vector<float> &mu_a,
                            std::vector<float> &var_a,
                            std::vector<float> &var_z_fp, size_t input_size,
                            size_t output_size, int B, int start_chunk,
                            int end_chunk, std::vector<float> &var_z,
                            std::vector<float> &var_z_f);

void linear_fwd_fc_full_var_mp(
    std::vector<float> &var_w, std::vector<float> &var_b,
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &var_z_fp, int input_size, int output_size,
    int batch_size, unsigned int num_threads, std::vector<float> &var_z,
    std::vector<float> &var_z_f);

void linear_bwd_fc_delta_z(std::vector<float> &mu_w, std::vector<float> &jcb,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var, size_t input_size,
                           size_t output_size, int B, int start_chunk,
                           int end_chunk, std::vector<float> &delta_mu_z,
                           std::vector<float> &delta_var_z);

void linear_bwd_fc_delta_z_mp(std::vector<float> &mu_w, std::vector<float> &jcb,
                              std::vector<float> &delta_mu,
                              std::vector<float> &delta_var, size_t input_size,
                              size_t output_size, int batch_size,
                              unsigned int num_threads,
                              std::vector<float> &delta_mu_z,
                              std::vector<float> &delta_var_z);

void linear_bwd_fc_delta_w(std::vector<float> &var_w, std::vector<float> &mu_a,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var, size_t input_size,
                           size_t output_size, int batch_size, int start_chunk,
                           int end_chunk, std::vector<float> &delta_mu_w,
                           std::vector<float> &delta_var_w);

void linear_bwd_fc_delta_w_mp(std::vector<float> &var_w,
                              std::vector<float> &mu_a,
                              std::vector<float> &delta_mu,
                              std::vector<float> &delta_var, size_t input_size,
                              size_t output_size, int batch_size,
                              unsigned int num_threads,
                              std::vector<float> &delta_mu_w,
                              std::vector<float> &delta_var_w);

void linear_bwd_fc_delta_b(std::vector<float> &var_b,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var, size_t output_size,
                           int batch_size, int start_chunk, int end_chunk,
                           std::vector<float> &delta_mu_b,
                           std::vector<float> &delta_var_b);

void linear_bwd_fc_delta_b_mp(std::vector<float> &var_b,
                              std::vector<float> &delta_mu,
                              std::vector<float> &delta_var, size_t output_size,
                              int batch_size, unsigned int num_threads,
                              std::vector<float> &delta_mu_b,
                              std::vector<float> &delta_var_b);

class Linear : public BaseLayer {
   public:
    float gain_w;
    float gain_b;
    std::string init_method;

    Linear(size_t ip_size, size_t op_size, bool bias = true,
           float gain_weight = 1.0f, float gain_bias = 1.0f,
           std::string method = "He");

    ~Linear();

    // Delete copy constructor and copy assignment
    Linear(const Linear &) = delete;
    Linear &operator=(const Linear &) = delete;

    // Optionally implement move constructor and move assignment
    Linear(Linear &&) = default;
    Linear &operator=(Linear &&) = default;

    virtual std::string get_layer_info() const override;

    virtual std::string get_layer_name() const override;

    virtual LayerType get_layer_type() const override;

    void init_weight_bias() override;

    virtual void forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states) override;

    virtual void backward(BaseDeltaStates &input_delta_states,
                          BaseDeltaStates &output_delta_states,
                          BaseTempStates &temp_states,
                          bool state_udapte) override;

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};
