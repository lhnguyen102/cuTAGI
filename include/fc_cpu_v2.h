///////////////////////////////////////////////////////////////////////////////
// File:         fc_cpu_v2.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 20, 2023
// Updated:      October 19, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base_layer.h"
#include "net_prop.h"
#include "struct_var.h"

class FullyConnectedLayer : public BaseLayer {
   public:
    size_t input_size, output_size;
    std::vector<float> mu_w;
    std::vector<float> var_w;
    std::vector<float> mu_b;
    std::vector<float> var_b;
    std::vector<float> mu_a;
    std::vector<float> jcb;
    std::vector<float> delta_mu;
    std::vector<float> delta_var;
    std::vector<float> delta_mu_w;
    std::vector<float> delta_var_w;
    std::vector<float> delta_mu_b;
    std::vector<float> delta_var_b;

    FullyConnectedLayer(size_t input_size, size_t output_size);

    ~FullyConnectedLayer();

    int get_input_size() override;

    int get_output_size() override;

    void init_weight_bias(float &gain_w, float &gain_b,
                          const std::string &init_method);

    void init_weight_bias();

    void fwd_mean_var(std::vector<float> &mu_w, std::vector<float> &var_w,
                      std::vector<float> &mu_b, std::vector<float> &var_b,
                      std::vector<float> &mu_a, std::vector<float> &var_a,
                      int start_chunk, int end_chunk, size_t input_size,
                      size_t output_size, int batch_size,
                      std::vector<float> &mu_z, std::vector<float> &var_z);

    void fwd_mean_var_mp(std::vector<float> &mu_a, std::vector<float> &var_a,
                         int output_size, unsigned int NUM_THREADS,
                         std::vector<float> &mu_z, std::vector<float> &var_z);

    void fwd_full_cov(std::vector<float> &mu_w, std::vector<float> &var_a_f,
                      size_t input_size, size_t output_size, int B,
                      int start_chunk, int end_chunk,
                      std::vector<float> &var_z_fp);

    void fwd_full_cov_mp(std::vector<float> &var_a_f, int B,
                         unsigned int NUM_THREADS,
                         std::vector<float> &var_z_fp);

    void fwd_fc_full_var(std::vector<float> &var_w, std::vector<float> &var_b,
                         std::vector<float> &mu_a, std::vector<float> &var_a,
                         std::vector<float> &var_z_fp, size_t input_size,
                         size_t output_size, int B, int start_chunk,
                         int end_chunk, std::vector<float> &var_z,
                         std::vector<float> &var_z_f);

    void fwd_fc_full_var_mp(std::vector<float> &mu_a, std::vector<float> &var_a,
                            std::vector<float> &var_z_fp, int B,
                            unsigned int NUM_THREADS, std::vector<float> &var_z,
                            std::vector<float> &var_z_f);

    // Hidden states
    void bwd_fc_delta_z(std::vector<float> &mu_w, std::vector<float> &jcb,
                        std::vector<float> &delta_mu,
                        std::vector<float> &delta_var, size_t input_size,
                        size_t output_size, int B, int start_chunk,
                        int end_chunk, std::vector<float> &delta_mu_z,
                        std::vector<float> &delta_var_z);

    void bwd_fc_delta_z_mp(std::vector<float> &jcb,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var, int B,
                           unsigned int NUM_THREADS,
                           std::vector<float> &delta_mu_z,
                           std::vector<float> &delta_var_z);

    // Parameters
    void bwd_fc_delta_w(std::vector<float> &var_w, std::vector<float> &mu_a,
                        std::vector<float> &delta_mu,
                        std::vector<float> &delta_var, size_t input_size,
                        size_t output_size, int batch_size, int start_chunk,
                        int end_chunk, std::vector<float> &delta_mu_w,
                        std::vector<float> &delta_var_w);

    void bwd_fc_delta_w_mp(std::vector<float> &mu_a,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var, int batch_size,
                           unsigned int NUM_THREADS,
                           std::vector<float> &delta_mu_w,
                           std::vector<float> &delta_var_w);

    void bwd_fc_delta_b(std::vector<float> &var_b, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var, int batch_size,
                        size_t output_size, int start_chunk, int end_chunk,
                        std::vector<float> &delta_mu_b,
                        std::vector<float> &delta_var_b);

    void bwd_fc_delta_b_mp(std::vector<float> &delta_mu,
                           std::vector<float> &delta_var, int batch_size,
                           unsigned int NUM_THREADS,
                           std::vector<float> &delta_mu_b,
                           std::vector<float> &delta_var_b);

    void forward(HiddenStates &input_states,
                 HiddenStates &output_states) override;

    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var) override;

    void param_backward() override;
};