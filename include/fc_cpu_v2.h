///////////////////////////////////////////////////////////////////////////////
// File:         fc_cpu_v2.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 20, 2023
// Updated:      September 20, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <thread>
#include <vector>

class FullyConnectedLayer {
   public:
    size_t input_size, output_size, batch_size;
    std::vector<float> mu_z;
    std::vector<float> var_z;
    std::vector<float> mu_a;
    std::vector<float> var_a;
    std::vector<float> jcb;

    FullyConnectedLayer(size_t input_size, size_t output_size,
                        size_t batch_size);
    ~FullyConnectedLayer();

    void fwd_mean_var(std::vector<float> &mu_w, std::vector<float> &var_w,
                      std::vector<float> &mu_b, std::vector<float> &var_b,
                      std::vector<float> &mu_a, std::vector<float> &var_a,
                      int start_chunk, int end_chunk, int w_pos, int b_pos,
                      int z_pos_in, int z_pos_out, int output_size,
                      int input_size, int batch_size, std::vector<float> &mu_z,
                      std::vector<float> &var_z);

    void fwd_mean_var_mp(std::vector<float> &mu_w, std::vector<float> &var_w,
                         std::vector<float> &mu_b, std::vector<float> &var_b,
                         std::vector<float> &mu_a, std::vector<float> &var_a,
                         int w_pos, int b_pos, int z_pos_in, int z_pos_out,
                         int output_size, int input_size, int batch_size,
                         unsigned int NUM_THREADS, std::vector<float> &mu_z,
                         std::vector<float> &var_z);
    void forward();
    void param_backward();
    void state_backward();
};
