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

    void fc_mean_cpu(std::vector<float> &mw, std::vector<float> &mb,
                     std::vector<float> &ma, int w_pos, int b_pos, int z_pos_in,
                     int z_pos_out, int m, int n, int k,
                     std::vector<float> &mz);
    void fc_var_cpu(std::vector<float> &mw, std::vector<float> &Sw,
                    std::vector<float> &Sb, std::vector<float> &ma,
                    std::vector<float> &Sa, int w_pos, int b_pos, int z_pos_in,
                    int z_pos_out, int m, int n, int k, std::vector<float> &Sz);
    void forward();
    void param_backward();
    void state_backward();
};
