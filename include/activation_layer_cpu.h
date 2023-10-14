///////////////////////////////////////////////////////////////////////////////
// File:         activation_layer_cpu.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      October 09, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <thread>

#include "base_layer.h"

class ReLU : public BaseLayer {
   public:
    ReLU();
    ~ReLU();
    void relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                       int start_chunk, int end_chunk, std::vector<float> &mu_a,
                       std::vector<float> &jcb, std::vector<float> &var_a);
    void relu_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                          int n, unsigned int NUM_THREADS,
                          std::vector<float> &mu_a, std::vector<float> &jcb,
                          std::vector<float> &var_a);
    void forward(HiddenStates &input_states, HiddenStates &output_states);
    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var);
    void param_backward();
};