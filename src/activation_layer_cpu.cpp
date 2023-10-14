///////////////////////////////////////////////////////////////////////////////
// File:         activation_layer_cpu.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      October 09, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/activation_layer_cpu.h"

ReLU::ReLU(){};
ReLU::~ReLU(){};
void ReLU::relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                         int start_chunk, int end_chunk,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
                         std::vector<float> &var_a)
/*
 */
{
    float zero_pad = 0;
    float one_pad = 1;
    float tmp;
    int col;
    for (col = start_chunk; col < end_chunk; col++) {
        tmp = std::max(mu_z[col], zero_pad);
        mu_a[col] = tmp;
        if (tmp == 0) {
            jcb[col] = zero_pad;
            var_a[col] = zero_pad;
        } else {
            jcb[col] = one_pad;
            var_a[col] = var_z[col];
        }
    }
}

void ReLU::relu_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                            int n, unsigned int NUM_THREADS,
                            std::vector<float> &mu_a, std::vector<float> &jcb,
                            std::vector<float> &var_a)
/*
 */
{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
    int start_chunk, end_chunk;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(&ReLU::relu_mean_var, this, std::ref(mu_z),
                        std::ref(var_z), start_chunk, end_chunk, std::ref(mu_a),
                        std::ref(jcb), std::ref(var_a));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void ReLU::forward(HiddenStates &input_states, HiddenStates &output_states)
/*
 */
{
    // TODO: figure it out how to compute batch size based on all member
    // variables
}