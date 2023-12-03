///////////////////////////////////////////////////////////////////////////////
// File:         fc_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      November 28, 2023
// Updated:      December 01, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "base_layer_cuda.cuh"
#include "struct_var.h"

__global__ void fc_mean_var(float const *mu_w, float const *var_w,
                            float const *mu_b, float const *var_b,
                            const float *mu_a, const float *var_a,
                            size_t input_size, size_t output_size,
                            int batch_size, float *mu_z, float *var_z);

__global__ void fc_cov(float const *mu_w, float const *var_a_f,
                       size_t input_size, size_t output_size, int batch_size,
                       float *var_z_fp);

__global__ void fc_full_var(float const *mu_w, float const *var_w,
                            float const *var_b, float const *mu_a,
                            float const *var_a, float const *var_z_fp,
                            size_t input_size, size_t output_size,
                            int batch_size, float *var_z, float *var_z_f);

__global__ void fc_delta_mu_z(float const *mu_w, float const *jcb,
                              float const *delta_mu, float const *delta_var,
                              int input_size, int output_size, int batch_size,
                              float *delta_mu_prev, float *delta_var_prev);

class FullConnectedCuda : public BaseLayerCuda {
   public:
    void forward(HiddenStates &input_states, HiddenStates &output_states,
                 TempStates &temp_states) override;
};