///////////////////////////////////////////////////////////////////////////////
// File:         fc_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      November 28, 2023
// Updated:      December 19, 2023
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

#include "base_layer.h"
#include "base_layer_cuda.cuh"
#include "data_struct.h"
#include "param_init.h"
#include "struct_var.h"

__global__ void linear_fwd_mean_var(float const *mu_w, float const *var_w,
                                    float const *mu_b, float const *var_b,
                                    const float *mu_a, const float *var_a,
                                    size_t input_size, size_t output_size,
                                    int batch_size, float *mu_z, float *var_z);

__global__ void linear_fwd_full_cov(float const *mu_w, float const *var_a_f,
                                    size_t input_size, size_t output_size,
                                    int batch_size, float *var_z_fp);

__global__ void linear_fwd_full_var(float const *mu_w, float const *var_w,
                                    float const *var_b, float const *mu_a,
                                    float const *var_a, float const *var_z_fp,
                                    size_t input_size, size_t output_size,
                                    int batch_size, float *var_z,
                                    float *var_z_f);

__global__ void linear_fwd_delta_z(float const *mu_w, float const *jcb,
                                   float const *delta_mu_out,
                                   float const *delta_var_out,
                                   size_t input_size, size_t output_size,
                                   int batch_size, float *delta_mu_in,
                                   float *delta_var_in);

__global__ void linear_bwd_delta_w(float const *var_w, float const *mu_a,
                                   float const *delta_mu_out,
                                   float const *delta_var_out,
                                   size_t input_size, size_t output_size,
                                   int batch_size, float *delta_mu_w,
                                   float *delta_var_w);

__global__ void linear_bwd_delta_b(float const *var_b,
                                   float const *delta_mu_out,
                                   float const *delta_var_out,
                                   size_t input_size, size_t output_size,
                                   int batch_size, float *delta_mu_b,
                                   float *delta_var_b);

class LinearCuda : public BaseLayerCuda {
   public:
    float gain_w;
    float gain_b;
    std::string init_method;

    LinearCuda(size_t ip_size, size_t op_size, float gain_weight = 1.0f,
               float gain_bias = 1.0f, std::string method = "He");

    ~LinearCuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void init_weight_bias();

    void allocate_param_delta();

    // Overloaded functions from base layer designed for cpu version
    using BaseLayerCuda::forward;
    using BaseLayerCuda::param_backward;
    using BaseLayerCuda::state_backward;

    void forward(HiddenStateCuda &input_states, HiddenStateCuda &output_states,
                 TempStateCuda &temp_states) override;

    void state_backward(BackwardStateCuda &next_bwd_states,
                        DeltaStateCuda &input_delta_states,
                        DeltaStateCuda &output_delta_states,
                        TempStateCuda &temp_states) override;

    void param_backward(BackwardStateCuda &bwd_states,
                        DeltaStateCuda &delta_states,
                        TempStateCuda &temp_states) override;
};