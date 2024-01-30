///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 24, 2024
// Updated:      January 30, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "../include/norm_layer.h"
#include "../include/norm_layer_cuda.cuh"

LayerNormCuda::LayerNormCuda(const std::vector<int> &normalized_shape,
                             float eps, bool bias)
    : /*
       */
{
    this->normalized_shape = normalized_shape;
    this->epsilon = eps;
    this->bias = bias;
    this->init_weight_bias();
    if (this->training) {
        this->allocate_param_delta();
    }
    if (normalized_shape.size() = 1) {
        this->input_size = normalized_shape[0];
        this->output_size = normalized_shape[0];
    } else if (normalized_shape.size() == 3) {
        this->in_channels = normalized_shape[0];
        this->in_width = normalized_shape[1];
        this->in_height = normalized_shape[2];
        this->out_channels = normalized_shape[0];
        this->out_width = normalized_shape[1];
        this->out_height = normalized_shape[2];
        this->input_size = this->in_channels * this->in_width * this->in_height;
        this->output_size =
            this->out_channels * this->out_width * this->out_height;
    } else {
        throw std::runtime_error(
            "Error in file: " + std::string(__FILE__) +
            " at line: " + std::to_string(__LINE__) +
            ". Normalized shape provided are not supported.");
    }
}

LayerNormCuda::~LayerNormCuda() {
    cudaFree(d_mu_ra);
    cudaFree(d_var_ra);
}

std::string LayerNormCuda::get_layer_info() const
/*
 */
{
    return "LayerNorm()";
}

std::string LayerNormCuda::get_layer_name() const
/*
 */
{
    return "LayerNormCuda";
}

LayerType LayerNormCuda::get_layer_type() const
/*
 */
{
    return LayerType::Norm;
}

void LayerNormCuda::init_weight_bias()
/*
 */
{
    std::tie(this->num_weights, this->num_biases) =
        get_number_params_layer_norm(this->normalized_shape);

    this->mu_w.resize(this->num_weights, 1.0f);
    this->var_w.resize(this->num_weights, 1.0f);
    if (this->bias) {
        this->mu_b.resize(this->num_weights, 0.0f);
        this->var_b.resize(this->num_weights, 0.0001f);

    } else {
        this->num_biases = 0;
    }
    this->allocate_param_memory();
    this->params_to_device();
}

void LayerNormCuda::allocate_param_delta()
/*
 */
{
    this->delta_mu_w.resize(this->num_weights, 0.0f);
    this->delta_var_w.resize(this->num_weights, 0.0f);
    this->delta_mu_b.resize(this->num_biases, 0.0f);
    this->delta_var_b.resize(this->num_biases, 0.0f);
    cudaMalloc(&this->d_delta_mu_w, this->num_weights * sizeof(float));
    cudaMalloc(&this->d_delta_var_w, this->num_weights * sizeof(float));
    cudaMalloc(&this->d_delta_mu_b, this->num_biases * sizeof(float));
    cudaMalloc(&this->d_delta_var_b, this->num_biases * sizeof(float));
}

void LayerNormCuda::forward(BaseHiddenStates &input_states,
                            BaseHiddenStates &output_states,
                            BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda *>(&temp_states);

    int batch_size = input_states.block_size;
    int num_threads = this->num_cuda_threads;
    unsigned int grid_size_ra = (batch_size + num_threads - 1) / num_threads;

    if (this->normalized_shape.size() == 1) {
        layernorm_stat_mean_var_cuda<<<grid_size_ra, num_threads>>>(
            cu_input_states->d_mu_a, cu_input_states->d_var_a, this->input_size,
            batch_size, cu_temp_states.d_tmp_1, cu_temp_states.d_tmp_2);

        layernorm_sample_var_cuda<<<grid_size_ra, num_threads>>>(
            cu_input_states->d_mu_a, cu_temp_states.d_tmp_1,
            cu_temp_states.d_tmp_2, this->input_size, batch_size,
            cu_temp_states.d_tmp_2);

        running_mean_var_cuda<<<grid_size_ra, num_threads>>>(
            cu_temp_states.d_tmp_1, cu_temp_states.d_tmp_2);
    } else {
    }
}

void LayerNormCuda::state_backward(BaseBackwardStates &next_bwd_states,
                                   BaseDeltaStates &input_delta_states,
                                   BaseDeltaStates &output_delta_states,
                                   BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(&next_bwd_states);
    DeltaStateCuda *cu_input_delta_states =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    // Initialization
    int batch_size = input_delta_states.block_size;
    int threads = this->num_cuda_threads;
}

void LayerNormCuda::param_backward(BaseBackwardStates &next_bwd_states,
                                   BaseDeltaStates &delta_states,
                                   BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(&next_bwd_states);
    DeltaStateCuda *cu_delta_states =
        dynamic_cast<DeltaStateCuda *>(&delta_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    // Initalization
    int batch_size = delta_states.block_size;
    int threads = this->num_cuda_threads;
    dim3 block_dim(threads, threads);
}

////////////////////////////////////////////////////////////////////////////////
//// CUDA Kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void layernorm_stat_mean_var_cuda(float const *mu_a,
                                             float const *var_a, int ni,
                                             int batch_size, float *mu_s,
                                             float *var_s)
/*
 */
{
    // ni in the case of conv2d will be wihi * fi
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < batch_size) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < ni; i++)  // n = wihi*B
        {
            sum_mu += mu_a[col * ni + i];
            sum_var += var_a[col * ni + i];
        }
        mu_s[col] = sum_mu / ni;
        var_s[col] = sum_var;
    }
}

__global__ void layernorm_sample_var_cuda(float const *mu_a, float const *mu_s,
                                          float const *var_s, int ni,
                                          int batch_size, float *var_sample)
/*
 */
{
    // ni in the case of conv2d will be wihi * fi
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < ni; i++) {
            sum += (mu_a[col * ni + i] - mu_s[col]) *
                   (mu_a[col * ni + i] - mu_s[col]);
        }
        var_sample[col] = (sum + var_s[col]) / (ni - 1);
    }
}

__global__ void running_mean_var_cuda(float const *mu_s, float const *var_s,
                                      float const *mu_ra_prev,
                                      float const *var_ra_prev, float momentum,
                                      int num_states, float *mu_ra,
                                      float *var_ra)
/*Copute the running average for the normalization layers.

Args:
    ms: New statistical mean of samples
    Ss: New statistical variance of samples
    mraprev: Previous mean for the normalization layers
    Sraprev: Previous statistical variance for the normalization layers
    momentum: Running average factor
    mra: Statistical mean for the normalization layers
    Sra: Statistical variance for the normalization layers
    N: Size of mra
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_states) {
        float tmp = mu_ra_prev[col] * momentum + mu_s[col] * (1 - momentum);
        var_ra[col] = var_ra_prev[col] * momentum + var_s[col] * (1 - momentum);
        mu_ra[col] = tmp;
    }
}

__global__ void layernorm_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a,
    float const *mu_ra, float const *var_ra, float epsilon, int ni, int B,
    float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < ni && row < B) {
        mu_z[col + row * ni] = (1 / sqrtf(var_ra[row] + epsilon)) *
                                   (mu_a[col + row * ni] - mu_ra[row]) *
                                   mu_w[col] +
                               mu_b[col];
        var_z[col + row * ni] =
            (1.0f / (var_ra[row] + epsilon)) *
                (var_a[col + row * ni] * mu_w[col] * mu_w[col] +
                 var_w[col] *
                     (mu_a[col + row * ni] * mu_a[col + row * ni] -
                      mu_ra[row] * mu_ra[row] + var_a[col + row * ni])) +
            var_b[col];
    }
}

__global__ void layernorm2d_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a,
    float const *mu_ra, float const *var_ra, float epsilon, int wihi, int m,
    int k, float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi * fi, m = B
    {
        mu_z[col + row * k] = (1.0f / sqrtf(var_ra[row] + epsilon)) *
                                  (mu_a[col + row * k] - mu_ra[row]) *
                                  mu_w[col / wihi] +
                              mu_b[col / wihi];
        var_z[col + row * k] =
            (1.0f / (var_ra[row] + epsilon)) *
                (var_a[col + row * k] * mu_w[col / wihi] * mu_w[col / wihi] +
                 var_w[col / wihi] *
                     (mu_a[col + row * k] * mu_a[col + row * k] -
                      mu_ra[row] * mu_ra[row] + var_a[col + row * k])) +
            var_b[col / wihi];
    }
}

////
// Layer Norm's backward
////
__global__ void layernorm_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *var_hat,
    float const *delta_mu_out, float const *delta_var_out, float epsilon,
    int ni, int batch_size, float *delta_mu, float *delta_var)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < batch_size) {
        float tmp = (1.0f / sqrtf(var_hat[row] + epsilon)) * mu_w[col] *
                    jcb[col + row * ni];

        delta_mu[col + row * ni] = tmp * delta_mu_out[col + row * ni];
        delta_var[col + row * ni] = tmp * delta_var_out[col + row * ni] * tmp;
    }
}

__global__ void layernorm_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_hat,
    float const *var_hat, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int ni, int batch_size, float *delta_mu_w,
    float *delta_var_w)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float A = (1.0f / sqrtf(var_hat[i] + epsilon)) *
                      (mu_a[col + i * ni] - mu_hat[i]) * var_w[col];
            sum_mu += A * delta_mu_out[col + i * ni];
            sum_var += A * delta_var_out[col + i * ni] * A;
        }
        delta_mu_w[col] = sum_mu;
        delta_var_w[col] = sum_var;
    }
}

__global__ void layernorm_bwd_delta_b_cuda(float const *var_b,
                                           float const *delta_mu_out,
                                           float const *delta_var_out,
                                           float epsilon, int ni,
                                           int batch_size, float *delta_mu_b,
                                           float *delta_var_b)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float A = var_b[col];
            sum_mu += A * delta_mu_out[col + i * ni];
            sum_var += A * delta_var_out[col + i * ni] * A;
        }
        delta_mu_b[col] = sum_mu;
        delta_var_b[col] = sum_var;
    }
}

__global__ void layernorm2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *var_hat,
    float const *delta_mu_out, float const *delta_var_out, float epsilon,
    int wihi, int fi, int m, float *delta_mu, float *delta_var)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < wihi && row < m)  // k = wihi * fi, m = B
    {
        float tmp = (1.0f / sqrtf(var_hat[row % fi] + epsilon)) *
                    mu_w[row % fi] * jcb[col + row * wihi];

        delta_mu[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];
        delta_var[col + row * wihi] =
            tmp * delta_var_out[col + row * wihi] * tmp;
    }
}

__global__ void layernorm2d_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_hat,
    float const *var_hat, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int wihi, int m, int k, float *delta_mu_w,
    float *delta_var_w)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        float A = (1.0f / sqrtf(var_hat[row] + epsilon)) *
                  (mu_a[col + row * k] - mu_hat[row]) * var_w[col / wihi];
        delta_mu_w[col + row * k] = A * delta_mu_out[col + row * k];
        delta_var_w[col + row * k] = A * delta_var_out[col + row * k] * A;
    }
}

__global__ void layernorm2d_bwd_delta_b_cuda(float const *var_b,
                                             float const *delta_mu_out,
                                             float const *delta_var_out,
                                             float epsilon, int wihi, int m,
                                             int k, float *delta_mu_b,
                                             float *delta_var_b)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        float A = var_b[col / wihi];
        delta_mu_b[col + row * k] = A * delta_mu_out[col + row * k];
        delta_var_b[col + row * k] = A * delta_var_out[col + row * k] * A;
    }
}
