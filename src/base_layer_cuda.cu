///////////////////////////////////////////////////////////////////////////////
// File:         base_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 13, 2023
// Updated:      December 19, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/base_layer_cuda.cuh"

__global__ void fill_bwd_states_on_device(float const *mu_a_in,
                                          float const *jcb_in, int size,
                                          float *mu_a, float *jcb)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size) {
        mu_a[col] = mu_a_in[col];
        jcb[col] = jcb_in[col];
    }
}

__global__ void fill_output_states_on_device(float const *mu_z,
                                             float const *var_z, int size,
                                             float *mu_a, float *jcb,
                                             float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size) {
        mu_a[col] = mu_z[col];
        var_a[col] = var_z[col];
        jcb[col] = 1.0f;
    }
}

__global__ void device_weight_update(float const *delta_mu_w,
                                     float const *delta_var_w, size_t size,
                                     float *mu_w, float *var_w)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < size) {
        mu_w[col] += delta_mu_w[col];
        var_w[col] += delta_var_w[col];
    }
}

__global__ void device_bias_update(float const *delta_mu_b,
                                   float const *delta_var_b, size_t size,
                                   float *mu_b, float *var_b)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < size) {
        mu_b[col] += delta_mu_b[col];
        var_b[col] += delta_var_b[col];
    }
}

__global__ void device_weight_update_with_limit(float const *delta_mu_w,
                                                float const *delta_var_w,
                                                float cap_factor, size_t size,
                                                float *mu_w, float *var_w)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float delta_mu_sign, delta_var_sign, delta_bar;
    if (col < size) {
        delta_mu_sign = (delta_mu_w[col] > 0) - (delta_mu_w[col] < 0);
        delta_var_sign = (delta_var_w[col] > 0) - (delta_var_w[col] < 0);
        delta_bar = powf(var_w[col], 0.5) / cap_factor;

        mu_w[col] += delta_mu_sign * min(fabsf(delta_mu_w[col]), delta_bar);
        var_w[col] += delta_var_sign * min(fabsf(delta_var_w[col]), delta_bar);
    }
}

__global__ void device_bias_update_with_limit(float const *delta_mu_b,
                                              float const *delta_var_b,
                                              float cap_factor, size_t size,
                                              float *mu_b, float *var_b)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float delta_mu_sign, delta_var_sign, delta_bar;
    if (col < size) {
        delta_mu_sign = (delta_mu_b[col] > 0) - (delta_mu_b[col] < 0);
        delta_var_sign = (delta_var_b[col] > 0) - (delta_var_b[col] < 0);
        delta_bar = powf(var_b[col], 0.5) / cap_factor;

        mu_b[col] += delta_mu_sign * min(fabsf(delta_mu_b[col]), delta_bar);
        var_b[col] += delta_var_sign * min(fabsf(delta_var_b[col]), delta_bar);
    }
}

BaseLayerCuda::BaseLayerCuda() {}

BaseLayerCuda::~BaseLayerCuda()
/*
 */
{
    cudaFree(d_mu_w);
    cudaFree(d_var_w);
    cudaFree(d_mu_b);
    cudaFree(d_var_b);
    cudaFree(d_delta_mu_w);
    cudaFree(d_delta_var_w);
    cudaFree(d_delta_mu_b);
    cudaFree(d_delta_var_b);
}

void BaseLayerCuda::forward(HiddenStateCuda &input_states,
                            HiddenStateCuda &output_states,
                            TempStateCuda &temp_states)
/*
 */
{
    if (this->device.compare("cuda") != 0) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Device mismatch");
    }
}

void BaseLayerCuda::state_backward(BackwardStateCuda &next_bwd_states,
                                   DeltaStateCuda &input_delta_states,
                                   DeltaStateCuda &output_delta_states,
                                   TempStateCuda &temp_states)
/*
 */
{
    if (this->device.compare("cuda") != 0) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Device mismatch");
    }
}

void BaseLayerCuda::param_backward(BackwardStateCuda &bwd_states,
                                   DeltaStateCuda &delta_states,
                                   TempStateCuda &temp_states)
/*
 */
{
    if (this->device.compare("cuda") != 0) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Device mismatch");
    }
}

void BaseLayerCuda::update_weights()
/*
 */
{
    // TODO: replace with capped update version
    unsigned int blocks = (this->num_weights + this->num_cuda_threads - 1) /
                          this->num_cuda_threads;

    device_weight_update<<<blocks, this->num_cuda_threads>>>(
        this->d_delta_mu_w, this->d_delta_var_w, this->num_weights,
        this->d_mu_w, this->d_var_w);
}

void BaseLayerCuda::update_biases()
/*
 */
{
    // TODO: replace with capped update version
    unsigned int blocks = (this->num_biases + this->num_cuda_threads - 1) /
                          this->num_cuda_threads;

    device_bias_update<<<blocks, this->num_cuda_threads>>>(
        this->d_delta_mu_b, this->d_delta_var_b, this->num_biases, this->d_mu_w,
        this->d_var_w);
}
