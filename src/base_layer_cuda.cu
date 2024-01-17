///////////////////////////////////////////////////////////////////////////////
// File:         base_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 13, 2023
// Updated:      January 03, 2024
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

__global__ void fill_output_states_on_device(int size, float *jcb)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size) {
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

BaseLayerCuda::BaseLayerCuda() {
    if (this->training) {
        this->bwd_states = std::make_unique<BackwardStateCuda>();
    }
}

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
        this->d_delta_mu_b, this->d_delta_var_b, this->num_biases, this->d_mu_b,
        this->d_var_b);
}

void BaseLayerCuda::allocate_param_memory()
/*
 */
{
    cudaMalloc(&this->d_mu_w, this->num_weights * sizeof(float));
    cudaMalloc(&this->d_var_w, this->num_weights * sizeof(float));
    cudaMalloc(&this->d_mu_b, this->num_biases * sizeof(float));
    cudaMalloc(&this->d_var_b, this->num_biases * sizeof(float));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Device memory allocation.");
    }
}

void BaseLayerCuda::params_to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_w, this->mu_w.data(),
               this->num_weights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_w, this->var_w.data(),
               this->num_weights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mu_b, this->mu_b.data(),
               this->num_biases * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_b, this->var_b.data(),
               this->num_biases * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Params host to device.");
    }
}

void BaseLayerCuda::params_to_host()
/*
 */
{
    cudaMemcpy(this->mu_w.data(), this->d_mu_w,
               this->num_weights * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_w.data(), this->d_var_w,
               this->num_weights * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mu_b.data(), this->d_mu_b,
               this->num_biases * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_b.data(), this->d_var_b,
               this->num_biases * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Params device to host.");
    }
}

std::unique_ptr<BaseLayer> BaseLayerCuda::to_host() {
    throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                             " at line: " + std::to_string(__LINE__) +
                             ". ErrorNotImplemented");
}

void BaseLayerCuda::store_states_for_training(HiddenStateCuda &input_states,
                                              HiddenStateCuda &output_states,
                                              BackwardStateCuda &bwd_states)
/*
 */
{
    int batch_size = input_states.block_size;
    int threads = this->num_cuda_threads;
    if (bwd_states.size == 0) {
        bwd_states.size = input_states.actual_size * batch_size;
        bwd_states.allocate_memory();
    }

    int act_size = input_states.actual_size * batch_size;
    unsigned int blocks = (act_size + threads - 1) / threads;

    fill_bwd_states_on_device<<<blocks, threads>>>(
        input_states.d_mu_a, input_states.d_jcb, act_size, bwd_states.d_mu_a,
        bwd_states.d_jcb);

    int out_size = this->output_size * batch_size;
    unsigned int out_blocks = (out_size + threads - 1) / threads;

    fill_output_states_on_device<<<out_blocks, threads>>>(out_size,
                                                          output_states.d_jcb);
}