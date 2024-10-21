///////////////////////////////////////////////////////////////////////////////
// File:         base_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 13, 2023
// Updated:      April 08, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/base_layer_cuda.cuh"
#include "../include/config.h"
#include "../include/cuda_error_checking.cuh"

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

__global__ void device_raw_weight_update(float const *delta_mu_w,
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

__global__ void device_raw_bias_update(float const *delta_mu_b,
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

__global__ void device_weight_update(float const *delta_mu_w,
                                     float const *delta_var_w,
                                     float cap_factor_udapte, size_t size,
                                     float *mu_w, float *var_w)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float delta_mu_sign, delta_var_sign, delta_bar;
    if (col < size) {
        float tmp_mu = delta_mu_w[col];
        float tmp_var = delta_var_w[col];
        delta_mu_sign = (tmp_mu > 0) - (tmp_mu < 0);
        delta_var_sign = (tmp_var > 0) - (tmp_var < 0);
        delta_bar = powf(var_w[col], 0.5) / cap_factor_udapte;

        mu_w[col] += delta_mu_sign * min(sqrt(tmp_mu * tmp_mu), delta_bar);
        var_w[col] += delta_var_sign * min(sqrt(tmp_var * tmp_var), delta_bar);
        if (var_w[col] <= 0.0f) {
            var_w[col] = 1E-5f;
            //printf("w"); //Constrain printout for debugging
        }
    }
}

__global__ void device_bias_update(float const *delta_mu_b,
                                   float const *delta_var_b,
                                   float cap_factor_udapte, size_t size,
                                   float *mu_b, float *var_b)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float delta_mu_sign, delta_var_sign, delta_bar;
    if (col < size) {
        delta_mu_sign = (delta_mu_b[col] > 0) - (delta_mu_b[col] < 0);
        delta_var_sign = (delta_var_b[col] > 0) - (delta_var_b[col] < 0);
        delta_bar = powf(var_b[col], 0.5) / cap_factor_udapte;

        mu_b[col] += delta_mu_sign * min(fabsf(delta_mu_b[col]), delta_bar);
        var_b[col] += delta_var_sign * min(fabsf(delta_var_b[col]), delta_bar);
        if (var_b[col] <= 0.0f) {
            var_b[col] = 1E-5f;
            //printf("b"); //Constrain printout for debugging
        }
    }
}

BaseLayerCuda::BaseLayerCuda() {
    this->device = "cuda";
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

void BaseLayerCuda::allocate_param_delta()
/*
 */
{
    // Recalculate size for the memory alignment
    unsigned int num_w =
        ((this->num_weights + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;

    this->delta_mu_w.resize(this->num_weights, 0.0f);
    this->delta_var_w.resize(this->num_weights, 0.0f);

    cudaMalloc((void **)&this->d_delta_mu_w, num_w * sizeof(float));
    cudaMalloc((void **)&this->d_delta_var_w, num_w * sizeof(float));
    if (this->bias) {
        unsigned int num_b =
            ((this->num_biases + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;
        this->delta_mu_b.resize(this->num_biases, 0.0f);
        this->delta_var_b.resize(this->num_biases, 0.0f);
        cudaMalloc((void **)&this->d_delta_mu_b, num_b * sizeof(float));
        cudaMalloc((void **)&this->d_delta_var_b, num_b * sizeof(float));
    }
    CHECK_LAST_CUDA_ERROR();
}

void BaseLayerCuda::raw_update_weights()
/*
 */
{
    // TODO: replace with capped update version
    unsigned int blocks = (this->num_weights + this->num_cuda_threads - 1) /
                          this->num_cuda_threads;

    device_raw_weight_update<<<blocks, this->num_cuda_threads>>>(
        this->d_delta_mu_w, this->d_delta_var_w, this->num_weights,
        this->d_mu_w, this->d_var_w);

    // this->params_to_host();
    // this->delta_params_to_host();
}

void BaseLayerCuda::raw_update_biases()
/*
 */
{
    if (this->bias) {
        // TODO: replace with capped update version
        unsigned int blocks = (this->num_biases + this->num_cuda_threads - 1) /
                              this->num_cuda_threads;

        device_raw_bias_update<<<blocks, this->num_cuda_threads>>>(
            this->d_delta_mu_b, this->d_delta_var_b, this->num_biases,
            this->d_mu_b, this->d_var_b);
    }

    // this->params_to_host();
    // this->delta_params_to_host();
}

void BaseLayerCuda::update_weights()
/*
 */
{
    // TODO: replace with capped update version
    unsigned int num_add_threads = 256;
    unsigned int blocks =
        (this->num_weights + num_add_threads - 1) / num_add_threads;

    device_weight_update<<<blocks, num_add_threads>>>(
        this->d_delta_mu_w, this->d_delta_var_w, this->cap_factor_update,
        this->num_weights, this->d_mu_w, this->d_var_w);
}

void BaseLayerCuda::update_biases()
/*
 */
{
    if (this->bias) {
        // TODO: replace with capped update version
        unsigned int num_add_threads = 256;
        unsigned int blocks =
            (this->num_biases + num_add_threads - 1) / num_add_threads;

        device_bias_update<<<blocks, num_add_threads>>>(
            this->d_delta_mu_b, this->d_delta_var_b, this->cap_factor_update,
            this->num_biases, this->d_mu_b, this->d_var_b);
    }
}

void BaseLayerCuda::set_cuda_threads(int num)
/**/
{
    this->num_cuda_threads = num;
}

void BaseLayerCuda::allocate_param_memory()
/*
 */
{
    unsigned int num_w =
        ((this->num_weights + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;

    cudaMalloc((void **)&this->d_mu_w, num_w * sizeof(float));
    cudaMalloc((void **)&this->d_var_w, num_w * sizeof(float));

    if (this->bias) {
        unsigned int num_b =
            ((this->num_biases + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;
        cudaMalloc((void **)&this->d_mu_b, num_b * sizeof(float));
        cudaMalloc((void **)&this->d_var_b, num_b * sizeof(float));
    }

    CHECK_LAST_CUDA_ERROR();
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

    CHECK_LAST_CUDA_ERROR();
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

    CHECK_LAST_CUDA_ERROR();
}

void BaseLayerCuda::delta_params_to_host()
/*
 */
{
    cudaMemcpy(this->delta_mu_w.data(), this->d_delta_mu_w,
               this->num_weights * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->delta_var_w.data(), this->d_delta_var_w,
               this->num_weights * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->delta_mu_b.data(), this->d_delta_mu_b,
               this->num_biases * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->delta_var_b.data(), this->d_delta_var_b,
               this->num_biases * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK_LAST_CUDA_ERROR();
}

void BaseLayerCuda::save(std::ofstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for saving");
    }
    // Transfer data to host
    this->params_to_host();

    // Save the name length and name
    auto layer_name = this->get_layer_info();
    size_t name_length = layer_name.length();
    file.write(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    file.write(layer_name.c_str(), name_length);

    for (const auto &m_w : this->mu_w) {
        file.write(reinterpret_cast<const char *>(&m_w), sizeof(m_w));
    }
    for (const auto &v_w : this->var_w) {
        file.write(reinterpret_cast<const char *>(&v_w), sizeof(v_w));
    }
    for (const auto &m_b : this->mu_b) {
        file.write(reinterpret_cast<const char *>(&m_b), sizeof(m_b));
    }
    for (const auto &v_b : this->var_b) {
        file.write(reinterpret_cast<const char *>(&v_b), sizeof(v_b));
    }
}

void BaseLayerCuda::load(std::ifstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for loading");
    }
    // Load the name length and name
    auto layer_name = this->get_layer_info();
    std::string loaded_name;
    size_t name_length;
    file.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    loaded_name.resize(name_length);
    file.read(&loaded_name[0], name_length);

    // Check layer name
    if (layer_name != loaded_name) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Layer name are not match. Expected: " +
                                 layer_name + ", Found: " + loaded_name);
    }

    for (auto &m_w : this->mu_w) {
        file.read(reinterpret_cast<char *>(&m_w), sizeof(m_w));
    }
    for (auto &v_w : this->var_w) {
        file.read(reinterpret_cast<char *>(&v_w), sizeof(v_w));
    }
    for (auto &m_b : this->mu_b) {
        file.read(reinterpret_cast<char *>(&m_b), sizeof(m_b));
    }
    for (auto &v_b : this->var_b) {
        file.read(reinterpret_cast<char *>(&v_b), sizeof(v_b));
    }

    this->num_weights = this->mu_w.size();
    this->num_biases = this->mu_b.size();
    if (this->training) {
        this->allocate_param_delta();
    }

    // Transfer data to device
    this->params_to_device();
}

std::unique_ptr<BaseLayer> BaseLayerCuda::to_host() {
    throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                             " at line: " + std::to_string(__LINE__) +
                             ". ErrorNotImplemented");
}

void BaseLayerCuda::store_states_for_training_cuda(
    HiddenStateCuda &input_states, HiddenStateCuda &output_states)
/*
 */
{
    BackwardStateCuda *cu_bwd_states =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());
    int batch_size = input_states.block_size;
    int act_size = input_states.actual_size * batch_size;
    if (cu_bwd_states->size != act_size) {
        cu_bwd_states->size = act_size;
        cu_bwd_states->allocate_memory();
    }

    constexpr unsigned int THREADS = 256;
    // unsigned int blocks = (act_size + THREADS - 1) / THREADS;

    // fill_bwd_states_on_device<<<blocks, THREADS>>>(
    //     input_states.d_mu_a, input_states.d_jcb, act_size,
    //     cu_bwd_states->d_mu_a, cu_bwd_states->d_jcb);

    cudaMemcpy(cu_bwd_states->d_mu_a, input_states.d_mu_a,
               act_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(cu_bwd_states->d_jcb, input_states.d_jcb,
               act_size * sizeof(float), cudaMemcpyDeviceToDevice);

    int out_size = this->output_size * batch_size;
    unsigned int out_blocks = (out_size + THREADS - 1) / THREADS;

    fill_output_states_on_device<<<out_blocks, THREADS>>>(out_size,
                                                          output_states.d_jcb);
}