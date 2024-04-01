///////////////////////////////////////////////////////////////////////////////
// File:         data_struct_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 10, 2023
// Updated:      March 28, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/data_struct_cuda.cuh"

////////////////////////////////////////////////////////////////////////////////
// Hidden States
////////////////////////////////////////////////////////////////////////////////
HiddenStateCuda::HiddenStateCuda(size_t size, size_t block_size)
    : BaseHiddenStates(size, block_size)
/*
 */
{
    // Allocate data on gpu device
    this->allocate_memory();
}

HiddenStateCuda::HiddenStateCuda() : BaseHiddenStates() {}

HiddenStateCuda::~HiddenStateCuda()
/*
Free GPU memory using cudaFree
*/
{
    cudaFree(this->d_mu_a);
    cudaFree(this->d_var_a);
    cudaFree(this->d_jcb);
}

void HiddenStateCuda::set_input_x(const std::vector<float> &mu_x,
                                  const std::vector<float> &var_x,
                                  const size_t block_size)
/*
 */
{
    size_t data_size = mu_x.size();
    this->actual_size = data_size / block_size;
    this->block_size = block_size;

    for (int i = 0; i < data_size; i++) {
        this->mu_a[i] = mu_x[i];
    }
    if (var_x.size() == data_size) {
        for (int i = 0; i < data_size; i++) {
            this->var_a[i] = var_x[i];
        }
    }
    this->chunks_to_device(data_size);
}

void HiddenStateCuda::allocate_memory() {
    // Allocate memory on the GPU using cudaMalloc
    cudaMalloc(&this->d_mu_a, size * sizeof(float));
    cudaMalloc(&this->d_var_a, size * sizeof(float));
    cudaMalloc(&this->d_jcb, size * sizeof(float));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Device memory allocation.");
    }
};

void HiddenStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_a, this->mu_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_a, this->var_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_jcb, this->jcb.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);

    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //     throw std::invalid_argument("Error in file: " + std::string(__FILE__)
    //     +
    //                                 " at line: " + std::to_string(__LINE__) +
    //                                 ". Copying host to device.");
    // }
}

void HiddenStateCuda::chunks_to_device(const size_t chunk_size)
/*
 */
{
    assert(chunk_size <= this->size);

    cudaMemcpy(this->d_mu_a, this->mu_a.data(), chunk_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_a, this->var_a.data(), chunk_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_jcb, this->jcb.data(), chunk_size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void HiddenStateCuda::to_host()
/*
 */
{
    cudaMemcpy(this->mu_a.data(), this->d_mu_a,
               this->mu_a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_a.data(), this->d_var_a,
               this->var_a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb.data(), this->d_jcb, this->jcb.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
}

////////////////////////////////////////////////////////////////////////////////
// Delta Hidden States
////////////////////////////////////////////////////////////////////////////////
DeltaStateCuda::DeltaStateCuda(size_t size, size_t block_size)
    : BaseDeltaStates(size, block_size)
/*
 */
{
    // Allocate data on gpu device
    this->allocate_memory();
}

DeltaStateCuda::DeltaStateCuda() : BaseDeltaStates() {}

DeltaStateCuda::~DeltaStateCuda()
/*
 */
{
    cudaFree(this->d_delta_mu);
    cudaFree(this->d_delta_var);
}

void DeltaStateCuda::allocate_memory()
/*
 */
{
    // Allocate memory on the GPU using cudaMalloc
    cudaMalloc(&this->d_delta_mu, size * sizeof(float));
    cudaMalloc(&this->d_delta_var, size * sizeof(float));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Device memory allocation.");
    }
}

void DeltaStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_delta_mu, this->delta_mu.data(),
               this->size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_delta_var, this->delta_var.data(),
               this->size * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Copying host to device.");
    }
}

void DeltaStateCuda::to_host()
/*
 */
{
    cudaMemcpy(this->delta_mu.data(), this->d_delta_mu,
               this->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->delta_var.data(), this->d_delta_var,
               this->size * sizeof(float), cudaMemcpyDeviceToHost);
}

void DeltaStateCuda::reset_zeros() {
    cudaMemset(d_delta_mu, 0, sizeof(float) * size);
    cudaMemset(d_delta_var, 0, sizeof(float) * size);
}

void DeltaStateCuda::copy_from(const BaseDeltaStates &source, int num_data)
/*
 */
{
    if (num_data == -1) {
        num_data = this->size;
    }

    const DeltaStateCuda *cu_source =
        dynamic_cast<const DeltaStateCuda *>(&source);

    if (!cu_source) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Invalid source.");
    }

    cudaMemcpy(this->d_delta_mu, cu_source->d_delta_mu,
               num_data * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->d_delta_var, cu_source->d_delta_var,
               num_data * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Copying data on device.");
    }
}

////////////////////////////////////////////////////////////////////////////////
// Temporary Hidden States
////////////////////////////////////////////////////////////////////////////////
TempStateCuda::TempStateCuda(size_t size, size_t block_size)
    : BaseTempStates(size, block_size)
/*
 */
{
    // Allocate memory on the GPU using cudaMalloc
    this->allocate_memory();
}

TempStateCuda::TempStateCuda() : BaseTempStates() {}

TempStateCuda::~TempStateCuda()
/*
 */
{
    cudaFree(this->d_tmp_1);
    cudaFree(this->d_tmp_2);
}

void TempStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_tmp_1, this->tmp_1.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_tmp_2, this->tmp_2.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void TempStateCuda::allocate_memory()
/*
 */
{
    cudaMalloc(&this->d_tmp_1, size * sizeof(float));
    cudaMalloc(&this->d_tmp_2, size * sizeof(float));
}

void TempStateCuda::to_host() {
    cudaMemcpy(this->tmp_1.data(), this->d_tmp_1, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->tmp_2.data(), this->d_tmp_2, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Copying device to host.");
    }
}

////////////////////////////////////////////////////////////////////////////////
// Backward States
////////////////////////////////////////////////////////////////////////////////

BackwardStateCuda::BackwardStateCuda() {}
BackwardStateCuda::~BackwardStateCuda() {
    cudaFree(this->d_mu_a);
    cudaFree(this->d_jcb);
}

void BackwardStateCuda::allocate_memory()
/*
 */
{
    this->mu_a.resize(this->size, 0);
    this->jcb.resize(this->size, 0);
    cudaMalloc(&this->d_mu_a, this->size * sizeof(float));
    cudaMalloc(&this->d_jcb, this->size * sizeof(float));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Device memory allocation.");
    }
}

void BackwardStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_a, this->mu_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_jcb, this->jcb.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Copying host to device.");
    }
}

void BackwardStateCuda::to_host()
/*
 */
{
    cudaMemcpy(this->mu_a.data(), this->d_mu_a, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb.data(), this->d_jcb, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Copying device to host.");
    }
}

////////////////////////////////////////////////////////////////////////////////
// Observation
////////////////////////////////////////////////////////////////////////////////

ObservationCuda::ObservationCuda() {}
ObservationCuda::~ObservationCuda() {
    cudaFree(d_mu_obs);
    cudaFree(d_var_obs);
    cudaFree(d_selected_idx);
}

void ObservationCuda::allocate_memory() {
    cudaMalloc(&this->d_mu_obs, this->size * sizeof(float));
    cudaMalloc(&this->d_var_obs, this->size * sizeof(float));

    if (this->idx_size != 0) {
        cudaMalloc(&this->d_selected_idx, this->idx_size * sizeof(int));
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Device memory allocation.");
    }
}

void ObservationCuda::to_device() {
    cudaMemcpy(this->d_mu_obs, this->mu_obs.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_obs, this->var_obs.data(),
               this->size * sizeof(float), cudaMemcpyHostToDevice);
    if (this->idx_size != 0) {
        cudaMemcpy(this->d_selected_idx, this->selected_idx.data(),
                   this->size * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Copying host to device.");
    }
}

void ObservationCuda::to_host() {
    cudaMemcpy(this->mu_obs.data(), this->d_mu_obs, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_obs.data(), this->d_var_obs,
               this->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->selected_idx.data(), this->d_selected_idx,
               this->size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Copying device to host.");
    }
}

////////////////////////////////////////////////////////////////////////////////
// LSTM states
////////////////////////////////////////////////////////////////////////////////
LSTMStateCuda::LSTMStateCuda() {}
LSTMStateCuda::LSTMStateCuda(size_t num_states, size_t num_inputs)
    : BaseLSTMStates(num_states, num_inputs)
/*
 */
{
    this->allocate_memory();
}

LSTMStateCuda::~LSTMStateCuda()
/*
 */
{
    cudaFree(d_mu_ha);
    d_mu_ha = nullptr;
    cudaFree(d_var_ha);
    d_var_ha = nullptr;
    cudaFree(d_mu_f_ga);
    d_mu_f_ga = nullptr;
    cudaFree(d_var_f_ga);
    d_var_f_ga = nullptr;
    cudaFree(d_jcb_f_ga);
    d_jcb_f_ga = nullptr;
    cudaFree(d_mu_i_ga);
    d_mu_i_ga = nullptr;
    cudaFree(d_var_i_ga);
    d_var_i_ga = nullptr;
    cudaFree(d_jcb_i_ga);
    d_jcb_i_ga = nullptr;
    cudaFree(d_mu_c_ga);
    d_mu_c_ga = nullptr;
    cudaFree(d_var_c_ga);
    d_var_c_ga = nullptr;
    cudaFree(d_jcb_c_ga);
    d_jcb_c_ga = nullptr;
    cudaFree(d_mu_o_ga);
    d_mu_o_ga = nullptr;
    cudaFree(d_var_o_ga);
    d_var_o_ga = nullptr;
    cudaFree(d_jcb_o_ga);
    d_jcb_o_ga = nullptr;
    cudaFree(d_mu_ca);
    d_mu_ca = nullptr;
    cudaFree(d_var_ca);
    d_var_ca = nullptr;
    cudaFree(d_jcb_ca);
    d_jcb_ca = nullptr;
    cudaFree(d_mu_c);
    d_mu_c = nullptr;
    cudaFree(d_var_c);
    d_var_c = nullptr;
    cudaFree(d_mu_c_prev);
    d_mu_c_prev = nullptr;
    cudaFree(d_var_c_prev);
    d_var_c_prev = nullptr;
    cudaFree(d_mu_h_prev);
    d_mu_h_prev = nullptr;
    cudaFree(d_var_h_prev);
    d_var_h_prev = nullptr;
    cudaFree(d_cov_i_c);
    d_cov_i_c = nullptr;
    cudaFree(d_cov_o_tanh_c);
    d_cov_o_tanh_c = nullptr;
}

void LSTMStateCuda::set_num_states(size_t num_states, size_t num_inputs)
/*
 */
{
    this->num_states = num_states;
    this->num_inputs = num_inputs;
    this->reset_zeros();
    this->allocate_memory();
}

void LSTMStateCuda::allocate_memory()
/*
 */
{
    size_t size = num_states * sizeof(float);
    size_t size_ha = (num_states + num_inputs) * sizeof(float);

    cudaMalloc((void **)&d_mu_ha, size_ha);
    cudaMalloc((void **)&d_var_ha, size_ha);

    cudaMalloc((void **)&d_mu_f_ga, size);
    cudaMalloc((void **)&d_var_f_ga, size);
    cudaMalloc((void **)&d_jcb_f_ga, size);

    cudaMalloc((void **)&d_mu_i_ga, size);
    cudaMalloc((void **)&d_var_i_ga, size);
    cudaMalloc((void **)&d_jcb_i_ga, size);

    cudaMalloc((void **)&d_mu_c_ga, size);
    cudaMalloc((void **)&d_var_c_ga, size);
    cudaMalloc((void **)&d_jcb_c_ga, size);

    cudaMalloc((void **)&d_mu_o_ga, size);
    cudaMalloc((void **)&d_var_o_ga, size);
    cudaMalloc((void **)&d_jcb_o_ga, size);

    cudaMalloc((void **)&d_mu_ca, size);
    cudaMalloc((void **)&d_var_ca, size);
    cudaMalloc((void **)&d_jcb_ca, size);

    cudaMalloc((void **)&d_mu_c, size);
    cudaMalloc((void **)&d_var_c, size);

    cudaMalloc((void **)&d_mu_c_prev, size);
    cudaMalloc((void **)&d_var_c_prev, size);

    cudaMalloc((void **)&d_mu_h_prev, size);
    cudaMalloc((void **)&d_var_h_prev, size);

    cudaMalloc((void **)&d_cov_i_c, size);
    cudaMalloc((void **)&d_cov_o_tanh_c, size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Device memory allocation.");
    }
}

void LSTMStateCuda::to_device() {
    // Copy mu_ha and var_ha
    cudaMemcpy(d_mu_ha, this->mu_ha.data(),
               (this->num_states + this->num_inputs) * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_ha, this->var_ha.data(),
               (this->num_states + this->num_inputs) * sizeof(float),
               cudaMemcpyHostToDevice);

    // Copy mu_f_ga and var_f_ga
    cudaMemcpy(d_mu_f_ga, this->mu_f_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_f_ga, this->var_f_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb_f_ga, this->jcb_f_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mu_i_ga, var_i_ga, and jcb_i_ga
    cudaMemcpy(d_mu_i_ga, this->mu_i_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_i_ga, this->var_i_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb_i_ga, this->jcb_i_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mu_c_ga, var_c_ga, and jcb_c_ga
    cudaMemcpy(d_mu_c_ga, this->mu_c_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_c_ga, this->var_c_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb_c_ga, this->jcb_c_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mu_o_ga, var_o_ga, and jcb_o_ga
    cudaMemcpy(d_mu_o_ga, this->mu_o_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_o_ga, this->var_o_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb_o_ga, this->jcb_o_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mu_ca, var_ca, and jcb_ca
    cudaMemcpy(d_mu_ca, this->mu_ca.data(), this->num_states * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_ca, this->var_ca.data(), this->num_states * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb_ca, this->jcb_ca.data(), this->num_states * sizeof(float),
               cudaMemcpyHostToDevice);

    // Copy mu_c, var_c, mu_c_prev, and var_c_prev
    cudaMemcpy(d_mu_c, this->mu_c.data(), this->num_states * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_c, this->var_c.data(), this->num_states * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_mu_c_prev, this->mu_c_prev.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_c_prev, this->var_c_prev.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mu_h_prev and var_h_prev
    cudaMemcpy(d_mu_h_prev, this->mu_h_prev.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_h_prev, this->var_h_prev.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy cov_i_c and cov_o_tanh_c
    cudaMemcpy(d_cov_i_c, this->cov_i_c.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cov_o_tanh_c, this->cov_o_tanh_c.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Host to device.");
    }
}

void LSTMStateCuda::to_host()
/*
 */
{
    // Copy back mu_ha and var_ha
    cudaMemcpy(this->mu_ha.data(), d_mu_ha,
               (this->num_states + this->num_inputs) * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_ha.data(), d_var_ha,
               (this->num_states + this->num_inputs) * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Copy back mu_f_ga and var_f_ga
    cudaMemcpy(this->mu_f_ga.data(), d_mu_f_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_f_ga.data(), d_var_f_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back jcb_f_ga
    cudaMemcpy(this->jcb_f_ga.data(), d_jcb_f_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back mu_i_ga, var_i_ga, and jcb_i_ga
    cudaMemcpy(this->mu_i_ga.data(), d_mu_i_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_i_ga.data(), d_var_i_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb_i_ga.data(), d_jcb_i_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back mu_c_ga, var_c_ga, and jcb_c_ga
    cudaMemcpy(this->mu_c_ga.data(), d_mu_c_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_c_ga.data(), d_var_c_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb_c_ga.data(), d_jcb_c_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back mu_o_ga, var_o_ga, and jcb_o_ga
    cudaMemcpy(this->mu_o_ga.data(), d_mu_o_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_o_ga.data(), d_var_o_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb_o_ga.data(), d_jcb_o_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back mu_ca, var_ca, and jcb_ca
    cudaMemcpy(this->mu_ca.data(), d_mu_ca, this->num_states * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_ca.data(), d_var_ca, this->num_states * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb_ca.data(), d_jcb_ca, this->num_states * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Copy back mu_c, var_c, mu_c_prev, and var_c_prev
    cudaMemcpy(this->mu_c.data(), d_mu_c, this->num_states * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_c.data(), d_var_c, this->num_states * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mu_c_prev.data(), d_mu_c_prev,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_c_prev.data(), d_var_c_prev,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back mu_h_prev and var_h_prev
    cudaMemcpy(this->mu_h_prev.data(), d_mu_h_prev,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_h_prev.data(), d_var_h_prev,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back cov_i_c and cov_o_tanh_c
    cudaMemcpy(this->cov_i_c.data(), d_cov_i_c,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cov_o_tanh_c.data(), d_cov_o_tanh_c,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Device to host.");
    }
}
