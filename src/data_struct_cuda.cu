///////////////////////////////////////////////////////////////////////////////
// File:         data_struct_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 10, 2023
// Updated:      March 18, 2024
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