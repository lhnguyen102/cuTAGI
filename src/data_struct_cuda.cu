///////////////////////////////////////////////////////////////////////////////
// File:         data_struct_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 10, 2023
// Updated:      December 15, 2023
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
    cudaFree(this->d_mu_z);
    cudaFree(this->d_var_z);
    cudaFree(this->d_mu_a);
    cudaFree(this->d_var_a);
    cudaFree(this->d_jcb);
}

void HiddenStateCuda::allocate_memory() {
    // Allocate memory on the GPU using cudaMalloc
    cudaMalloc(&this->d_mu_z, size * sizeof(float));
    cudaMalloc(&this->d_var_z, size * sizeof(float));
    cudaMalloc(&this->d_mu_a, size * sizeof(float));
    cudaMalloc(&this->d_var_a, size * sizeof(float));
    cudaMalloc(&this->d_jcb, size * sizeof(float));
};

void HiddenStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_z, this->mu_z.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_z, this->var_z.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mu_a, this->mu_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_a, this->var_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_jcb, this->jcb.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void HiddenStateCuda::to_host()
/*
 */
{
    cudaMemcpy(this->mu_z.data(), this->d_mu_z,
               this->mu_z.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_z.data(), this->d_var_z,
               this->var_z.size() * sizeof(float), cudaMemcpyDeviceToHost);
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
}

void DeltaStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_delta_mu, this->delta_mu.data(),
               this->size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_delta_var, this->delta_var.data(),
               this->size * sizeof(float), cudaMemcpyHostToDevice);
}

void DeltaStateCuda::to_host()
/*
 */
{
    cudaMemcpy(this->delta_mu.data(), this->d_delta_mu,
               this->delta_mu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->delta_var.data(), this->d_delta_var,
               this->delta_var.size() * sizeof(float), cudaMemcpyDeviceToHost);
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
    cudaMalloc(&this->d_mu_a, this->size * sizeof(float));
    cudaMalloc(&this->d_jcb, this->size * sizeof(float));
}

void BackwardStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_a, this->mu_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_jcb, this->jcb.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void BackwardStateCuda::to_host()
/*
 */
{
    cudaMemcpy(this->mu_a.data(), this->d_mu_a, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb.data(), this->d_jcb, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);
}