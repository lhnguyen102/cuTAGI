///////////////////////////////////////////////////////////////////////////////
// File:         data_struct_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 10, 2023
// Updated:      December 11, 2023
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
    // Allocate memory on the GPU using cudaMalloc
    cudaMalloc(&d_mu_z, size * sizeof(float));
    cudaMalloc(&d_var_z, size * sizeof(float));
    cudaMalloc(&d_mu_a, size * sizeof(float));
    cudaMalloc(&d_var_a, size * sizeof(float));
    cudaMalloc(&d_jcb, size * sizeof(float));
}

HiddenStateCuda::HiddenStateCuda() : BaseHiddenStates() {}

HiddenStateCuda::~HiddenStateCuda()
/*
Free GPU memory using cudaFree
*/
{
    cudaFree(d_mu_z);
    cudaFree(d_var_z);
    cudaFree(d_mu_a);
    cudaFree(d_var_a);
    cudaFree(d_jcb);
}

void HiddenStateCuda::to_device()
/*
 */
{
    cudaMemcpy(d_mu_z, this->mu_z.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_z, this->var_z.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu_a, this->mu_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_a, this->var_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb, this->jcb.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
}

////////////////////////////////////////////////////////////////////////////////
// Delta Hidden States
////////////////////////////////////////////////////////////////////////////////
DeltaStateCuda::DeltaStateCuda(size_t size, size_t block_size)
    : BaseDeltaStates(size, block_size)
/*
 */
{
    // Allocate memory on the GPU using cudaMalloc
    cudaMalloc(&d_delta_mu, size * sizeof(float));
    cudaMalloc(&d_delta_var, size * sizeof(float));
}

DeltaStateCuda::DeltaStateCuda() : BaseDeltaStates() {}

DeltaStateCuda::~DeltaStateCuda()
/*
 */
{
    cudaFree(d_delta_mu);
    cudaFree(d_delta_var);
}

void DeltaStateCuda::to_device()
/*
 */
{
    cudaMemcpy(d_delta_mu, this->delta_mu.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_var, this->delta_var.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
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
    cudaMalloc(&d_tmp_1, size * sizeof(float));
    cudaMalloc(&d_tmp_2, size * sizeof(float));
}

TempStateCuda::TempStateCuda() : BaseTempStates() {}

TempStateCuda::~TempStateCuda()
/*
 */
{
    cudaFree(d_tmp_1);
    cudaFree(d_tmp_2);
}

void TempStateCuda::to_device()
/*
 */
{
    cudaMemcpy(d_tmp_1, this->tmp_1.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmp_2, this->tmp_2.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
}

////////////////////////////////////////////////////////////////////////////////
// Backward States
////////////////////////////////////////////////////////////////////////////////

BackwardStateCuda::BackwardStateCuda() {}
BackwardStateCuda::~BackwardStateCuda() {
    cudaFree(d_mu_a);
    cudaFree(d_jcb);
}