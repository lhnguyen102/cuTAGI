///////////////////////////////////////////////////////////////////////////////
// File:         base_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 13, 2023
// Updated:      December 13, 2023
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