///////////////////////////////////////////////////////////////////////////////
// File:         base_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      November 29, 2023
// Updated:      December 24, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <memory>
#include <vector>

#include "base_layer.h"
#include "data_struct_cuda.cuh"
#include "struct_var.h"

__global__ void fill_bwd_states_on_device(float const *mu_a_in,
                                          float const *jcb_in, int size,
                                          float *mu_a, float *jcb);

__global__ void fill_output_states_on_device(float const *mu_z,
                                             float const *var_z, int size,
                                             float *mu_a, float *jcb,
                                             float *var_a);

class BaseLayerCuda : public BaseLayer {
   public:
    float *d_mu_w = nullptr;
    float *d_var_w = nullptr;
    float *d_mu_b = nullptr;
    float *d_var_b = nullptr;
    float *d_delta_mu_w = nullptr;
    float *d_delta_var_w = nullptr;
    float *d_delta_mu_b = nullptr;
    float *d_delta_var_b = nullptr;
    unsigned int num_cuda_threads = 32;

    BaseLayerCuda();

    ~BaseLayerCuda();

    // Delete copy constructor and copy assignment
    BaseLayerCuda(const BaseLayerCuda &) = delete;
    BaseLayerCuda &operator=(const BaseLayerCuda &) = delete;

    // Optionally implement move constructor and move assignment
    BaseLayerCuda(BaseLayerCuda &&) = default;
    BaseLayerCuda &operator=(BaseLayerCuda &&) = default;

    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    void update_weights() override;

    void update_biases() override;

    virtual std::unique_ptr<BaseLayer> to_host();

   protected:
    virtual void allocate_param_memory();
    virtual void params_to_device();
    virtual void params_to_host();
};
