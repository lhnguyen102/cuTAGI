///////////////////////////////////////////////////////////////////////////////
// File:         base_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      November 29, 2023
// Updated:      December 13, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
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
    float *d_mu_w;
    float *d_var_w;
    float *d_mu_b;
    float *d_var_b;
    float *d_delta_mu_w;
    float *d_delta_var_w;
    float *d_delta_mu_b;
    float *d_delta_var_b;
    unsigned int num_cuda_threads = 16;
    // TODO does it overide the base layer
    BackwardStateCuda bwd_states;

    BaseLayerCuda();

    ~BaseLayerCuda();
    using BaseLayer::forward;
    using BaseLayer::param_backward;
    using BaseLayer::state_backward;

    virtual void forward(HiddenStateCuda &input_states,
                         HiddenStateCuda &output_states,
                         TempStateCuda &temp_states);

    virtual void state_backward(BackwardStateCuda &next_bwd_states,
                                DeltaStateCuda &input_delta_states,
                                DeltaStateCuda &output_delta_states,
                                TempStateCuda &temp_states);

    virtual void param_backward(BackwardStateCuda &bwd_states,
                                DeltaStateCuda &delta_states,
                                TempStateCuda &temp_states);

    void update_weights() override;

    void update_biases() override;
};
