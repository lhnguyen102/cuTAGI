///////////////////////////////////////////////////////////////////////////////
// File:         base_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      November 29, 2023
// Updated:      December 04, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <vector>

#include "base_layer.h"
#include "struct_var.h"

class BaseLayerCuda : public BaseLayer {
   public:
    float *d_mu_w;
    float *d_var_w;
    float *d_mu_b;
    float *d_var_b;
    float *d_jcb;
    float *d_delta_mu_w;
    float *d_delta_var_w;
    float *d_delta_mu_b;
    float *d_delta_var_b;

    BaseLayerCuda() {}

    ~BaseLayerCuda() {
        cudaFree(d_mu_w);
        cudaFree(d_var_w);
        cudaFree(d_mu_b);
        cudaFree(d_var_b);
        cudaFree(d_jcb);
        cudaFree(d_delta_mu_w);
        cudaFree(d_delta_var_w);
        cudaFree(d_delta_mu_b);
        cudaFree(d_delta_var_b);
    }
};
