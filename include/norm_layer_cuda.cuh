///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 24, 2024
// Updated:      January 29, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "base_layer_cuda.cuh"

class LayerNormCuda : public : BaseLayerCuda {
   public:
    LayerNormCuda(const std::vector<int>& normalized_shape, float eps = 1e-5);
    ~LayerNormCuda;
};