///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 24, 2024
// Updated:      January 30, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "base_layer.h"

class LayerNorm : public BaseLayer {
   public:
    LayerNorm(const std::vector<int> &normalized_shape, float eps = 1e-5,
              bool bias = true);
    ~LayerNorm();
};