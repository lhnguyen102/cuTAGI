///////////////////////////////////////////////////////////////////////////////
// File:         fc_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      November 28, 2023
// Updated:      November 28, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>

#include "base_layer.h"

class FullConnectedCuda : public BaseLayer {
   public:
    float gain_w;
    float gain_b;
    std::string init_method;
};