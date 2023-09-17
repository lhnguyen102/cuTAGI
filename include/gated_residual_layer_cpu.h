///////////////////////////////////////////////////////////////////////////////
// File:         gated_residual_layer_cpu.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 17, 2023
// Updated:      September 17, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>

struct GatedResidualProp {
    int input_size, hidden_size, output_size, context_size;
    int num_weights = 0, num_biases = 0;
    bool residual = false;
    std::vector<int> w_pos, b_pos;
};
