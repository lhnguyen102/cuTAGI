///////////////////////////////////////////////////////////////////////////////
// File:         debugger.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 14, 2024
// Updated:      February 14, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "sequential.h"

void debug_forward(Sequential &test_model, Sequential &ref_model,
                   const std::vector<float> &mu_x,
                   const std::vector<float> &var_x);
void debug_backward(Sequential &test_model, Sequential &ref_model,
                    OutputUpdater &output_updater, std::vector<float> &y_batch,
                    std::vector<float> &var_obs,
                    std::vector<int> &idx_ud_batch);
