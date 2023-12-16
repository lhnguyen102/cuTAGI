///////////////////////////////////////////////////////////////////////////////
// File:         param_init.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 15, 2023
// Updated:      December 15, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "net_prop.h"

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_linear(const std::string &init_method, const float gain_w,
                        const float gain_b, const int input_size,
                        const int output_size);