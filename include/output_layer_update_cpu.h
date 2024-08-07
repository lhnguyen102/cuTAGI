///////////////////////////////////////////////////////////////////////////////
// File:         output_layer_update_cpu.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 22, 2023
// Updated:      November 25, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <math.h>

#include <thread>
#include <vector>

#include "base_output_updater.h"
#include "data_struct.h"
// #include "struct_var.h"

void update_output_delta_z(BaseHiddenStates &last_layer_states,
                           std::vector<float> &obs, std::vector<float> &var_obs,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var);

void update_selected_output_delta_z(BaseHiddenStates &last_layer_states,
                                    std::vector<float> &obs,
                                    std::vector<float> &var_obs,
                                    std::vector<int> &selected_idx,
                                    std::vector<float> &delta_mu,
                                    std::vector<float> &delta_var);