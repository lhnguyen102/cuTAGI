///////////////////////////////////////////////////////////////////////////////
// File:         output_layer_update_cpu.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 22, 2023
// Updated:      November 17, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <math.h>

#include <thread>
#include <vector>

#include "struct_var.h"

void compute_delta_z_output(std::vector<float> &mu_a, std::vector<float> &var_a,
                            std::vector<float> &var_z, std::vector<float> &jcb,
                            std::vector<float> &obs, std::vector<float> &var_v,
                            int start_chunk, int end_chunk,
                            std::vector<float> &delta_mu,
                            std::vector<float> &delta_var);

void compute_delta_z_output_mp(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &var_z, std::vector<float> &jcb, std::vector<float> &obs,
    std::vector<float> &var_v, int n, unsigned int num_threads,
    std::vector<float> &delta_mu, std::vector<float> &delta_var);

void compute_delta_z_output_with_indices(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &var_z, std::vector<float> &jcb, std::vector<float> &obs,
    std::vector<float> &var_v, std::vector<int> &ud_idx, int n_obs, int n_enc,
    int start_chunk, int end_chunk, std::vector<float> &delta_mu,
    std::vector<float> &delta_var);

void compute_delta_z_output_with_indices_mp(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &var_z, std::vector<float> &jcb, std::vector<float> &obs,
    std::vector<float> &var_v, std::vector<int> &ud_idx, int n_obs, int n_enc,
    int n, unsigned int num_threads, std::vector<float> &delta_mu,
    std::vector<float> &delta_var);

void update_output_delta_z(HiddenStates &last_layer_states,
                           std::vector<float> &obs, std::vector<float> &var_obs,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var);