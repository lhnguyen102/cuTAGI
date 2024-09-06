///////////////////////////////////////////////////////////////////////////////
// File:         output_layer_update_cpu.cpu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 22, 2023
// Updated:      November 17, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/output_layer_update_cpu.h"

void update_output_delta_z(BaseHiddenStates &last_layer_states,
                           std::vector<float> &obs, std::vector<float> &var_obs,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = obs.size();
    compute_delta_z_output(last_layer_states.mu_a, last_layer_states.var_a,
                           last_layer_states.jcb, obs, var_obs, start_chunk,
                           end_chunk, delta_mu, delta_var);
}

void update_selected_output_delta_z(BaseHiddenStates &last_layer_states,
                                    std::vector<float> &obs,
                                    std::vector<float> &var_obs,
                                    std::vector<int> &selected_idx,
                                    std::vector<float> &delta_mu,
                                    std::vector<float> &delta_var)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = selected_idx.size();
    int n_enc = selected_idx.size() / last_layer_states.block_size;
    int n_obs = last_layer_states.actual_size;
    compute_selected_delta_z_output(
        last_layer_states.mu_a, last_layer_states.var_a, last_layer_states.jcb,
        obs, var_obs, selected_idx, n_obs, n_enc, start_chunk, end_chunk,
        delta_mu, delta_var);
}