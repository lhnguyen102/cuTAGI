///////////////////////////////////////////////////////////////////////////////
// File:         struct_var.h
// Description:  Header file for struct variable in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 17, 2023
// Updated:      March 17, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2023 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/struct_var.h"

void init_remax(Remax &remax_state, std::vector<int> &num_states,
                int batch_size)
/**/
{
    remax_state.z_pos.resize(num_states.size(), 0);
    remax_state.z_sum_pos.resize(batch_size, 0);
    int total_state = num_states[0];
    int total_sum_state = batch_size;
    for (int i = 1; i < num_states.size(); i++) {
        remax_state.z_pos[i] =
            remax_state.z_pos[i - 1] + num_states[i - 1] * batch_size;
        remax_state.z_sum_pos[i] = remax_state.z_sum_pos[i - 1] + batch_size;
        total_state += num_states[i] * batch_size;
        total_sum_state += batch_size;
    }
    remax_state.mu_m.resize(total_state, 0);
    remax_state.var_m.resize(total_state, 0);
    remax_state.J_m.resize(total_state, 0);
    remax_state.mu_log.resize(total_state, 0);
    remax_state.var_log.resize(total_state, 0);
    remax_state.mu_sum.resize(total_sum_state, 0);
    remax_state.var_sum.resize(total_sum_state, 0);
}

void init_multi_head_attention(MultiHeadAttention &mha_state,
                               std::vector<int> &num_states,
                               std::vector<int> &num_heads,
                               std::vector<int> &time_step,
                               std::vector<int> &head_size, int batch_size)
/**/
{}