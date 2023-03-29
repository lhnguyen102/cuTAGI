///////////////////////////////////////////////////////////////////////////////
// File:         struct_var.h
// Description:  Header file for struct variable in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 17, 2023
// Updated:      March 18, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
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
    remax_state.mu_logsum.resize(total_sum_state, 0);
    remax_state.var_logsum.resize(total_sum_state, 0);
    remax_state.cov_log_logsum.resize(total_state, 0);
    remax_state.cov_m_a.resize(total_state, 0);
    remax_state.cov_m_a_check.resize(total_state, 0);
}

void init_multi_head_attention(MultiHeadAttentionState &mha_state,
                               std::vector<int> &num_heads,
                               std::vector<int> &timestep,
                               std::vector<int> &head_size, int batch_size)
/**/
{
    // Initalize the self-attention state
    mha_state.qkv_pos.resize(num_heads.size(), 0);
    mha_state.att_pos.resize(num_heads.size(), 0);
    std::vector<int> num_remax_states(num_heads.size(), 0);
    int qkv_size, att_size;
    for (int i = 0; i < num_heads.size(); i++) {
        qkv_size = batch_size * num_heads[i] * timestep[i] * head_size[i];
        att_size = batch_size * num_heads[i] * timestep[i] * timestep[i];
        mha_state.mu_k.resize(qkv_size, 0);
        mha_state.var_k.resize(qkv_size, 0);
        mha_state.mu_q.resize(qkv_size, 0);
        mha_state.var_q.resize(qkv_size, 0);
        mha_state.mu_v.resize(qkv_size, 0);
        mha_state.var_v.resize(qkv_size, 0);
        mha_state.mu_att_score.resize(att_size, 0);
        mha_state.var_att_score.resize(att_size, 0);
        mha_state.mu_att.resize(qkv_size, 0);
        mha_state.var_att.resize(qkv_size, 0);
        if (i < num_heads.size()) {
            mha_state.qkv_pos[i + 1] += qkv_size;
            mha_state.att_pos[i + 1] += att_size;
        }
        num_remax_states[i] = att_size;
    }
    // Initialize the remax state
    init_remax(mha_state.remax, num_remax_states, batch_size);
}