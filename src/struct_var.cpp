///////////////////////////////////////////////////////////////////////////////
// File:         struct_var.h
// Description:  Header file for struct variable in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 17, 2023
// Updated:      May 03, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "../include/struct_var.h"

void init_remax_state_pos(Remax &remax_state, std::vector<int> &num_states,
                          std::vector<int> &num_sum_states)
/**/
{
    int num_layers = num_states.size();
    remax_state.z_pos.resize(num_layers, 0);
    remax_state.z_sum_pos.resize(num_layers, 0);
    for (int i = 1; i < num_layers; i++) {
        remax_state.z_pos[i] = remax_state.z_pos[i - 1] + num_states[i - 1];
        remax_state.z_sum_pos[i] =
            remax_state.z_sum_pos[i - 1] + num_sum_states[i - 1];
    }
}

void init_remax_states(Remax &remax_state, int tot_num_states,
                       int tot_num_sum_states) {
    remax_state.mu_m.resize(tot_num_states, 0);
    remax_state.var_m.resize(tot_num_states, 0);
    remax_state.J_m.resize(tot_num_states, 0);
    remax_state.mu_log.resize(tot_num_states, 0);
    remax_state.var_log.resize(tot_num_states, 0);
    remax_state.mu_sum.resize(tot_num_sum_states, 0);
    remax_state.var_sum.resize(tot_num_sum_states, 0);
    remax_state.mu_logsum.resize(tot_num_sum_states, 0);
    remax_state.var_logsum.resize(tot_num_sum_states, 0);
    remax_state.cov_log_logsum.resize(tot_num_states, 0);
    remax_state.cov_m_a.resize(tot_num_states, 0);
    remax_state.cov_m_a_check.resize(tot_num_states, 0);
}

void init_multi_head_attention_states(MultiHeadAttentionState &mha_state,
                                      MultiHeadAttentionProp &mha_prop,
                                      int batch_size)
/**/
{
    // Initalize the self-attention state
    int num_layers = mha_prop.num_heads.size();
    mha_state.qkv_pos.resize(num_layers, 0);
    mha_state.att_pos.resize(num_layers, 0);
    mha_state.in_proj_pos.resize(num_layers, 0);
    std::vector<int> num_remax_states(num_layers, 0);
    std::vector<int> num_remax_sum_states(num_layers, 0);
    int qkv_size, att_size, in_proj_size, mha_i;
    int buffer_size = 0;
    int buffer_size_sv = 0;
    int tot_remax_states = 0, tot_remax_sum_states = 0, max_size;
    int tot_qkv_size = 0;
    int tot_att_size = 0;
    for (int i = 0; i < num_layers; i++) {
        // State size
        qkv_size = batch_size * mha_prop.num_heads[i] * mha_prop.timestep[i] *
                   mha_prop.head_size[i];
        att_size = batch_size * mha_prop.num_heads[i] * mha_prop.timestep[i] *
                   mha_prop.timestep[i];
        in_proj_size =
            3 * batch_size * mha_prop.num_heads[i] * mha_prop.head_size[i];
        buffer_size_sv = std::max(std::max(buffer_size_sv, qkv_size), att_size);
        buffer_size = std::max(buffer_size, in_proj_size);

        // Set all state values to zero
        if (i < num_layers) {
            mha_state.qkv_pos[i + 1] += qkv_size;
            mha_state.att_pos[i + 1] += att_size;
            mha_state.in_proj_pos[i + 1] += in_proj_size;
        }

        // Remax state
        num_remax_states[i] = batch_size * mha_prop.num_heads[i] *
                              mha_prop.timestep[i] * mha_prop.timestep[i];
        num_remax_sum_states[i] =
            batch_size * mha_prop.num_heads[i] * mha_prop.timestep[i];
        tot_remax_states += batch_size * mha_prop.num_heads[i] *
                            mha_prop.timestep[i] * mha_prop.timestep[i];
        tot_remax_sum_states +=
            batch_size * mha_prop.num_heads[i] * mha_prop.timestep[i];
        tot_qkv_size += qkv_size;
        tot_att_size += att_size;
    }
    // Initalize all states required in the backward pass
    mha_state.mu_q.resize(tot_qkv_size, 0);
    mha_state.var_q.resize(tot_qkv_size, 0);
    mha_state.mu_k.resize(tot_qkv_size, 0);
    mha_state.var_k.resize(tot_qkv_size, 0);
    mha_state.mu_v.resize(tot_qkv_size, 0);
    mha_state.var_v.resize(tot_qkv_size, 0);
    mha_state.mu_att_score.resize(tot_att_size, 0);
    mha_state.var_att_score.resize(tot_att_size, 0);
    mha_state.mu_out_proj.resize(tot_qkv_size, 0);
    mha_state.var_out_proj.resize(tot_qkv_size, 0);
    mha_state.J_out_proj.resize(tot_qkv_size, 0);
    mha_state.mu_mqk.resize(tot_att_size, 0);
    mha_state.var_mqk.resize(tot_att_size, 0);
    mha_state.J_mqk.resize(tot_att_size, 0);

    // Initialize buffer states
    mha_state.mu_sv.resize(buffer_size_sv, 0);
    mha_state.var_sv.resize(buffer_size_sv, 0);
    mha_state.mu_in_proj.resize(buffer_size, 0);
    mha_state.var_in_proj.resize(buffer_size, 0);

    // Initialize the remax state
    init_remax_states(*mha_state.remax, tot_remax_states, tot_remax_sum_states);
    init_remax_state_pos(*mha_state.remax, num_remax_states,
                         num_remax_sum_states);
}

void init_multi_head_attention_delta_states(
    MultiHeadAttentionDelta &delta_mha_state, MultiHeadAttentionProp &mha_prop,
    int batch_size) {
    // Initalize the self-attention state
    int num_layers = mha_prop.num_heads.size();
    int qkv_size, att_size, in_proj_size;
    int buffer_size = 0;
    int buffer_size_sv = 0;
    int tot_qkv_size = 0;
    for (int i = 0; i < num_layers; i++) {
        // State size
        qkv_size = batch_size * mha_prop.num_heads[i] * mha_prop.timestep[i] *
                   mha_prop.head_size[i];
        att_size = batch_size * mha_prop.num_heads[i] * mha_prop.timestep[i] *
                   mha_prop.timestep[i];
        in_proj_size =
            3 * batch_size * mha_prop.num_heads[i] * mha_prop.head_size[i];
        buffer_size_sv = std::max(std::max(buffer_size_sv, qkv_size), att_size);
        buffer_size = std::max(buffer_size, in_proj_size);
        tot_qkv_size += qkv_size;
    }
    // Initialize all delta states required in the backward pass
    delta_mha_state.delta_mu_in_proj.resize(tot_qkv_size * 3, 0);
    delta_mha_state.delta_var_in_proj.resize(tot_qkv_size * 3, 0);
    delta_mha_state.delta_mu_buffer.resize(buffer_size, 0);
    delta_mha_state.delta_var_buffer.resize(buffer_size, 0);
    delta_mha_state.delta_mu_k.resize(buffer_size_sv, 0);
    delta_mha_state.delta_var_k.resize(buffer_size_sv, 0);
    delta_mha_state.delta_mu_v.resize(buffer_size_sv, 0);
    delta_mha_state.delta_var_v.resize(buffer_size_sv, 0);
    delta_mha_state.delta_mu_q.resize(buffer_size_sv, 0);
    delta_mha_state.delta_var_q.resize(buffer_size_sv, 0);
    delta_mha_state.delta_mu_r.resize(buffer_size_sv, 0);
    delta_mha_state.delta_var_r.resize(buffer_size_sv, 0);
    delta_mha_state.delta_mu_att_score.resize(buffer_size_sv, 0);
    delta_mha_state.delta_var_att_score.resize(buffer_size_sv, 0);
}