///////////////////////////////////////////////////////////////////////////////
// File:         self_attention_cpu.cpp
// Description:  CPU version for self attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 13, 2023
// Updated:      March 15, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2023 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/self_attention_cpu.h"

void query_key(std::vector<float> &mu_q, std::vector<float> &var_q,
               std::vector<float> &mu_k, std::vector<float> &var_k,
               int batch_size, int num_heads, int time_step, int head_size,
               std::vector<float> &mu_qk, std::vector<float> &var_qk) {
    int idx_q, idx_k;
    for (int i = 0; i < batch_size; i < batch_size) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < time_step; k++) {
                for (int l = 0; l < head_size; l++) {
                    idx_q = i * batch_size + j * time_step + k * num_heads + l;
                    idx_k = i * batch_size + j * time_step + k + l * num_heads;
                    mu_qk[idx_q] = mu_q[idx_q] * mu_k[idx_k];
                    var_qk[idx_q] = var_q[idx_q] * var_k[idx_k] +
                                    var_q[idx_q] * powf(mu_k[idx_k], 2) +
                                    var_k[idx_k] * powf(mu_q[idx_q], 2);
                }
            }
        }
    }
}

void self_attention_forward_cpu(MultiHeadAttention &mha_state, int z_pos,
                                int batch_size, int num_heads, int time_step,
                                int head_size, float omega_tol)
/*Multi-head self-attentiopn mecanism.

Args:
    mth_state: State of multi-heads self attention

*/
{
    // query x key
    query_key(mha_state.mu_q, mha_state.var_q, mha_state.mu_k, mha_state.var_k,
              batch_size, num_heads, time_step, head_size, mha_state.mu_att,
              mha_state.var_att);

    // // Apply remax on the product of querry and key
    // remax_cpu_v2(mha_state.mu_att, mha_state.var_att, mha_state.remax.mu_m,
    //              mha_state.remax.var_m, mha_state.remax.J_m,
    //              mha_state.remax.mu_log, mha_state.remax.var_log,
    //              mha_state.remax.mu_sum, mha_state.remax.var_sum,
    //              mha_state.remax.mu_logsum, mha_state.remax.var_logsum,
    //              mha_state.remax.cov_log_logsum, mha_state.mu_att,
    //              mha_state.var_att, z_pos, time_step, batch_size, omega_tol);
}