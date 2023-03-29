///////////////////////////////////////////////////////////////////////////////
// File:         self_attention_cpu.cpp
// Description:  CPU version for self attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 13, 2023
// Updated:      March 18, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/self_attention_cpu.h"

void query_key(std::vector<float> &mu_q, std::vector<float> &var_q,
               std::vector<float> &mu_k, std::vector<float> &var_k, int qk_pos,
               int qkv_pos, int batch_size, int num_heads, int time_step,
               int head_size, std::vector<float> &mu_qk,
               std::vector<float> &var_qk)
/* 4D matrix multiplication of qerry matrix with key matrix*/
{
    int idx_q, idx_k, idx_qk;
    float sum_mu, sum_var;
    for (int i = 0; i < batch_size; i < batch_size) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < time_step; k++) {
                for (int l = 0; l < time_step; l++) {
                    sum_mu = 0;
                    sum_var = 0;
                    for (int m = 0; m < head_size; m++) {
                        idx_q = i * num_heads * time_step * head_size +
                                j * time_step * head_size + k * head_size + m +
                                qkv_pos;
                        idx_k = i * num_heads * time_step * head_size +
                                j * time_step * head_size + l + m * time_step +
                                qkv_pos;

                        sum_mu += mu_q[idx_q] * mu_k[idx_k];
                        sum_var += var_q[idx_q] * var_k[idx_k] +
                                   var_q[idx_q] * powf(mu_k[idx_k], 2) +
                                   var_k[idx_k] * powf(mu_q[idx_q], 2);
                    }
                    idx_qk = i * num_heads * time_step * time_step +
                             j * time_step * time_step + k * time_step + l +
                             qk_pos;
                    mu_qk[idx_qk] = sum_mu;
                    var_qk[idx_qk] = sum_var;
                }
            }
        }
    }
}

void self_attention_forward_cpu(MultiHeadAttentionState &mha_state,
                                MultiHeadAttentionProp &mha_prop,
                                int batch_size, float omega_tol, int l)
/*Multi-head self-attentiopn mecanism.

Args:
    mth_state: State of multi-heads self attention

*/
{
    int num_heads = mha_prop.num_heads[l];
    int timestep = mha_prop.timestep[l];
    int head_size = mha_prop.head_size[l];
    int att_pos = mha_state.att_pos[l];
    int qkv_pos = mha_state.qkv_pos[l];
    int z_remax_pos = mha_state.remax.z_pos[l];
    int z_sum_remax_pos = mha_state.remax.z_sum_pos[l];

    // query x key
    query_key(mha_state.mu_q, mha_state.var_q, mha_state.mu_k, mha_state.var_k,
              att_pos, qkv_pos, batch_size, num_heads, timestep, head_size,
              mha_state.mu_att_score, mha_state.var_att_score);

    // Apply remax on the product of querry and key
    remax_cpu_v2(mha_state.mu_att_score, mha_state.var_att_score,
                 mha_state.remax.mu_m, mha_state.remax.var_m,
                 mha_state.remax.J_m, mha_state.remax.mu_log,
                 mha_state.remax.var_log, mha_state.remax.mu_sum,
                 mha_state.remax.var_sum, mha_state.remax.mu_logsum,
                 mha_state.remax.var_logsum, mha_state.remax.cov_log_logsum,
                 mha_state.mu_att_score, mha_state.var_att_score, att_pos,
                 z_remax_pos, z_sum_remax_pos, timestep, batch_size, omega_tol);
}