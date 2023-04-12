///////////////////////////////////////////////////////////////////////////////
// File:         self_attention_cpu.cpp
// Description:  CPU version for self attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 13, 2023
// Updated:      April 10, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/self_attention_cpu.h"

void tagi_4d_matrix_mul(std::vector<float> &mu_a, std::vector<float> &var_a,
                        std::vector<float> &mu_b, std::vector<float> &var_b,
                        int a_pos, int b_pos, int ab_pos, int N, int C, int H,
                        int W, int D, std::vector<float> &mu_ab,
                        std::vector<float> &var_ab) {
    int idx_a, idx_b, idx_ab;
    float sum_mu, sum_var, sum_mu_masked, sum_var_masked;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < C; j++) {
            for (int k = 0; k < H; k++) {
                for (int l = 0; l < W; l++) {
                    sum_mu = 0;
                    sum_var = 0;
                    for (int m = 0; m < D; m++) {
                        idx_a = i * C * H * W + j * H * W + k * H + m + a_pos;
                        idx_b = i * C * H * W + j * H * W + l + m * W + b_pos;

                        sum_mu += mu_a[idx_a] * mu_b[idx_b];
                        sum_var += var_a[idx_a] * var_b[idx_b] +
                                   var_a[idx_a] * powf(mu_b[idx_b], 2) +
                                   var_a[idx_a] * powf(mu_b[idx_b], 2);
                    }
                    idx_ab = i * C * H * W + j * H * W + k * W + l + ab_pos;
                    mu_ab[idx_ab] = sum_mu;
                    var_ab[idx_ab] = sum_var;
                }
            }
        }
    }
}

void query_key(std::vector<float> &mu_q, std::vector<float> &var_q,
               std::vector<float> &mu_k, std::vector<float> &var_k, int qkv_pos,
               int batch_size, int num_heads, int timestep, int head_size,
               std::vector<float> &mu_qk, std::vector<float> &var_qk)
/* 4D matrix multiplication of query matrix with key matrix*/
{
    int idx_q, idx_k, idx_qk;
    float sum_mu, sum_var, sum_mu_masked, sum_var_masked;
    for (int i = 0; i < batch_size; i < batch_size) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l < timestep; l++) {
                    sum_mu = 0;
                    sum_var = 0;
                    for (int m = 0; m < head_size; m++) {
                        idx_q = i * num_heads * timestep * head_size +
                                j * timestep * head_size + k * timestep + m +
                                qkv_pos;
                        idx_k = i * num_heads * timestep * head_size +
                                j * timestep * head_size + l + m * timestep +
                                qkv_pos;

                        sum_mu += mu_q[idx_q] * mu_k[idx_k];
                        sum_var += var_q[idx_q] * var_k[idx_k] +
                                   var_q[idx_q] * powf(mu_k[idx_k], 2) +
                                   var_k[idx_k] * powf(mu_q[idx_q], 2);
                    }
                    idx_qk = i * num_heads * timestep * timestep +
                             j * timestep * timestep + k * timestep + l;
                    mu_qk[idx_qk] = sum_mu;
                    var_qk[idx_qk] = sum_var;
                }
            }
        }
    }
}

void mask_query_key(std::vector<float> &mu_qk, std::vector<float> &var_qk,
                    int batch_size, int num_heads, int timestep, int head_size,
                    std::vector<float> &mu_mqk, std::vector<float> &var_mqk) {
    float sum_mu = 0, sum_var = 0;
    int idx_qk, idx_mqk;
    for (int i = 0; i < batch_size; i < batch_size) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l < timestep; l++) {
                    sum_mu = 0;
                    sum_var = 0;
                    for (int m = 0; m < timestep; m++) {
                        if (m <= k) {
                            idx_qk = i * num_heads * timestep * timestep +
                                     j * timestep * timestep + k * timestep + m;
                            sum_mu += mu_qk[idx_qk];
                            sum_var += var_qk[idx_qk];
                        }
                    }
                    idx_mqk = i * num_heads * timestep * timestep +
                              j * timestep * timestep + k * timestep + l;
                    mu_mqk[idx_mqk] = sum_mu;
                    var_mqk[idx_mqk] = sum_var;
                }
            }
        }
    }
}

void self_attention_forward_cpu(Network &net_prop, NetState &state, int l)
/*Multi-head self-attention mecanism.

Args:
    mth_state: State of multi-heads self attention

*/
{
    int batch_size = net_prop.batch_size;
    int num_heads = net_prop.mha->num_heads[l];
    int timestep = net_prop.mha->timestep[l];
    int head_size = net_prop.mha->head_size[l];
    int att_pos = state.mha->att_pos[l];
    int qkv_pos = state.mha->qkv_pos[l];
    int z_remax_pos = state.mha->remax->z_pos[l];
    int z_sum_remax_pos = state.mha->remax->z_sum_pos[l];
    int z_pos = net_prop.z_pos[l];

    // query x key
    query_key(state.mha->mu_q, state.mha->var_q, state.mha->mu_k,
              state.mha->var_k, qkv_pos, batch_size, num_heads, timestep,
              head_size, state.mha->mu_qk, state.mha->var_qk);

    // Masked the product query x key
    mask_query_key(state.mha->mu_qk, state.mha->var_qk, batch_size, num_heads,
                   timestep, head_size, state.mha->mu_mqk, state.mha->var_mqk);

    // Apply remax on the product of querry and key
    remax_cpu_v2(state.mha->mu_mqk, state.mha->var_mqk, state.mha->remax->mu_m,
                 state.mha->remax->var_m, state.mha->remax->J_m,
                 state.mha->remax->mu_log, state.mha->remax->var_log,
                 state.mha->remax->mu_sum, state.mha->remax->var_sum,
                 state.mha->remax->mu_logsum, state.mha->remax->var_logsum,
                 state.mha->remax->cov_log_logsum, state.mha->mu_att_score,
                 state.mha->var_att_score, att_pos, z_remax_pos,
                 z_sum_remax_pos, timestep, batch_size, net_prop.omega_tol);

    // // Score time values
    tagi_4d_matrix_mul(state.mha->mu_att_score, state.mha->var_att_score,
                       state.mha->mu_v, state.mha->var_v, att_pos, qkv_pos,
                       z_pos, batch_size, num_heads, timestep, head_size,
                       timestep, state.mz, state.Sz);
}

///////////////////////////////////////////////////////////////////////////////
/// BACKWARD PASS
///////////////////////////////////////////////////////////////////////////////
void mha_delta_score(std::vector<float> &mu_v, std::vector<float> &var_s,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int z_pos, int qkv_pos,
                     int att_pos, int batch_size, int num_heads, int timestep,
                     int head_size, std::vector<float> &delta_mu_s,
                     std::vector<float> &delta_var_s)
/*Compute update values for the hidden states of the score*/
{
    float sum_mu, sum_var;
    int idx_scr, idx_val, idx_cov;
    int idx_v, idx_s, idx_obs;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l < timestep; k++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int m = 0; m < head_size; m++) {
                        idx_v = i * num_heads * timestep * timestep +
                                j * timestep * timestep + l * head_size + m +
                                qkv_pos;
                        idx_obs = i * num_heads * timestep * timestep +
                                  j * timestep * timestep + k * head_size + m +
                                  z_pos;
                        sum_mu += mu_v[idx_v] * delta_mu[idx_obs];
                        sum_var +=
                            mu_v[idx_v] * delta_var[idx_obs] * mu_v[idx_v];
                    }
                    idx_s = i * num_heads * timestep * timestep +
                            j * timestep * timestep + k * timestep + l;
                    delta_mu_s[idx_s] = sum_mu * var_s[idx_s + att_pos];
                    delta_var_s[idx_s] = var_s[idx_s + att_pos] * sum_var *
                                         var_s[idx_s + att_pos];
                }
            }
        }
    }
}

void mha_delta_value(std::vector<float> &mu_s, std::vector<float> &var_v,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int z_pos, int qkv_pos,
                     int att_pos, int batch_size, int num_heads, int timestep,
                     int head_size, std::vector<float> &delta_mu_v,
                     std::vector<float> &delta_var_v)
/*Compute update values for the hidden states of the value*/
{
    float sum_mu, sum_var;
    int idx_scr, idx_val, idx_cov;
    int idx_v, idx_s, idx_obs;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int m = 0; m < head_size; m++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int l = 0; l < timestep; l++) {
                        idx_s = i * num_heads * timestep * timestep +
                                j * timestep * timestep + l * timestep + k +
                                att_pos;
                        idx_obs = i * num_heads * timestep * timestep +
                                  j * timestep * timestep + m * timestep + l +
                                  z_pos;
                        sum_mu += mu_s[idx_s] * delta_mu[idx_obs];
                        sum_var +=
                            mu_s[idx_s] * delta_var[idx_obs] * mu_s[idx_s];
                    }
                    idx_v = i * num_heads * timestep * timestep +
                            j * timestep * timestep + k * timestep + m;
                    delta_mu_v[idx_v] = sum_mu * var_v[idx_v + qkv_pos];
                    delta_var_v[idx_v] = var_v[idx_v + qkv_pos] * sum_var *
                                         var_v[idx_v + qkv_pos];
                }
            }
        }
    }
}

void mha_delta_query(std::vector<float> &var_q, std::vector<float> &mu_k,
                     std::vector<float> &var_s, std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int qkv_pos, int att_pos,
                     int batch_size, int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_q,
                     std::vector<float> &delta_var_q)
/**
 * Computes the update values for the query's hidden states. See
 * Multi-Head Self-Attention - QKV formulation for further details
 *
 * This function performs composed operations to compute the update values
 * for the query's hidden states, based on the given input variables. It takes
 * into account the specified batch size, number of heads, timestep, and head
 * size.
 */
{
    int idx_q, idx_k, idx_s;
    float sum_mu, sum_var;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int m = 0; m < head_size; m++) {
                for (int k = 0; k < timestep; k++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int l = 0; l < timestep; l++) {
                        // TO BE TESTED
                        int reduced_row = ((k * timestep + m) / head_size);
                        if (reduced_row * k * timestep <= l) {
                            idx_k = i * num_heads * timestep * timestep +
                                    j * timestep * timestep + l * head_size +
                                    m + qkv_pos;
                            idx_s = i * num_heads * timestep * timestep +
                                    j * timestep * timestep + k * timestep + l +
                                    att_pos;
                            sum_mu +=
                                mu_k[idx_k] * delta_mu[idx_s] / var_s[idx_s];
                            sum_var += (mu_k[idx_k] / var_s[idx_s]) *
                                       delta_var[idx_s] *
                                       (mu_k[idx_k] / var_s[idx_s]);
                        }
                    }
                    idx_q = i * num_heads * timestep * timestep +
                            j * timestep * timestep + m + k * timestep;

                    delta_mu_q[idx_q] = sum_mu * var_q[idx_q + qkv_pos];
                    delta_var_q[idx_q] = var_q[idx_q + qkv_pos] * sum_var *
                                         var_q[idx_q + qkv_pos];
                }
            }
        }
    }
}

void mha_delta_key(std::vector<float> &var_k, std::vector<float> &mu_q,
                   std::vector<float> &var_s, std::vector<float> &delta_mu,
                   std::vector<float> &delta_var, int qkv_pos, int att_pos,
                   int batch_size, int num_heads, int timestep, int head_size,
                   std::vector<float> &delta_mu_k,
                   std::vector<float> &delta_var_k)
/**
 * Computes the update values for the key's hidden states. See
 * Multi-Head Self-Attention - QKV formulation for further details
 *
 * This function performs composed operations to compute the update values
 * for the key's hidden states, based on the given input variables. It takes
 * into account the specified batch size, number of heads, timestep, and head
 * size.
 */
{
    int idx_q, idx_k, idx_s;
    float sum_mu, sum_var;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int m = 0; m < head_size; m++) {
                for (int k = 0; k < timestep; k++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int l = 0; l < timestep; l++) {
                        // TO BE TESTED
                        int reduced_row = ((k * timestep + m) / head_size);
                        if (reduced_row * k * timestep <= l) {
                            idx_k = i * num_heads * timestep * timestep +
                                    j * timestep * timestep + l * head_size +
                                    m + qkv_pos;
                            idx_s = i * num_heads * timestep * timestep +
                                    j * timestep * timestep + k * timestep + l +
                                    att_pos;
                            // TODO: double check on this formulation again
                            // since delta_mu and delta_var are no longer
                            // innovation vector like any other layer
                            sum_mu +=
                                var_k[idx_k] * delta_mu[idx_s] / var_s[idx_s];
                            sum_var += (var_k[idx_k] / var_s[idx_s]) *
                                       delta_var[idx_s] *
                                       (var_k[idx_k] / var_s[idx_s]);
                        }
                    }
                    idx_q = i * num_heads * timestep * timestep +
                            j * timestep * timestep + m + k * timestep;

                    delta_mu_k[idx_q] = sum_mu * mu_q[idx_q + qkv_pos];
                    delta_var_k[idx_q] =
                        mu_q[idx_q + qkv_pos] * sum_var * mu_q[idx_q + qkv_pos];
                }
            }
        }
    }
}

void update_self_attention_state(Network &net_prop, NetState &state,
                                 DeltaState &d_state, int l) {
    int batch_size = net_prop.batch_size;
    int z_pos = net_prop.z_pos[l];
    int num_heads = net_prop.mha->num_heads[l];
    int timestep = net_prop.mha->timestep[l];
    int head_size = net_prop.mha->head_size[l];
    int att_pos = state.mha->att_pos[l];
    int qkv_pos = state.mha->qkv_pos[l];

    // Update values for score hidden states
    mha_delta_score(state.mha->mu_att_score, state.mha->var_att_score,
                    d_state.delta_m, d_state.delta_S, z_pos, qkv_pos, att_pos,
                    batch_size, num_heads, timestep, head_size,
                    d_state.mha->delta_mu_att_score,
                    d_state.mha->delta_var_att_score);

    // Update values for value hidden states
    mha_delta_value(state.mha->mu_att_score, state.mha->var_v, d_state.delta_m,
                    d_state.delta_S, z_pos, qkv_pos, att_pos, batch_size,
                    num_heads, timestep, head_size, d_state.mha->delta_mu_v,
                    d_state.mha->delta_var_v);

    // Update values for query hidden states
    mha_delta_query(state.mha->var_q, state.mha->mu_k, state.mha->var_att_score,
                    d_state.mha->delta_mu_att_score,
                    d_state.mha->delta_var_att_score, qkv_pos, att_pos,
                    batch_size, num_heads, timestep, head_size,
                    d_state.mha->delta_mu_q, d_state.mha->delta_var_q);

    // Update values for key hidden states
    mha_delta_key(state.mha->var_att_score, state.mha->mu_q,
                  state.mha->var_att_score, d_state.mha->delta_mu_att_score,
                  d_state.mha->delta_var_att_score, qkv_pos, att_pos,
                  batch_size, num_heads, timestep, head_size,
                  d_state.mha->delta_mu_k, d_state.mha->delta_var_k);

    // TODO: Missing the remax backward
    // TODO: handling the position for each vector in mha state and delta state
    // (see struct_var.cpp)
}