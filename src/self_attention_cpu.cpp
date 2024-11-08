#include "../include/self_attention_cpu.h"
std::vector<float> transpose(std::vector<float> input,
                             std::vector<int> input_shape,
                             std::vector<int> transpose_dims) {
    std::vector<float> output(input.size());
    std::vector<int> output_shape(input_shape);
    for (int i = 0; i < transpose_dims.size(); ++i) {
        std::swap(output_shape[transpose_dims[i]], output_shape[i]);
    }
    std::vector<int> input_strides(input_shape.size());
    std::vector<int> output_strides(output_shape.size());
    input_strides[input_shape.size() - 1] = 1;
    output_strides[output_shape.size() - 1] = 1;
    for (int i = input_shape.size() - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }
    for (int i = 0; i < input.size(); ++i) {
        int input_idx = 0;
        int output_idx = 0;
        for (int j = 0; j < input_shape.size(); ++j) {
            int coord = i / input_strides[j] % input_shape[j];
            input_idx += coord * input_strides[j];
            output_idx += coord * output_strides[j];
        }
        output_idx += i % output_strides[0];
        output[output_idx] = input[input_idx];
    }
    return output;
}

void project_output_forward(std::vector<float> &mu_in,
                            std::vector<float> &var_in, int in_pos, int out_pos,
                            int batch_size, int num_heads, int timestep,
                            int head_size, std::vector<float> &mu_out,
                            std::vector<float> &var_out)
/*Swap dimensions timestep and num_heads where in(batch_size, num_heads,
   timestep, head_size) . out(batch_size, timestep, num_heads, head_size)*/
{
    int out_idx, in_idx;
    for (int i = 0; i < batch_size; i++) {
        for (int k = 0; k < timestep; k++) {
            for (int j = 0; j < num_heads; j++) {
                for (int m = 0; m < head_size; m++) {
                    out_idx = i * timestep * num_heads * head_size +
                              k * num_heads * head_size + j * num_heads + m +
                              out_pos;
                    in_idx = i * timestep * num_heads * head_size +
                             j * timestep * head_size + k * head_size + m +
                             in_pos;
                    mu_out[out_idx] = mu_in[in_idx];
                    var_out[out_idx] = var_in[in_idx];
                }
            }
        }
    }
}

void project_output_backward(std::vector<float> &mu_in,
                             std::vector<float> &var_in, int in_pos,
                             int out_pos, int batch_size, int num_heads,
                             int timestep, int head_size,
                             std::vector<float> &mu_out,
                             std::vector<float> &var_out)
/*Swap dimensions timestep and num_heads where in(batch_size, timestep,
   num_heads, head_size) . out(batch_size,  num_heads, timestep, head_size)*/
{
    int out_idx, in_idx;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int m = 0; m < head_size; m++) {
                    out_idx = i * timestep * num_heads * head_size +
                              j * timestep * head_size + k * head_size + m +
                              out_pos;
                    in_idx = i * timestep * num_heads * head_size +
                             k * num_heads * head_size + j * num_heads + m +
                             in_pos;
                    mu_out[out_idx] = mu_in[in_idx];
                    var_out[out_idx] = var_in[in_idx];
                }
            }
        }
    }
}

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
                        idx_a = i * C * H * D + j * H * D + k * H + m + a_pos;
                        idx_b = i * C * H * W + j * H * W + l + m * W + b_pos;

                        sum_mu += mu_a[idx_a] * mu_b[idx_b];
                        sum_var += var_a[idx_a] * var_b[idx_b] +
                                   var_a[idx_a] * powf(mu_b[idx_b], 2) +
                                   var_b[idx_b] * powf(mu_a[idx_a], 2);
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
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l < timestep; l++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int m = 0; m < head_size; m++) {
                        idx_q = i * num_heads * timestep * head_size +
                                j * timestep * head_size + k * head_size + m +
                                qkv_pos;
                        idx_k = i * num_heads * timestep * head_size +
                                j * timestep * head_size + l * head_size + m +
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
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l < timestep; l++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int m = 0; m < timestep; m++) {
                        if (m <= k) {
                            idx_qk = i * num_heads * timestep * timestep +
                                     j * timestep * timestep + m * timestep + l;
                            sum_mu += mu_qk[idx_qk];
                            sum_var +=
                                2 * var_qk[idx_qk] + powf(mu_qk[idx_qk], 2);
                        }
                    }
                    idx_mqk = i * num_heads * timestep * timestep +
                              j * timestep * timestep + k * timestep + l;
                    mu_mqk[idx_mqk] = sum_mu / powf(head_size, 0.5);
                    var_mqk[idx_mqk] = sum_var / head_size;
                }
            }
        }
    }
}

void separate_input_projection_components(
    std::vector<float> &mu_embs, std::vector<float> &var_embs, int emb_pos,
    int qkv_pos, int batch_size, int num_heads, int timestep, int head_size,
    std::vector<float> &mu_q, std::vector<float> &var_q,
    std::vector<float> &mu_k, std::vector<float> &var_k,
    std::vector<float> &mu_v, std::vector<float> &var_v)
/*The ordering of the embedding vectors are query, key, and values with a space
   of comp_size */
{
    int comp_idx, emb_idx;
    int comp_size = batch_size * num_heads * timestep * head_size;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int m = 0; m < head_size; m++) {
                    comp_idx = i * num_heads * timestep * head_size +
                               j * timestep * head_size + k * head_size + m +
                               qkv_pos;
                    emb_idx = i * num_heads * timestep * head_size +
                              k * num_heads * head_size + j * head_size + m +
                              emb_pos;
                    // Query
                    mu_q[comp_idx] = mu_embs[emb_idx];
                    var_q[comp_idx] = var_embs[emb_idx];

                    // Key
                    mu_k[comp_idx] = mu_embs[emb_idx + comp_size];
                    var_k[comp_idx] = var_embs[emb_idx + comp_size];

                    // Value
                    mu_v[comp_idx] = mu_embs[emb_idx + 2 * comp_size];
                    var_v[comp_idx] = var_embs[emb_idx + 2 * comp_size];
                }
            }
        }
    }
}

void cat_intput_projection_components(
    std::vector<float> &mu_q, std::vector<float> &var_q,
    std::vector<float> &mu_k, std::vector<float> &var_k,
    std::vector<float> &mu_v, std::vector<float> &var_v, int qkv_pos,
    int emb_pos, int batch_size, int num_heads, int timestep, int head_size,
    std::vector<float> &mu_embs, std::vector<float> &var_embs)
/*Concatenate query, key, and value vectors into a single vector*/
{
    int qkv_idx, emb_idx;
    int comp_size = batch_size * num_heads * timestep * head_size;
    for (int i = 0; i < batch_size; i++) {
        for (int k = 0; k < timestep; k++) {
            for (int j = 0; j < num_heads; j++) {
                for (int m = 0; m < head_size; m++) {
                    qkv_idx = i * batch_size * num_heads * timestep +
                              j * timestep * head_size + k * head_size + m +
                              qkv_pos;
                    emb_idx = i * batch_size * num_heads * timestep +
                              k * num_heads * head_size + j * head_size + m +
                              emb_pos;
                    // Insert query to embeddings
                    mu_embs[emb_idx] = mu_q[qkv_idx];
                    var_embs[emb_idx] = var_q[qkv_idx];

                    // Insert key to embeddings
                    mu_embs[emb_idx + comp_size] = mu_k[qkv_idx];
                    var_embs[emb_idx + comp_size] = var_k[qkv_idx];

                    // Insert value to embeddings
                    mu_embs[emb_idx + 2 * comp_size] = mu_v[qkv_idx];
                    var_embs[emb_idx + 2 * comp_size] = var_v[qkv_idx];
                }
            }
        }
    }
}

void self_attention_forward_cpu(Network &net_prop, NetState &state,
                                Param &theta, int l)
/*Multi-head self-attention mecanism.

Args:
    mth_state: State of multi-heads self attention

*/
{
    auto mha_l =
        get_sub_layer_idx(net_prop.layers, l, net_prop.layer_names.mha);
    int batch_size = net_prop.batch_size;
    int num_heads = net_prop.mha.head_size[mha_l];
    int timestep = net_prop.mha.timestep[mha_l];
    int head_size = net_prop.mha.head_size[mha_l];
    int num_embs = num_heads * head_size;
    int att_pos = state.mha.att_pos[mha_l];
    int qkv_pos = state.mha.qkv_pos[mha_l];
    int z_remax_pos = state.mha.remax.z_pos[mha_l];
    int z_sum_remax_pos = state.mha.remax.z_sum_pos[mha_l];
    int z_pos_out = net_prop.z_pos[l + 1];
    int z_pos_in = net_prop.z_pos[l];
    int w_in_proj_pos = net_prop.w_pos[l];
    int b_in_proj_pos = net_prop.b_pos[l];
    int w_out_proj_pos = net_prop.w_pos[l] + 3 * num_embs * num_embs;
    int b_out_proj_pos = net_prop.b_pos[l] + 3 * num_embs;

    // Query, key, and value projection through a fully-connected layer
    fc_mean_cpu(theta.mw, theta.mb, state.ma, w_in_proj_pos, b_in_proj_pos,
                z_pos_in, 0, 3 * num_embs, num_embs, batch_size * timestep,
                state.mha.mu_in_proj);
    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.ma, state.Sa, w_in_proj_pos,
               b_in_proj_pos, z_pos_in, 0, 3 * num_embs, num_embs,
               batch_size * timestep, state.mha.var_in_proj);

    // Separate the projection componenents into query, key, and values
    separate_input_projection_components(
        state.mha.mu_in_proj, state.mha.var_in_proj, 0, qkv_pos, batch_size,
        num_heads, timestep, head_size, state.mha.mu_q, state.mha.var_q,
        state.mha.mu_k, state.mha.var_k, state.mha.mu_v, state.mha.var_v);

    // query x key
    query_key(state.mha.mu_q, state.mha.var_q, state.mha.mu_k, state.mha.var_k,
              qkv_pos, batch_size, num_heads, timestep, head_size,
              state.mha.mu_qk, state.mha.var_qk);

    // Masked the product query x key. TODO: double check if it divise by
    // sqrt(number of heads)
    mask_query_key(state.mha.mu_qk, state.mha.var_qk, batch_size, num_heads,
                   timestep, head_size, state.mha.mu_mqk, state.mha.var_mqk);

    // Apply remax on the product of querry and key
    // TODO: Merge with current remax
    remax_cpu_v2(state.mha.mu_mqk, state.mha.var_mqk, state.mha.remax.mu_m,
                 state.mha.remax.var_m, state.mha.remax.J_m,
                 state.mha.remax.mu_log, state.mha.remax.var_log,
                 state.mha.remax.mu_sum, state.mha.remax.var_sum,
                 state.mha.remax.mu_logsum, state.mha.remax.var_logsum,
                 state.mha.remax.cov_log_logsum, state.mha.mu_att_score,
                 state.mha.var_att_score, att_pos, z_remax_pos, z_sum_remax_pos,
                 timestep, timestep * num_heads * batch_size);

    // Score time values
    tagi_4d_matrix_mul(state.mha.mu_att_score, state.mha.var_att_score,
                       state.mha.mu_v, state.mha.var_v, att_pos, qkv_pos,
                       qkv_pos, batch_size, num_heads, timestep, head_size,
                       timestep, state.mha.mu_sv, state.mha.var_sv);

    // Projection output forward
    project_output_forward(state.mha.mu_sv, state.mha.var_sv, qkv_pos, qkv_pos,
                           batch_size, num_heads, timestep, head_size,
                           state.mha.mu_out_proj, state.mha.var_out_proj);

    // Output projections
    fc_mean_cpu(theta.mw, theta.mb, state.mha.mu_out_proj, w_out_proj_pos,
                b_out_proj_pos, qkv_pos, z_pos_out, num_embs, num_embs,
                batch_size * timestep, state.mz);
    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.mha.mu_out_proj,
               state.mha.var_out_proj, w_out_proj_pos, b_out_proj_pos, qkv_pos,
               z_pos_out, num_embs, num_embs, batch_size * timestep, state.Sz);
    int check = 1;
}

///////////////////////////////////////////////////////////////////////////////
/// BACKWARD PASS
///////////////////////////////////////////////////////////////////////////////
void mha_delta_score(std::vector<float> &mu_v, std::vector<float> &var_s,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int qkv_pos, int att_pos,
                     int batch_size, int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_s,
                     std::vector<float> &delta_var_s)
/*Compute update values for the hidden states of the score*/
{
    float sum_mu, sum_var;
    int idx_scr, idx_val, idx_cov;
    int idx_v, idx_s, idx_obs;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l < timestep; l++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int m = 0; m < head_size; m++) {
                        idx_v = i * num_heads * timestep * head_size +
                                j * timestep * head_size + l * head_size + m +
                                qkv_pos;
                        idx_obs = i * num_heads * timestep * head_size +
                                  j * timestep * head_size + k * head_size + m;
                        sum_mu += mu_v[idx_v] * delta_mu[idx_obs];
                        sum_var +=
                            mu_v[idx_v] * delta_var[idx_obs] * mu_v[idx_v];
                    }
                    idx_s = i * num_heads * timestep * timestep +
                            j * timestep * timestep + k * timestep + l;
                    // NOTE: We compute directly the delta innovation
                    delta_mu_s[idx_s] = sum_mu / var_s[idx_s + att_pos];
                    delta_var_s[idx_s] =
                        sum_var / powf(var_s[idx_s + att_pos], 2);
                }
            }
        }
    }
}

void mha_delta_value(std::vector<float> &mu_s, std::vector<float> &var_v,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int qkv_pos, int att_pos,
                     int batch_size, int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_v,
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
                        idx_obs = i * num_heads * timestep * head_size +
                                  j * timestep * head_size + l * head_size + m;
                        sum_mu += mu_s[idx_s] * delta_mu[idx_obs];
                        sum_var +=
                            mu_s[idx_s] * delta_var[idx_obs] * mu_s[idx_s];
                    }
                    idx_v = i * num_heads * timestep * head_size +
                            j * timestep * head_size + k * head_size + m;
                    // NOTE: We compute directly the delta innovation
                    delta_mu_v[idx_v] = sum_mu / var_v[idx_v + qkv_pos];
                    delta_var_v[idx_v] =
                        sum_var / powf(var_v[idx_v + qkv_pos], 2);
                }
            }
        }
    }
}

void backward_delta_z_y_remax_cpu(std::vector<float> &delta_mu,
                                  std::vector<float> &delta_var,
                                  std::vector<float> &var_z,
                                  std::vector<float> &cov_z_y, int y_pos,
                                  int batch_size, int no,
                                  std::vector<float> &delta_mu_z,
                                  std::vector<float> &delta_var_z)
/*Smoother update given the inovation vectors and covariance between hidden
   states (z) and observations (y)*/
{
    int idx;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < no; j++) {
            idx = i * no + j;
            delta_mu_z[idx] = cov_z_y[idx] * delta_mu[idx] / var_z[idx];
            delta_var_z[idx] =
                cov_z_y[idx] * delta_var[idx] / powf(var_z[idx], 2);
        }
    }
}

void mha_delta_query(std::vector<float> &var_q, std::vector<float> &mu_k,
                     std::vector<float> &delta_mu,
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
 * NOTE: Delta_mu and delta_var are the update values for the hidde states, not
 * the inovation vectors
 */
{
    int idx_q, idx_k, idx_s, block_row, block_col;
    float sum_mu, sum_var;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int m = 0; m < head_size; m++) {
                for (int k = 0; k < timestep; k++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    block_col = (k * head_size + m);
                    for (int l = 0; l < timestep; l++) {
                        block_row = (k * timestep + l);
                        if (block_row > block_col) {
                            idx_k = i * num_heads * timestep * head_size +
                                    j * timestep * head_size + l * head_size +
                                    m + qkv_pos;
                            idx_s = i * num_heads * timestep * timestep +
                                    j * timestep * timestep + k * timestep + l +
                                    att_pos;
                            sum_mu += mu_k[idx_k] * delta_mu[idx_s];
                            sum_var +=
                                mu_k[idx_k] * delta_var[idx_s] * mu_k[idx_k];
                        }
                    }
                    idx_q = i * num_heads * timestep * head_size +
                            j * timestep * head_size + m + k * head_size;

                    // NOTE: We compute directly the delta innovation
                    delta_mu_q[idx_q] = sum_mu / powf(num_heads, 0.5);
                    delta_var_q[idx_q] = sum_var / num_heads;
                }
            }
        }
    }
}

void mha_delta_key(std::vector<float> &var_k, std::vector<float> &mu_q,
                   std::vector<float> &delta_mu, std::vector<float> &delta_var,
                   int qkv_pos, int att_pos, int batch_size, int num_heads,
                   int timestep, int head_size, std::vector<float> &delta_mu_k,
                   std::vector<float> &delta_var_k)
/**
 * Computes the update values for the key's hidden states. See
 * Multi-Head Self-Attention - QKV formulation for further details
 *
 * This function performs composed operations to compute the update values
 * for the key's hidden states, based on the given input variables. It takes
 * into account the specified batch size, number of heads, timestep, and head
 * size.
 * NOTE: Delta_mu and delta_var are the update values for the hidde states,
 * not the inovation vectors
 */
{
    int idx_q, idx_k, idx_s, block_row, block_col;
    float sum_mu, sum_var;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int m = 0; m < head_size; m++) {
                for (int k = 0; k < timestep; k++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    block_col = (k * head_size + m);
                    for (int l = 0; l < timestep; l++) {
                        block_row = (k * timestep + l);
                        if (block_row > block_col) {
                            idx_k = i * num_heads * timestep * head_size +
                                    j * timestep * head_size + l * head_size +
                                    m + qkv_pos;
                            idx_s = i * num_heads * timestep * timestep +
                                    j * timestep * timestep + k * timestep + l +
                                    att_pos;
                            // TODO: Double check this as well
                            sum_mu += var_k[idx_k] * delta_mu[idx_s];
                            sum_var +=
                                var_k[idx_k] * delta_var[idx_s] * var_k[idx_k];
                        }
                    }
                    idx_q = i * num_heads * timestep * head_size +
                            j * timestep * head_size + m + k * head_size;

                    delta_mu_k[idx_q] =
                        sum_mu * mu_q[idx_q + qkv_pos] / powf(num_heads, 0.5);
                    delta_var_k[idx_q] = mu_q[idx_q + qkv_pos] * sum_var *
                                         mu_q[idx_q + qkv_pos] / num_heads /
                                         powf(var_k[idx_q + qkv_pos], 2);
                }
            }
        }
    }
}

void update_self_attention_state(Network &net_prop, NetState &state,
                                 Param &theta, DeltaState &d_state, int k) {
    int z_pos_in = net_prop.z_pos[k];
    int z_pos_out = net_prop.z_pos[k + 1];
    auto mha_k =
        get_sub_layer_idx(net_prop.layers, k + 1, net_prop.layer_names.mha);
    int batch_size = net_prop.batch_size;
    int num_heads = net_prop.mha.num_heads[mha_k];
    int timestep = net_prop.mha.timestep[mha_k];
    int head_size = net_prop.mha.head_size[mha_k];
    int num_embs = num_heads * head_size;
    int num_batch_remax = batch_size * timestep * num_heads;
    int emb_batch_timestep = num_embs * batch_size * timestep;
    int batch_timestep = batch_size * timestep;
    int att_pos = state.mha.att_pos[mha_k];
    int qkv_pos = state.mha.qkv_pos[mha_k];
    int in_proj_pos = state.mha.in_proj_pos[mha_k];
    int z_remax_pos = state.mha.remax.z_pos[mha_k];
    int z_sum_remax_pos = state.mha.remax.z_sum_pos[mha_k];
    int w_in_proj_pos = net_prop.w_pos[k];
    int b_in_proj_pos = net_prop.b_pos[k];
    int w_out_proj_pos = net_prop.w_pos[k] + 3 * num_embs * num_embs;
    int b_out_proj_pos = net_prop.b_pos[k] + 3 * num_embs;

    ////////////////////////////////////////////////////////////////////
    // Update output projections
    fc_delta_mz(theta.mw, state.mha.var_out_proj, state.mha.J_out_proj,
                d_state.delta_m, w_out_proj_pos, qkv_pos, z_pos_out, num_embs,
                num_embs, batch_timestep, d_state.mha.delta_mu_buffer);
    fc_delta_Sz(theta.mw, state.mha.var_out_proj, state.mha.J_out_proj,
                d_state.delta_S, w_out_proj_pos, qkv_pos, z_pos_out, num_embs,
                num_embs, batch_timestep, d_state.mha.delta_var_buffer);

    // Inovation for output projections
    inovation_mean(state.mha.var_out_proj, d_state.mha.delta_mu_buffer, qkv_pos,
                   0, emb_batch_timestep, d_state.mha.delta_mu_out_proj);
    inovation_var(state.mha.var_out_proj, d_state.mha.delta_var_buffer, qkv_pos,
                  0, emb_batch_timestep, d_state.mha.delta_var_out_proj);

    project_output_backward(
        d_state.mha.delta_mu_out_proj, d_state.mha.delta_var_out_proj, 0, 0,
        batch_size, num_heads, timestep, head_size, d_state.mha.delta_mu_buffer,
        d_state.mha.delta_var_buffer);

    //////////////////////////////////////////////////////////////////////
    // Update values for value hidden states
    mha_delta_value(state.mha.mu_att_score, state.mha.var_v,
                    d_state.mha.delta_mu_buffer, d_state.mha.delta_var_buffer,
                    qkv_pos, att_pos, batch_size, num_heads, timestep,
                    head_size, d_state.mha.delta_mu_v, d_state.mha.delta_var_v);

    // Update values for score hidden states
    mha_delta_score(state.mha.mu_att_score, state.mha.var_att_score,
                    d_state.mha.delta_mu_buffer, d_state.mha.delta_var_buffer,
                    qkv_pos, att_pos, batch_size, num_heads, timestep,
                    head_size, d_state.mha.delta_mu_att_score,
                    d_state.mha.delta_var_att_score);

    //////////////////////////////////////////////////////////////////////
    // Update values for retrieval hidden states R
    compute_cov_m_a_check_cpu(
        state.mha.remax.var_log, state.mha.remax.cov_log_logsum,
        state.mha.remax.mu_m, z_remax_pos, z_sum_remax_pos, timestep,
        num_batch_remax, state.mha.remax.cov_m_a_check);

    // TODO: position for z_pos_mqk
    compute_cov_m_a_cpu(state.mha.remax.cov_m_a_check, state.mha.mu_att_score,
                        state.mha.remax.var_m, state.mha.var_mqk,
                        state.mha.J_mqk, z_remax_pos, att_pos, timestep,
                        num_batch_remax, state.mha.remax.cov_m_a);
    backward_delta_z_y_remax_cpu(
        d_state.mha.delta_mu_att_score, d_state.mha.delta_var_att_score,
        state.mha.var_mqk, state.mha.remax.cov_m_a, att_pos, num_batch_remax,
        timestep, d_state.mha.delta_mu_r, d_state.mha.delta_var_r);

    // Update values for query hidden states
    mha_delta_query(state.mha.var_q, state.mha.mu_k, d_state.mha.delta_mu_r,
                    d_state.mha.delta_var_r, qkv_pos, att_pos, batch_size,
                    num_heads, timestep, head_size, d_state.mha.delta_mu_q,
                    d_state.mha.delta_var_q);

    // Update values for key hidden states
    mha_delta_key(state.mha.var_k, state.mha.mu_q, d_state.mha.delta_mu_r,
                  d_state.mha.delta_var_r, qkv_pos, att_pos, batch_size,
                  num_heads, timestep, head_size, d_state.mha.delta_mu_k,
                  d_state.mha.delta_var_k);

    ///////////////////////////////////////////////////////////////////////
    // Concatenate the input projection components (3 x embeddings)
    cat_intput_projection_components(
        d_state.mha.delta_mu_q, d_state.mha.delta_var_q, d_state.mha.delta_mu_k,
        d_state.mha.delta_var_k, d_state.mha.delta_mu_v,
        d_state.mha.delta_var_v, 0, in_proj_pos, batch_size, num_heads,
        timestep, head_size, d_state.mha.delta_mu_in_proj,
        d_state.mha.delta_var_in_proj);

    // Input of the embeddings
    fc_delta_mz(theta.mw, state.Sz, state.J, d_state.mha.delta_mu_in_proj,
                w_in_proj_pos, z_pos_in, in_proj_pos, num_embs, 3 * num_embs,
                batch_timestep, d_state.delta_mz);
    fc_delta_Sz(theta.mw, state.Sz, state.J, d_state.mha.delta_var_in_proj,
                w_in_proj_pos, z_pos_in, in_proj_pos, num_embs, 3 * num_embs,
                batch_timestep, d_state.delta_Sz);
}

void update_self_attention_param(Network &net_prop, Param &theta,
                                 NetState &state, DeltaState &d_state,
                                 DeltaParam &d_theta, int k_layer)
/**/
{
    auto mha_k = get_sub_layer_idx(net_prop.layers, k_layer + 1,
                                   net_prop.layer_names.mha);
    int z_pos_out = net_prop.z_pos[k_layer + 1];
    int z_pos_in = net_prop.z_pos[k_layer];
    int att_pos = state.mha.att_pos[mha_k];
    int qkv_pos = state.mha.qkv_pos[mha_k];
    int in_proj_pos = state.mha.in_proj_pos[mha_k];
    int num_heads = net_prop.mha.num_heads[mha_k];
    int timestep = net_prop.mha.timestep[mha_k];
    int head_size = net_prop.mha.head_size[mha_k];
    int num_embs = num_heads * head_size;
    int batch_timestep = net_prop.batch_size * timestep;
    int w_in_proj_pos = net_prop.w_pos[k_layer];
    int b_in_proj_pos = net_prop.b_pos[k_layer];
    int w_out_proj_pos = net_prop.w_pos[k_layer] + 3 * num_embs * num_embs;
    int b_out_proj_pos = net_prop.b_pos[k_layer] + 3 * num_embs;

    // Parameters for output projections
    fc_delta_mw(theta.Sw, state.mha.mu_out_proj, d_state.delta_m,
                w_out_proj_pos, qkv_pos, z_pos_out, num_embs, batch_timestep,
                num_embs, d_theta.delta_mw);
    fc_delta_Sw(theta.Sw, state.mha.mu_out_proj, d_state.delta_S,
                w_out_proj_pos, qkv_pos, z_pos_out, num_embs, batch_timestep,
                num_embs, d_theta.delta_Sw);
    fc_delta_mb(theta.Sb, d_state.delta_m, b_out_proj_pos, z_pos_out, num_embs,
                batch_timestep, 1, d_theta.delta_mb);
    fc_delta_mb(theta.Sb, d_state.delta_S, b_out_proj_pos, z_pos_out, num_embs,
                batch_timestep, 1, d_theta.delta_Sb);

    // Parameters for input projections. TODO: need to revise the postion for
    // delta_mu_in_proj
    fc_delta_mw(theta.Sw, state.ma, d_state.mha.delta_mu_in_proj, w_in_proj_pos,
                z_pos_in, in_proj_pos, num_embs, batch_timestep, num_embs,
                d_theta.delta_mw);
    fc_delta_Sw(theta.Sw, state.ma, d_state.mha.delta_var_in_proj,
                w_in_proj_pos, z_pos_in, in_proj_pos, num_embs, batch_timestep,
                num_embs, d_theta.delta_Sw);
    fc_delta_mb(theta.Sb, d_state.mha.delta_mu_in_proj, b_in_proj_pos,
                in_proj_pos, num_embs, batch_timestep, 1, d_theta.delta_mb);
    fc_delta_mb(theta.Sb, d_state.mha.delta_var_in_proj, b_in_proj_pos,
                in_proj_pos, num_embs, batch_timestep, 1, d_theta.delta_Sb);
}
