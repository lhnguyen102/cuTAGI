#include "../include/attention.h"

#include "../include/activation.h"
#include "../include/common.h"
#include "../include/custom_logger.h"
#include "../include/linear_layer.h"

// #ifdef USE_CUDA
// #include "../include/attention_cuda.cuh"
// #endif

void separate_input_projection_components(
    std::vector<float> &mu_embs, std::vector<float> &var_embs, int batch_size,
    int num_heads, int timestep, int head_dim, std::vector<float> &mu_q,
    std::vector<float> &var_q, std::vector<float> &mu_k,
    std::vector<float> &var_k, std::vector<float> &mu_v,
    std::vector<float> &var_v)
/*Separate input projection components into query, key, and value

embs: [batch_size, num_heads, timestep, head_dim]
q: [batch_size, num_heads, timestep, head_dim]
k: [batch_size, num_heads, timestep, head_dim]
v: [batch_size, num_heads, timestep, head_dim]
*/
{
    int comp_idx, emb_idx;
    int comp_size = batch_size * num_heads * timestep * head_dim;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int m = 0; m < head_dim; m++) {
                    comp_idx = i * num_heads * timestep * head_dim +
                               j * timestep * head_dim + k * head_dim + m;
                    emb_idx = i * num_heads * timestep * head_dim +
                              k * num_heads * head_dim + j * head_dim + m;
                    mu_q[comp_idx] = mu_embs[emb_idx];
                    var_q[comp_idx] = var_embs[emb_idx];

                    mu_k[comp_idx] = mu_embs[emb_idx + comp_size];
                    var_k[comp_idx] = var_embs[emb_idx + comp_size];

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
    std::vector<float> &mu_v, std::vector<float> &var_v, int batch_size,
    int num_heads, int timestep, int head_size, std::vector<float> &mu_embs,
    std::vector<float> &var_embs)
/*Concatenate query, key, and value vectors into a single vector

q: [batch_size, num_heads, timestep, head_dim]
k: [batch_size, num_heads, timestep, head_dim]
v: [batch_size, num_heads, timestep, head_dim]
embs: [batch_size, num_heads, timestep, head_dim]
*/
{
    int qkv_idx, emb_idx;
    int comp_size = batch_size * num_heads * timestep * head_size;
    for (int i = 0; i < batch_size; i++) {
        for (int k = 0; k < timestep; k++) {
            for (int j = 0; j < num_heads; j++) {
                for (int m = 0; m < head_size; m++) {
                    qkv_idx = i * num_heads * timestep * head_size +
                              j * timestep * head_size + k * head_size + m;
                    emb_idx = i * num_heads * timestep * head_size +
                              k * num_heads * head_size + j * head_size + m;
                    mu_embs[emb_idx] = mu_q[qkv_idx];
                    var_embs[emb_idx] = var_q[qkv_idx];

                    mu_embs[emb_idx + comp_size] = mu_k[qkv_idx];
                    var_embs[emb_idx + comp_size] = var_k[qkv_idx];

                    mu_embs[emb_idx + 2 * comp_size] = mu_v[qkv_idx];
                    var_embs[emb_idx + 2 * comp_size] = var_v[qkv_idx];
                }
            }
        }
    }
}

void query_key(std::vector<float> &mu_q, std::vector<float> &var_q,
               std::vector<float> &mu_k, std::vector<float> &var_k,
               int batch_size, int num_heads, int timestep, int head_size,
               std::vector<float> &mu_qk, std::vector<float> &var_qk)
/*4D matrix multiplication of query matrix with key matrix

q: [batch_size, num_heads, timestep, head_dim]
k: [batch_size, num_heads, timestep, head_dim]
qk: [batch_size, num_heads, timestep, timestep]
*/
{
    int idx_q, idx_k, idx_qk;
    float sum_mu, sum_var;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l < timestep; l++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int m = 0; m < head_size; m++) {
                        idx_q = i * num_heads * timestep * head_size +
                                j * timestep * head_size + k * head_size + m;
                        idx_k = i * num_heads * timestep * head_size +
                                j * timestep * head_size + l * head_size + m;

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
                    std::vector<float> &mu_mqk, std::vector<float> &var_mqk)
/*Mask query key matrix to ensure we are not attending to future timesteps

qk: [batch_size, num_heads, timestep, timestep]
mqk: [batch_size, num_heads, timestep, timestep]
*/
{
    int idx_qk;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l < timestep; l++) {
                    idx_qk = i * num_heads * timestep * timestep +
                             j * timestep * timestep + k * timestep + l;
                    if (l <= k) {
                        mu_mqk[idx_qk] = mu_qk[idx_qk];
                        var_mqk[idx_qk] = var_qk[idx_qk];
                    } else {
                        mu_mqk[idx_qk] = 0.0f;
                        var_mqk[idx_qk] = 0.0f;
                    }
                }
            }
        }
    }
}

void tagi_4d_matrix_mul(std::vector<float> &mu_a, std::vector<float> &var_a,
                        std::vector<float> &mu_b, std::vector<float> &var_b,
                        int N, int C, int H, int W, int D,
                        std::vector<float> &mu_ab, std::vector<float> &var_ab)
/*4D matrix multiplication of two 4D matrices

a: [batch_size, num_heads, timestep, timestep]
b: [batch_size, num_heads, timestep, head_dim]
a@b: [batch_size, num_heads, timestep, head_dim]
*/
{
    int idx_a, idx_b, idx_ab;
    float sum_mu, sum_var;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < C; j++) {
            for (int k = 0; k < H; k++) {
                for (int l = 0; l < W; l++) {
                    sum_mu = 0;
                    sum_var = 0;
                    for (int m = 0; m < D; m++) {
                        idx_a = i * C * H * D + j * H * D + k * H + m;
                        idx_b = i * C * H * W + j * H * W + l + m * W;

                        sum_mu += mu_a[idx_a] * mu_b[idx_b];
                        sum_var += var_a[idx_a] * var_b[idx_b] +
                                   var_a[idx_a] * powf(mu_b[idx_b], 2) +
                                   var_b[idx_b] * powf(mu_a[idx_a], 2);
                    }
                    idx_ab = i * C * H * W + j * H * W + k * W + l;
                    mu_ab[idx_ab] = sum_mu;
                    var_ab[idx_ab] = sum_var;
                }
            }
        }
    }
}

void project_output_forward(std::vector<float> &mu_in,
                            std::vector<float> &var_in, int batch_size,
                            int num_heads, int timestep, int head_size,
                            std::vector<float> &mu_out,
                            std::vector<float> &var_out)
/*Swap dimensions timestep and num_heads where,
in(batch_size, num_heads, timestep, head_size) ->
out(batch_size, timestep, num_heads, head_size)
*/
{
    int out_idx, in_idx;
    for (int i = 0; i < batch_size; i++) {
        for (int k = 0; k < timestep; k++) {
            for (int j = 0; j < num_heads; j++) {
                for (int m = 0; m < head_size; m++) {
                    out_idx = i * timestep * num_heads * head_size +
                              k * num_heads * head_size + j * head_size + m;
                    in_idx = i * timestep * num_heads * head_size +
                             j * timestep * head_size + k * head_size + m;
                    mu_out[out_idx] = mu_in[in_idx];
                    var_out[out_idx] = var_in[in_idx];
                }
            }
        }
    }
}

void project_output_backward(std::vector<float> &mu_in,
                             std::vector<float> &var_in, int batch_size,
                             int num_heads, int timestep, int head_size,
                             std::vector<float> &mu_out,
                             std::vector<float> &var_out)
/*
in(batch_size, timestep, num_heads, head_size) ->
out(batch_size, num_heads, timestep, head_size)
*/
{
    int out_idx, in_idx;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int m = 0; m < head_size; m++) {
                    out_idx = i * timestep * num_heads * head_size +
                              j * timestep * head_size + k * head_size + m;
                    in_idx = i * timestep * num_heads * head_size +
                             k * num_heads * head_size + j * head_size + m;
                    mu_out[out_idx] = mu_in[in_idx];
                    var_out[out_idx] = var_in[in_idx];
                }
            }
        }
    }
}

void mha_delta_score(std::vector<float> &mu_v, std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int batch_size,
                     int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_s,
                     std::vector<float> &delta_var_s) {
    float sum_mu, sum_var;
    int idx_v, idx_s, idx_obs;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l < timestep; l++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int m = 0; m < head_size; m++) {
                        idx_v = i * num_heads * timestep * head_size +
                                j * timestep * head_size + l * head_size + m;
                        idx_obs = i * num_heads * timestep * head_size +
                                  j * timestep * head_size + k * head_size + m;
                        sum_mu += mu_v[idx_v] * delta_mu[idx_obs];
                        sum_var +=
                            mu_v[idx_v] * delta_var[idx_obs] * mu_v[idx_v];
                    }
                    idx_s = i * num_heads * timestep * timestep +
                            j * timestep * timestep + k * timestep + l;
                    delta_mu_s[idx_s] = sum_mu;
                    delta_var_s[idx_s] = sum_var;
                }
            }
        }
    }
}

void mha_delta_value(std::vector<float> &mu_s, std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int batch_size,
                     int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_v,
                     std::vector<float> &delta_var_v) {
    float sum_mu, sum_var;
    int idx_v, idx_s, idx_obs;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int m = 0; m < head_size; m++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int l = 0; l < timestep; l++) {
                        idx_s = i * num_heads * timestep * timestep +
                                j * timestep * timestep + l * timestep + k;
                        idx_obs = i * num_heads * timestep * head_size +
                                  j * timestep * head_size + l * head_size + m;
                        sum_mu += mu_s[idx_s] * delta_mu[idx_obs];
                        sum_var +=
                            mu_s[idx_s] * delta_var[idx_obs] * mu_s[idx_s];
                    }
                    idx_v = i * num_heads * timestep * head_size +
                            j * timestep * head_size + k * head_size + m;
                    delta_mu_v[idx_v] = sum_mu;
                    delta_var_v[idx_v] = sum_var;
                }
            }
        }
    }
}

void mha_delta_query(std::vector<float> &var_q, std::vector<float> &mu_k,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, std::vector<float> &jcb_mqk,
                     int batch_size, int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_q,
                     std::vector<float> &delta_var_q) {
    int idx_q, idx_k, idx_s;
    float sum_mu, sum_var;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int m = 0; m < head_size; m++) {
                for (int k = 0; k < timestep; k++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int l = 0; l < timestep; l++) {
                        if (l <= k) {
                            idx_k = i * num_heads * timestep * head_size +
                                    j * timestep * head_size + l * head_size +
                                    m;
                            idx_s = i * num_heads * timestep * timestep +
                                    j * timestep * timestep + k * timestep + l;
                            sum_mu +=
                                mu_k[idx_k] * delta_mu[idx_s] * jcb_mqk[idx_s];
                            sum_var += mu_k[idx_k] * delta_var[idx_s] *
                                       mu_k[idx_k] * jcb_mqk[idx_s] *
                                       jcb_mqk[idx_s];
                        }
                    }
                    idx_q = i * num_heads * timestep * head_size +
                            j * timestep * head_size + m + k * head_size;

                    delta_mu_q[idx_q] = sum_mu / powf(num_heads, 0.5);
                    delta_var_q[idx_q] = sum_var / num_heads;
                }
            }
        }
    }
}

void mha_delta_key(std::vector<float> &var_k, std::vector<float> &mu_q,
                   std::vector<float> &delta_mu, std::vector<float> &delta_var,
                   std::vector<float> &jcb_mqk, int batch_size, int num_heads,
                   int timestep, int head_size, std::vector<float> &delta_mu_k,
                   std::vector<float> &delta_var_k) {
    int idx_q, idx_s;
    float sum_mu, sum_var;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int m = 0; m < head_size; m++) {
                for (int k = 0; k < timestep; k++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int l = 0; l < timestep; l++) {
                        if (l <= k) {
                            idx_s = i * num_heads * timestep * timestep +
                                    j * timestep * timestep + k * timestep + l;

                            sum_mu += delta_mu[idx_s] * jcb_mqk[idx_s];
                            sum_var +=
                                delta_var[idx_s] * powf(jcb_mqk[idx_s], 2);
                        }
                    }
                    idx_q = i * num_heads * timestep * head_size +
                            j * timestep * head_size + m + k * head_size;

                    delta_mu_k[idx_q] =
                        sum_mu * mu_q[idx_q] / powf(num_heads, 0.5);
                    delta_var_k[idx_q] =
                        mu_q[idx_q] * sum_var * mu_q[idx_q] / num_heads;
                }
            }
        }
    }
}

void generate_rope_cache(int max_seq_len, int head_dim, float theta,
                         std::vector<float> &cos_cache,
                         std::vector<float> &sin_cache) {
    int half_dim = head_dim / 2;
    cos_cache.resize(max_seq_len * half_dim);
    sin_cache.resize(max_seq_len * half_dim);

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
            float angle = pos * freq;
            int idx = pos * half_dim + i;
            cos_cache[idx] = cosf(angle);
            sin_cache[idx] = sinf(angle);
        }
    }
}

void apply_rope(std::vector<float> &mu_in, std::vector<float> &var_in,
                std::vector<float> &cos_cache, std::vector<float> &sin_cache,
                int batch_size, int num_heads, int timestep, int head_dim,
                std::vector<float> &mu_out, std::vector<float> &var_out) {
    int half_dim = head_dim / 2;
    int idx_in, idx_cache;
    float mu_x1, mu_x2, var_x1, var_x2, cos_val, sin_val;

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int t = 0; t < timestep; t++) {
                for (int d = 0; d < half_dim; d++) {
                    idx_in = i * num_heads * timestep * head_dim +
                             j * timestep * head_dim + t * head_dim + 2 * d;
                    idx_cache = t * half_dim + d;

                    mu_x1 = mu_in[idx_in];
                    mu_x2 = mu_in[idx_in + 1];
                    var_x1 = var_in[idx_in];
                    var_x2 = var_in[idx_in + 1];

                    cos_val = cos_cache[idx_cache];
                    sin_val = sin_cache[idx_cache];

                    mu_out[idx_in] = mu_x1 * cos_val - mu_x2 * sin_val;
                    mu_out[idx_in + 1] = mu_x1 * sin_val + mu_x2 * cos_val;

                    var_out[idx_in] =
                        var_x1 * cos_val * cos_val + var_x2 * sin_val * sin_val;
                    var_out[idx_in + 1] =
                        var_x1 * sin_val * sin_val + var_x2 * cos_val * cos_val;
                }
            }
        }
    }
}

void rope_backward(std::vector<float> &delta_mu_in,
                   std::vector<float> &delta_var_in,
                   std::vector<float> &cos_cache, std::vector<float> &sin_cache,
                   int batch_size, int num_heads, int timestep, int head_dim,
                   std::vector<float> &delta_mu_out,
                   std::vector<float> &delta_var_out) {
    int half_dim = head_dim / 2;
    int idx_in, idx_cache;
    float dmu_y1, dmu_y2, dvar_y1, dvar_y2, cos_val, sin_val;

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int t = 0; t < timestep; t++) {
                for (int d = 0; d < half_dim; d++) {
                    idx_in = i * num_heads * timestep * head_dim +
                             j * timestep * head_dim + t * head_dim + 2 * d;
                    idx_cache = t * half_dim + d;

                    dmu_y1 = delta_mu_in[idx_in];
                    dmu_y2 = delta_mu_in[idx_in + 1];
                    dvar_y1 = delta_var_in[idx_in];
                    dvar_y2 = delta_var_in[idx_in + 1];

                    cos_val = cos_cache[idx_cache];
                    sin_val = sin_cache[idx_cache];

                    delta_mu_out[idx_in] = dmu_y1 * cos_val + dmu_y2 * sin_val;
                    delta_mu_out[idx_in + 1] =
                        -dmu_y1 * sin_val + dmu_y2 * cos_val;

                    delta_var_out[idx_in] = dvar_y1 * cos_val * cos_val +
                                            dvar_y2 * sin_val * sin_val;
                    delta_var_out[idx_in + 1] = dvar_y1 * sin_val * sin_val +
                                                dvar_y2 * cos_val * cos_val;
                }
            }
        }
    }
}

void AttentionStates::set_size(int batch_size, int num_heads, int timestep,
                               int head_size) {
    int num_embs = num_heads * head_size;
    int comp_size = batch_size * num_heads * timestep * head_size;
    int qk_size = batch_size * num_heads * timestep * timestep;
    int num_batch_remax = batch_size * timestep * num_heads;

    mu_in_proj.resize(3 * comp_size, 0.0f);
    var_in_proj.resize(3 * comp_size, 0.0f);

    mu_q.resize(comp_size, 0.0f);
    var_q.resize(comp_size, 0.0f);
    mu_k.resize(comp_size, 0.0f);
    var_k.resize(comp_size, 0.0f);
    mu_v.resize(comp_size, 0.0f);
    var_v.resize(comp_size, 0.0f);

    mu_q_rope.resize(comp_size, 0.0f);
    var_q_rope.resize(comp_size, 0.0f);
    mu_k_rope.resize(comp_size, 0.0f);
    var_k_rope.resize(comp_size, 0.0f);

    mu_qk.resize(qk_size, 0.0f);
    var_qk.resize(qk_size, 0.0f);

    mu_mqk.resize(qk_size, 0.0f);
    var_mqk.resize(qk_size, 0.0f);

    mu_att_score.resize(qk_size, 0.0f);
    var_att_score.resize(qk_size, 0.0f);

    mu_sv.resize(comp_size, 0.0f);
    var_sv.resize(comp_size, 0.0f);
}

void AttentionDeltaStates::set_size(int batch_size, int num_heads, int timestep,
                                    int head_size) {
    int num_embs = num_heads * head_size;
    int comp_size = batch_size * num_heads * timestep * head_size;
    int qk_size = batch_size * num_heads * timestep * timestep;
    int emb_batch_timestep = num_embs * batch_size * timestep;

    delta_mu_buffer.resize(comp_size, 0.0f);
    delta_var_buffer.resize(comp_size, 0.0f);
    delta_mu_v.resize(comp_size, 0.0f);
    delta_var_v.resize(comp_size, 0.0f);
    delta_mu_att_score.resize(qk_size, 0.0f);
    delta_var_att_score.resize(qk_size, 0.0f);
    delta_mu_q.resize(comp_size, 0.0f);
    delta_var_q.resize(comp_size, 0.0f);
    delta_mu_k.resize(comp_size, 0.0f);
    delta_var_k.resize(comp_size, 0.0f);
    delta_mu_q_rope.resize(comp_size, 0.0f);
    delta_var_q_rope.resize(comp_size, 0.0f);
    delta_mu_k_rope.resize(comp_size, 0.0f);
    delta_var_k_rope.resize(comp_size, 0.0f);
    delta_mu_in_proj.resize(3 * comp_size, 0.0f);
    delta_var_in_proj.resize(3 * comp_size, 0.0f);
}

MultiheadAttention::MultiheadAttention(size_t embed_dim, size_t num_heads,
                                       size_t num_kv_heads, bool bias,
                                       float gain_w, float gain_b,
                                       std::string init_method, bool use_rope,
                                       float rope_theta, size_t max_seq_len,
                                       int device_idx)
    : embed_dim(embed_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method),
      use_rope(use_rope),
      rope_theta(rope_theta),
      max_seq_len(max_seq_len) {
    this->input_size = embed_dim;
    this->output_size = this->embed_dim;
    this->head_dim = embed_dim / num_heads;
    this->bias = bias;
    this->device_idx = device_idx;

    // query: Linear(num_embs, num_heads * head_dim)
    // key: Linear(num_embs, num_kv_heads * head_dim)
    // value: Linear(num_embs, num_kv_heads * head_dim)
    this->num_weights =
        embed_dim * ((num_heads + 2 * num_kv_heads) * this->head_dim);
    this->num_biases = 0;
    if (this->bias) {
        this->num_biases = (num_heads + 2 * num_kv_heads) * this->head_dim;
    }

    if (this->device.compare("cpu") == 0) {
        this->init_weight_bias();
    }

    if (this->training && this->device.compare("cpu") == 0) {
        this->allocate_param_delta();
    }

    remax_layer = std::make_unique<Remax>();

    if (this->use_rope) {
        generate_rope_cache(this->max_seq_len, this->head_dim, this->rope_theta,
                            this->cos_cache, this->sin_cache);
    }
}

MultiheadAttention::~MultiheadAttention() {}

std::string MultiheadAttention::get_layer_info() const {
    return "SelfAttention(heads=" + std::to_string(this->num_heads) +
           ", kv_heads=" + std::to_string(this->num_kv_heads) +
           ", emb_size=" + std::to_string(this->embed_dim) + ")";
}

std::string MultiheadAttention::get_layer_name() const {
    return "MultiheadAttention";
}

LayerType MultiheadAttention::get_layer_type() const {
    return LayerType::MultiheadAttention;
}

void MultiheadAttention::init_weight_bias() {
    int q_input = embed_dim;
    int q_output = num_heads * head_dim;
    int k_input = embed_dim;
    int k_output = num_kv_heads * head_dim;
    int v_input = embed_dim;
    int v_output = num_kv_heads * head_dim;

    int q_weights = q_input * q_output;
    int k_weights = k_input * k_output;
    int v_weights = v_input * v_output;

    int q_biases = this->bias ? q_output : 0;
    int k_biases = this->bias ? k_output : 0;
    int v_biases = this->bias ? v_output : 0;

    std::vector<float> mu_w_q, var_w_q, mu_b_q, var_b_q;
    std::vector<float> mu_w_k, var_w_k, mu_b_k, var_b_k;
    std::vector<float> mu_w_v, var_w_v, mu_b_v, var_b_v;
    std::vector<float> mu_w_o, var_w_o, mu_b_o, var_b_o;

    std::tie(mu_w_q, var_w_q, mu_b_q, var_b_q) =
        init_weight_bias_linear(this->init_method, this->gain_w, this->gain_b,
                                q_input, q_output, q_weights, q_biases);

    std::tie(mu_w_k, var_w_k, mu_b_k, var_b_k) =
        init_weight_bias_linear(this->init_method, this->gain_w, this->gain_b,
                                k_input, k_output, k_weights, k_biases);

    std::tie(mu_w_v, var_w_v, mu_b_v, var_b_v) =
        init_weight_bias_linear(this->init_method, this->gain_w, this->gain_b,
                                v_input, v_output, v_weights, v_biases);

    this->mu_w.resize(this->num_weights);
    this->var_w.resize(this->num_weights);
    this->mu_b.resize(this->num_biases);
    this->var_b.resize(this->num_biases);

    std::copy(mu_w_q.begin(), mu_w_q.end(), this->mu_w.begin());
    std::copy(mu_w_k.begin(), mu_w_k.end(), this->mu_w.begin() + q_weights);
    std::copy(mu_w_v.begin(), mu_w_v.end(),
              this->mu_w.begin() + q_weights + k_weights);

    std::copy(var_w_q.begin(), var_w_q.end(), this->var_w.begin());
    std::copy(var_w_k.begin(), var_w_k.end(), this->var_w.begin() + q_weights);
    std::copy(var_w_v.begin(), var_w_v.end(),
              this->var_w.begin() + q_weights + k_weights);

    if (this->bias) {
        std::copy(mu_b_q.begin(), mu_b_q.end(), this->mu_b.begin());
        std::copy(mu_b_k.begin(), mu_b_k.end(), this->mu_b.begin() + q_biases);
        std::copy(mu_b_v.begin(), mu_b_v.end(),
                  this->mu_b.begin() + q_biases + k_biases);

        std::copy(var_b_q.begin(), var_b_q.end(), this->var_b.begin());
        std::copy(var_b_k.begin(), var_b_k.end(),
                  this->var_b.begin() + q_biases);
        std::copy(var_b_v.begin(), var_b_v.end(),
                  this->var_b.begin() + q_biases + k_biases);
    }
}

void MultiheadAttention::forward(BaseHiddenStates &input_states,
                                 BaseHiddenStates &output_states,
                                 BaseTempStates &temp_states) {
    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    // TODO: double check with LSTM layer
    if (this->input_size * this->seq_len != input_states.actual_size) {
        this->seq_len = input_states.actual_size / this->input_size;
    }

    attn_states.set_size(batch_size, num_heads, this->seq_len, head_dim);

    // query, key, value
    size_t input_qkv_size = this->embed_dim;
    size_t output_qkv_size =
        this->head_dim * (this->num_heads + 2 * this->num_kv_heads);
    linear_fwd_mean_var_mp(
        this->mu_w, this->var_w, this->mu_b, this->var_b, input_states.mu_a,
        input_states.var_a, input_qkv_size, output_qkv_size,
        batch_size * this->seq_len, this->bias, this->num_threads,
        attn_states.mu_in_proj, attn_states.var_in_proj);

    separate_input_projection_components(
        attn_states.mu_in_proj, attn_states.var_in_proj, batch_size, num_heads,
        this->seq_len, head_dim, attn_states.mu_q, attn_states.var_q,
        attn_states.mu_k, attn_states.var_k, attn_states.mu_v,
        attn_states.var_v);

    if (this->use_rope) {
        apply_rope(attn_states.mu_q, attn_states.var_q, this->cos_cache,
                   this->sin_cache, batch_size, num_heads, this->seq_len,
                   head_dim, attn_states.mu_q_rope, attn_states.var_q_rope);

        apply_rope(attn_states.mu_k, attn_states.var_k, this->cos_cache,
                   this->sin_cache, batch_size, num_heads, this->seq_len,
                   head_dim, attn_states.mu_k_rope, attn_states.var_k_rope);

        query_key(attn_states.mu_q_rope, attn_states.var_q_rope,
                  attn_states.mu_k_rope, attn_states.var_k_rope, batch_size,
                  num_heads, this->seq_len, head_dim, attn_states.mu_qk,
                  attn_states.var_qk);
    } else {
        query_key(attn_states.mu_q, attn_states.var_q, attn_states.mu_k,
                  attn_states.var_k, batch_size, num_heads, this->seq_len,
                  head_dim, attn_states.mu_qk, attn_states.var_qk);
    }

    mask_query_key(attn_states.mu_qk, attn_states.var_qk, batch_size, num_heads,
                   this->seq_len, head_dim, attn_states.mu_mqk,
                   attn_states.var_mqk);

    // Apply Remax (probabilistic softmax) on masked query-key product
    int qk_size = batch_size * num_heads * this->seq_len * this->seq_len;
    remax_input.mu_a = attn_states.mu_mqk;
    remax_input.var_a = attn_states.var_mqk;
    remax_input.block_size = batch_size * this->seq_len * this->num_heads;
    remax_input.actual_size = this->seq_len;

    remax_output.set_size(qk_size,
                          batch_size * this->seq_len * this->num_heads);

    remax_layer->forward(remax_input, remax_output, remax_temp);

    attn_states.mu_att_score = remax_output.mu_a;
    attn_states.var_att_score = remax_output.var_a;
    attn_states.j_mqk = remax_output.jcb;

    tagi_4d_matrix_mul(attn_states.mu_att_score, attn_states.var_att_score,
                       attn_states.mu_v, attn_states.var_v, batch_size,
                       num_heads, this->seq_len, head_dim, this->seq_len,
                       attn_states.mu_sv, attn_states.var_sv);

    project_output_forward(attn_states.mu_sv, attn_states.var_sv, batch_size,
                           num_heads, this->seq_len, head_dim,
                           output_states.mu_a, output_states.var_a);

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size * this->seq_len;

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void MultiheadAttention::backward(BaseDeltaStates &input_delta_states,
                                  BaseDeltaStates &output_delta_states,
                                  BaseTempStates &temp_states,
                                  bool state_udapte) {
    int batch_size = input_delta_states.block_size;

    attn_delta_states.set_size(batch_size, num_heads, this->seq_len, head_dim);
    int batch_seq_len = batch_size * this->seq_len;

    size_t input_qkv_size = this->embed_dim;
    size_t output_qkv_size =
        this->head_dim * (this->num_heads + 2 * this->num_kv_heads);

    project_output_backward(
        input_delta_states.delta_mu, input_delta_states.delta_var, batch_size,
        this->num_heads, this->seq_len, this->head_dim,
        attn_delta_states.delta_mu_buffer, attn_delta_states.delta_var_buffer);

    mha_delta_value(attn_states.mu_att_score, attn_delta_states.delta_mu_buffer,
                    attn_delta_states.delta_var_buffer, batch_size,
                    this->num_heads, this->seq_len, this->head_dim,
                    attn_delta_states.delta_mu_v,
                    attn_delta_states.delta_var_v);

    mha_delta_score(attn_states.mu_att_score, attn_delta_states.delta_mu_buffer,
                    attn_delta_states.delta_var_buffer, batch_size,
                    this->num_heads, this->seq_len, this->head_dim,
                    attn_delta_states.delta_mu_att_score,
                    attn_delta_states.delta_var_att_score);

    if (this->use_rope) {
        mha_delta_query(attn_states.var_q, attn_states.mu_k_rope,
                        attn_delta_states.delta_mu_att_score,
                        attn_delta_states.delta_var_att_score,
                        attn_states.j_mqk, batch_size, num_heads, this->seq_len,
                        this->head_dim, attn_delta_states.delta_mu_q_rope,
                        attn_delta_states.delta_var_q_rope);

        mha_delta_key(attn_states.var_k, attn_states.mu_q_rope,
                      attn_delta_states.delta_mu_att_score,
                      attn_delta_states.delta_var_att_score, attn_states.j_mqk,
                      batch_size, num_heads, this->seq_len, this->head_dim,
                      attn_delta_states.delta_mu_k_rope,
                      attn_delta_states.delta_var_k_rope);

        rope_backward(attn_delta_states.delta_mu_q_rope,
                      attn_delta_states.delta_var_q_rope, this->cos_cache,
                      this->sin_cache, batch_size, num_heads, this->seq_len,
                      this->head_dim, attn_delta_states.delta_mu_q,
                      attn_delta_states.delta_var_q);

        rope_backward(attn_delta_states.delta_mu_k_rope,
                      attn_delta_states.delta_var_k_rope, this->cos_cache,
                      this->sin_cache, batch_size, num_heads, this->seq_len,
                      this->head_dim, attn_delta_states.delta_mu_k,
                      attn_delta_states.delta_var_k);
    } else {
        mha_delta_query(attn_states.var_q, attn_states.mu_k,
                        attn_delta_states.delta_mu_att_score,
                        attn_delta_states.delta_var_att_score,
                        attn_states.j_mqk, batch_size, num_heads, this->seq_len,
                        this->head_dim, attn_delta_states.delta_mu_q,
                        attn_delta_states.delta_var_q);

        mha_delta_key(attn_states.var_k, attn_states.mu_q,
                      attn_delta_states.delta_mu_att_score,
                      attn_delta_states.delta_var_att_score, attn_states.j_mqk,
                      batch_size, num_heads, this->seq_len, this->head_dim,
                      attn_delta_states.delta_mu_k,
                      attn_delta_states.delta_var_k);
    }

    cat_intput_projection_components(
        attn_delta_states.delta_mu_q, attn_delta_states.delta_var_q,
        attn_delta_states.delta_mu_k, attn_delta_states.delta_var_k,
        attn_delta_states.delta_mu_v, attn_delta_states.delta_var_v, batch_size,
        num_heads, this->seq_len, this->head_dim,
        attn_delta_states.delta_mu_in_proj,
        attn_delta_states.delta_var_in_proj);

    if (state_udapte) {
        linear_bwd_fc_delta_z_mp(
            this->mu_w, this->bwd_states->jcb,
            attn_delta_states.delta_mu_in_proj,
            attn_delta_states.delta_var_in_proj, input_qkv_size,
            output_qkv_size, batch_seq_len, this->num_threads,
            output_delta_states.delta_mu, output_delta_states.delta_var);
    }

    if (this->param_update) {
        // TODO: mu_out_proj or this->bwd_states->mu_a?
        linear_bwd_fc_delta_w_mp(this->var_w, this->bwd_states->mu_a,
                                 attn_delta_states.delta_mu_in_proj,
                                 attn_delta_states.delta_var_in_proj,
                                 input_qkv_size, output_qkv_size, batch_seq_len,
                                 this->num_threads, this->delta_mu_w,
                                 this->delta_var_w);

        if (this->bias) {
            linear_bwd_fc_delta_b_mp(
                this->var_b, attn_delta_states.delta_mu_in_proj,
                attn_delta_states.delta_var_in_proj, output_qkv_size,
                batch_seq_len, this->num_threads, this->delta_mu_b,
                this->delta_var_b);
        }
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> MultiheadAttention::to_cuda(int device_idx) {
    this->device = "cuda";
    this->device_idx = device_idx;
    LOG(LogLevel::ERROR,
        "CUDA support for MultiheadAttention not yet implemented");
    return nullptr;
}
#endif
