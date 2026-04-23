#include "../include/attention.h"

#include <cstdio>

#include "../include/activation.h"
#include "../include/common.h"
#include "../include/custom_logger.h"
#include "../include/linear_layer.h"
#ifdef USE_CUDA
#include "../include/attention_cuda.cuh"
#endif

static void causal_mask_pre_remax(const std::vector<float> &mu_qk,
                                  const std::vector<float> &var_qk,
                                  int batch_size, int num_heads, int timestep,
                                  std::vector<float> &mu_mqk,
                                  std::vector<float> &var_mqk) {
    constexpr float MASK_MU = -1e4f;
    constexpr float MASK_VAR = 1e-4f;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l < timestep; l++) {
                    int idx = i * num_heads * timestep * timestep +
                              j * timestep * timestep + k * timestep + l;
                    if (l <= k) {
                        mu_mqk[idx] = mu_qk[idx];
                        var_mqk[idx] = var_qk[idx];
                    } else {
                        mu_mqk[idx] = MASK_MU;
                        var_mqk[idx] = MASK_VAR;
                    }
                }
            }
        }
    }
}

static void causal_mask_post_remax(std::vector<float> &mu_att,
                                   std::vector<float> &var_att,
                                   std::vector<float> &jcb, int batch_size,
                                   int num_heads, int timestep) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = k + 1; l < timestep; l++) {
                    int idx = i * num_heads * timestep * timestep +
                              j * timestep * timestep + k * timestep + l;
                    mu_att[idx] = 0.0f;
                    var_att[idx] = 0.0f;
                    jcb[idx] = 0.0f;
                }
            }
        }
    }
}

void print_magnitude_stats(const char *name, const std::vector<float> &mu,
                           const std::vector<float> &var) {
    if (mu.empty()) {
        std::printf("[attn-diag] %s: empty\n", name);
        return;
    }
    float mu_sum = 0.0f, mu_sq_sum = 0.0f;
    float mu_abs_sum = 0.0f, mu_abs_max = 0.0f;
    float var_sum = 0.0f, var_max = 0.0f;
    for (size_t i = 0; i < mu.size(); i++) {
        float m = mu[i];
        mu_sum += m;
        mu_sq_sum += m * m;
        float a = std::fabs(m);
        mu_abs_sum += a;
        if (a > mu_abs_max) mu_abs_max = a;
        float v = var[i];
        var_sum += v;
        if (v > var_max) var_max = v;
    }
    size_t n = mu.size();
    float mu_mean = mu_sum / n;
    float mu_std = std::sqrt(std::max(0.0f, mu_sq_sum / n - mu_mean * mu_mean));
    std::printf(
        "[attn-diag] %-10s n=%zu  mu mean=%.4e std=%.4e  |mu| mean=%.4e "
        "max=%.4e  var mean=%.4e max=%.4e\n",
        name, n, mu_mean, mu_std, mu_abs_sum / n, mu_abs_max, var_sum / n,
        var_max);
}

void print_magnitude_stats_causal(const char *name,
                                  const std::vector<float> &mu,
                                  const std::vector<float> &var, int batch_size,
                                  int num_heads, int timestep) {
    float mu_sum = 0.0f, mu_sq_sum = 0.0f;
    float mu_abs_sum = 0.0f, mu_abs_max = 0.0f;
    float var_sum = 0.0f, var_max = 0.0f;
    size_t n = 0;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int l = 0; l <= k; l++) {
                    int idx = i * num_heads * timestep * timestep +
                              j * timestep * timestep + k * timestep + l;
                    float m = mu[idx];
                    mu_sum += m;
                    mu_sq_sum += m * m;
                    float a = std::fabs(m);
                    mu_abs_sum += a;
                    if (a > mu_abs_max) mu_abs_max = a;
                    float v = var[idx];
                    var_sum += v;
                    if (v > var_max) var_max = v;
                    n++;
                }
            }
        }
    }
    float mu_mean = mu_sum / n;
    float mu_std = std::sqrt(std::max(0.0f, mu_sq_sum / n - mu_mean * mu_mean));
    std::printf(
        "[attn-diag] %-10s n=%zu  mu mean=%.4e std=%.4e  |mu| mean=%.4e "
        "max=%.4e  var mean=%.4e max=%.4e\n",
        name, n, mu_mean, mu_std, mu_abs_sum / n, mu_abs_max, var_sum / n,
        var_max);
}

void separate_input_projection_components(
    std::vector<float> &mu_embs, std::vector<float> &var_embs, int batch_size,
    int num_heads, int timestep, int head_dim, std::vector<float> &mu_q,
    std::vector<float> &var_q, std::vector<float> &mu_k,
    std::vector<float> &var_k, std::vector<float> &mu_v,
    std::vector<float> &var_v)
/*Separate input projection components into query, key, and value.

The linear layer outputs (batch_size * timestep, 3 * num_heads * head_dim)
in row-major order, so for each token the layout is [Q(C) | K(C) | V(C)]
where C = num_heads * head_dim.

embs: [batch_size * timestep, 3 * num_heads * head_dim]
q: [batch_size, num_heads, timestep, head_dim]
k: [batch_size, num_heads, timestep, head_dim]
v: [batch_size, num_heads, timestep, head_dim]
*/
{
    int comp_idx, emb_idx_q, emb_idx_k, emb_idx_v;
    int emb_size = num_heads * head_dim;
    int row_size = 3 * emb_size;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                for (int m = 0; m < head_dim; m++) {
                    comp_idx = i * num_heads * timestep * head_dim +
                               j * timestep * head_dim + k * head_dim + m;
                    int token_idx = i * timestep + k;
                    int head_offset = j * head_dim + m;
                    emb_idx_q = token_idx * row_size + head_offset;
                    emb_idx_k = token_idx * row_size + emb_size + head_offset;
                    emb_idx_v =
                        token_idx * row_size + 2 * emb_size + head_offset;

                    mu_q[comp_idx] = mu_embs[emb_idx_q];
                    var_q[comp_idx] = var_embs[emb_idx_q];

                    mu_k[comp_idx] = mu_embs[emb_idx_k];
                    var_k[comp_idx] = var_embs[emb_idx_k];

                    mu_v[comp_idx] = mu_embs[emb_idx_v];
                    var_v[comp_idx] = var_embs[emb_idx_v];
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
/*Concatenate query, key, and value vectors back into linear layer layout.

The output must match the linear layer's (batch_size * timestep, 3*C) layout
where each token row is [Q(C) | K(C) | V(C)] and C = num_heads * head_size.

q: [batch_size, num_heads, timestep, head_dim]
k: [batch_size, num_heads, timestep, head_dim]
v: [batch_size, num_heads, timestep, head_dim]
embs: [batch_size * timestep, 3 * num_heads * head_size]
*/
{
    int qkv_idx, emb_idx_q, emb_idx_k, emb_idx_v;
    int emb_size = num_heads * head_size;
    int row_size = 3 * emb_size;
    for (int i = 0; i < batch_size; i++) {
        for (int k = 0; k < timestep; k++) {
            for (int j = 0; j < num_heads; j++) {
                for (int m = 0; m < head_size; m++) {
                    qkv_idx = i * num_heads * timestep * head_size +
                              j * timestep * head_size + k * head_size + m;
                    int token_idx = i * timestep + k;
                    int head_offset = j * head_size + m;
                    emb_idx_q = token_idx * row_size + head_offset;
                    emb_idx_k = token_idx * row_size + emb_size + head_offset;
                    emb_idx_v =
                        token_idx * row_size + 2 * emb_size + head_offset;

                    mu_embs[emb_idx_q] = mu_q[qkv_idx];
                    var_embs[emb_idx_q] = var_q[qkv_idx];

                    mu_embs[emb_idx_k] = mu_k[qkv_idx];
                    var_embs[emb_idx_k] = var_k[qkv_idx];

                    mu_embs[emb_idx_v] = mu_v[qkv_idx];
                    var_embs[emb_idx_v] = var_v[qkv_idx];
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
    float scale = 1.0f / sqrtf(static_cast<float>(head_size));
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
                    mu_qk[idx_qk] = sum_mu * scale;
                    var_qk[idx_qk] = sum_var * scale * scale;
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
    float scale = 1.0f / sqrtf(static_cast<float>(head_size));
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int m = 0; m < head_size; m++) {
                for (int k = 0; k < timestep; k++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int l = 0; l < timestep; l++) {
                        idx_k = i * num_heads * timestep * head_size +
                                j * timestep * head_size + l * head_size + m;
                        idx_s = i * num_heads * timestep * timestep +
                                j * timestep * timestep + k * timestep + l;
                        sum_mu +=
                            mu_k[idx_k] * delta_mu[idx_s] * jcb_mqk[idx_s];
                        sum_var += mu_k[idx_k] * delta_var[idx_s] *
                                   mu_k[idx_k] * jcb_mqk[idx_s] *
                                   jcb_mqk[idx_s];
                    }
                    idx_q = i * num_heads * timestep * head_size +
                            j * timestep * head_size + m + k * head_size;

                    delta_mu_q[idx_q] = sum_mu * scale;
                    delta_var_q[idx_q] = sum_var * scale * scale;
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
    int idx_q, idx_s, idx_k;
    float sum_mu, sum_var;
    float scale = 1.0f / sqrtf(static_cast<float>(head_size));
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int m = 0; m < head_size; m++) {
                for (int k = 0; k < timestep; k++) {
                    sum_mu = 0.0f;
                    sum_var = 0.0f;
                    for (int p = 0; p < timestep; p++) {
                        idx_s = i * num_heads * timestep * timestep +
                                j * timestep * timestep + p * timestep + k;
                        idx_q = i * num_heads * timestep * head_size +
                                j * timestep * head_size + p * head_size + m;

                        sum_mu +=
                            mu_q[idx_q] * delta_mu[idx_s] * jcb_mqk[idx_s];
                        sum_var += mu_q[idx_q] * delta_var[idx_s] *
                                   mu_q[idx_q] * jcb_mqk[idx_s] *
                                   jcb_mqk[idx_s];
                    }
                    idx_k = i * num_heads * timestep * head_size +
                            j * timestep * head_size + k * head_size + m;

                    delta_mu_k[idx_k] = sum_mu * scale;
                    delta_var_k[idx_k] = sum_var * scale * scale;
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

    float log_theta = -logf(theta) / head_dim;
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = expf((2.0f * i) * log_theta);
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

    mu_q_pe.resize(comp_size, 0.0f);
    var_q_pe.resize(comp_size, 0.0f);
    mu_k_pe.resize(comp_size, 0.0f);
    var_k_pe.resize(comp_size, 0.0f);

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
    delta_mu_q_pe.resize(comp_size, 0.0f);
    delta_var_q_pe.resize(comp_size, 0.0f);
    delta_mu_k_pe.resize(comp_size, 0.0f);
    delta_var_k_pe.resize(comp_size, 0.0f);
    delta_mu_in_proj.resize(3 * comp_size, 0.0f);
    delta_var_in_proj.resize(3 * comp_size, 0.0f);
}

MultiheadAttention::MultiheadAttention(size_t embed_dim, size_t num_heads,
                                       size_t num_kv_heads, size_t seq_len_,
                                       bool bias, float gain_w, float gain_b,
                                       std::string init_method,
                                       std::string pos_emb, float rope_theta,
                                       size_t max_seq_len, bool use_causal_mask,
                                       int device_idx)
    : embed_dim(embed_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method),
      pos_emb(pos_emb),
      rope_theta(rope_theta),
      max_seq_len(max_seq_len),
      use_causal_mask(use_causal_mask) {
    this->input_size = embed_dim;
    this->output_size = this->embed_dim;
    this->seq_len = seq_len_;
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

    if (this->pos_emb == "rope") {
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
    int qkv_output = (num_heads + 2 * num_kv_heads) * head_dim;

    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_linear(this->init_method, this->gain_w, this->gain_b,
                                this->embed_dim, qkv_output, this->num_weights,
                                this->num_biases);
}

void MultiheadAttention::forward(BaseHiddenStates &input_states,
                                 BaseHiddenStates &output_states,
                                 BaseTempStates &temp_states) {
    // TODO: check it is correct for 2 consecutive attention layers
    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size * this->seq_len);

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

    if (this->pos_emb == "rope") {
        apply_rope(attn_states.mu_q, attn_states.var_q, this->cos_cache,
                   this->sin_cache, batch_size, num_heads, this->seq_len,
                   head_dim, attn_states.mu_q_pe, attn_states.var_q_pe);

        apply_rope(attn_states.mu_k, attn_states.var_k, this->cos_cache,
                   this->sin_cache, batch_size, num_heads, this->seq_len,
                   head_dim, attn_states.mu_k_pe, attn_states.var_k_pe);

        query_key(attn_states.mu_q_pe, attn_states.var_q_pe,
                  attn_states.mu_k_pe, attn_states.var_k_pe, batch_size,
                  num_heads, this->seq_len, head_dim, attn_states.mu_qk,
                  attn_states.var_qk);
    } else {
        query_key(attn_states.mu_q, attn_states.var_q, attn_states.mu_k,
                  attn_states.var_k, batch_size, num_heads, this->seq_len,
                  head_dim, attn_states.mu_qk, attn_states.var_qk);
    }

    if (this->use_causal_mask) {
        causal_mask_pre_remax(attn_states.mu_qk, attn_states.var_qk, batch_size,
                              num_heads, this->seq_len, attn_states.mu_mqk,
                              attn_states.var_mqk);
    }

    // Apply Remax (probabilistic softmax) on query-key product
    int qk_size = batch_size * num_heads * this->seq_len * this->seq_len;
    remax_input.mu_a =
        this->use_causal_mask ? attn_states.mu_mqk : attn_states.mu_qk;
    remax_input.var_a =
        this->use_causal_mask ? attn_states.var_mqk : attn_states.var_qk;
    remax_input.block_size = batch_size * this->seq_len * this->num_heads;
    remax_input.actual_size = this->seq_len;

    remax_output.set_size(qk_size,
                          batch_size * this->seq_len * this->num_heads);

    if (this->debug) {
        std::printf("[attn-diag] MHA pre-remax (rope=%s, mask=%d)\n",
                    this->pos_emb.c_str(), (int)this->use_causal_mask);
        print_magnitude_stats("W_qkv", this->mu_w, this->var_w);
        if (this->bias) {
            print_magnitude_stats("b_qkv", this->mu_b, this->var_b);
        }
        if (this->pos_emb == "rope") {
            print_magnitude_stats("Q(rope)", attn_states.mu_q_pe,
                                  attn_states.var_q_pe);
            print_magnitude_stats("K(rope)", attn_states.mu_k_pe,
                                  attn_states.var_k_pe);
        } else {
            print_magnitude_stats("Q", attn_states.mu_q, attn_states.var_q);
            print_magnitude_stats("K", attn_states.mu_k, attn_states.var_k);
        }
        print_magnitude_stats("V", attn_states.mu_v, attn_states.var_v);
        if (this->use_causal_mask) {
            print_magnitude_stats_causal("QK", remax_input.mu_a,
                                         remax_input.var_a, batch_size,
                                         num_heads, this->seq_len);
        } else {
            print_magnitude_stats("QK", remax_input.mu_a, remax_input.var_a);
        }
    }

    remax_layer->forward(remax_input, remax_output, remax_temp);

    if (this->use_causal_mask) {
        causal_mask_post_remax(remax_output.mu_a, remax_output.var_a,
                               remax_output.jcb, batch_size, num_heads,
                               this->seq_len);
    }

    attn_states.mu_att_score = remax_output.mu_a;
    attn_states.var_att_score = remax_output.var_a;
    attn_states.j_mqk = remax_output.jcb;

    if (this->debug) {
        if (this->use_causal_mask) {
            print_magnitude_stats_causal("att_score", attn_states.mu_att_score,
                                         attn_states.var_att_score, batch_size,
                                         num_heads, this->seq_len);
            print_magnitude_stats_causal("j_mqk", attn_states.j_mqk,
                                         attn_states.j_mqk, batch_size,
                                         num_heads, this->seq_len);
        } else {
            print_magnitude_stats("att_score", attn_states.mu_att_score,
                                  attn_states.var_att_score);
            print_magnitude_stats("j_mqk", attn_states.j_mqk,
                                  attn_states.j_mqk);
        }
    }

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
    output_states.seq_len = this->seq_len;
    output_states.actual_size = this->output_size;

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

    mha_delta_score(attn_states.mu_v, attn_delta_states.delta_mu_buffer,
                    attn_delta_states.delta_var_buffer, batch_size,
                    this->num_heads, this->seq_len, this->head_dim,
                    attn_delta_states.delta_mu_att_score,
                    attn_delta_states.delta_var_att_score);

    if (this->pos_emb == "rope") {
        mha_delta_query(attn_states.var_q, attn_states.mu_k_pe,
                        attn_delta_states.delta_mu_att_score,
                        attn_delta_states.delta_var_att_score,
                        attn_states.j_mqk, batch_size, num_heads, this->seq_len,
                        this->head_dim, attn_delta_states.delta_mu_q_pe,
                        attn_delta_states.delta_var_q_pe);

        mha_delta_key(attn_states.var_k, attn_states.mu_q_pe,
                      attn_delta_states.delta_mu_att_score,
                      attn_delta_states.delta_var_att_score, attn_states.j_mqk,
                      batch_size, num_heads, this->seq_len, this->head_dim,
                      attn_delta_states.delta_mu_k_pe,
                      attn_delta_states.delta_var_k_pe);

        rope_backward(attn_delta_states.delta_mu_q_pe,
                      attn_delta_states.delta_var_q_pe, this->cos_cache,
                      this->sin_cache, batch_size, num_heads, this->seq_len,
                      this->head_dim, attn_delta_states.delta_mu_q,
                      attn_delta_states.delta_var_q);

        rope_backward(attn_delta_states.delta_mu_k_pe,
                      attn_delta_states.delta_var_k_pe, this->cos_cache,
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

AttentionScores MultiheadAttention::get_attention_scores() {
    AttentionScores s;
    s.num_heads = (int)this->num_heads;
    s.timestep = (int)this->seq_len;
    int per_batch = s.num_heads * s.timestep * s.timestep;
    if (per_batch <= 0) return s;
    s.batch_size = (int)(attn_states.mu_att_score.size() / per_batch);
    s.mu = attn_states.mu_att_score;
    s.var = attn_states.var_att_score;
    return s;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> MultiheadAttention::to_cuda(int device_idx) {
    this->device = "cuda";
    this->device_idx = device_idx;
    auto cuda_layer = std::make_unique<MultiheadAttentionCuda>(
        this->embed_dim, this->num_heads, this->num_kv_heads, this->seq_len,
        this->bias, this->gain_w, this->gain_b, this->init_method,
        this->pos_emb, this->rope_theta, this->max_seq_len,
        this->use_causal_mask, device_idx);
    auto base_cuda = dynamic_cast<BaseLayerCuda *>(cuda_layer.get());
    base_cuda->copy_params_from(*this);
    return cuda_layer;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// MultiheadAttentionV2 — Separate Q, K, V projections
////////////////////////////////////////////////////////////////////////////////

// Reshape linear output [batch*seq_len, num_heads*head_dim] to
// [batch, num_heads, seq_len, head_dim]
static void reshape_proj_to_heads(std::vector<float> &mu_proj,
                                  std::vector<float> &var_proj, int batch_size,
                                  int num_heads, int seq_len, int head_dim,
                                  std::vector<float> &mu_out,
                                  std::vector<float> &var_out) {
    int emb_size = num_heads * head_dim;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < seq_len; k++) {
                for (int m = 0; m < head_dim; m++) {
                    int comp_idx = i * num_heads * seq_len * head_dim +
                                   j * seq_len * head_dim + k * head_dim + m;
                    int token_idx = i * seq_len + k;
                    int proj_idx = token_idx * emb_size + j * head_dim + m;
                    mu_out[comp_idx] = mu_proj[proj_idx];
                    var_out[comp_idx] = var_proj[proj_idx];
                }
            }
        }
    }
}

// Reshape [batch, num_heads, seq_len, head_dim] back to
// [batch*seq_len, num_heads*head_dim]
static void reshape_heads_to_proj(std::vector<float> &mu_heads,
                                  std::vector<float> &var_heads, int batch_size,
                                  int num_heads, int seq_len, int head_dim,
                                  std::vector<float> &mu_out,
                                  std::vector<float> &var_out) {
    int emb_size = num_heads * head_dim;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < seq_len; k++) {
                for (int m = 0; m < head_dim; m++) {
                    int comp_idx = i * num_heads * seq_len * head_dim +
                                   j * seq_len * head_dim + k * head_dim + m;
                    int token_idx = i * seq_len + k;
                    int proj_idx = token_idx * emb_size + j * head_dim + m;
                    mu_out[proj_idx] = mu_heads[comp_idx];
                    var_out[proj_idx] = var_heads[comp_idx];
                }
            }
        }
    }
}

static void capped_update(std::vector<float> &mu, std::vector<float> &var,
                          std::vector<float> &delta_mu,
                          std::vector<float> &delta_var, float cap_factor) {
    for (size_t i = 0; i < mu.size(); i++) {
        float delta_mu_sign = (delta_mu[i] > 0) - (delta_mu[i] < 0);
        float delta_var_sign = (delta_var[i] > 0) - (delta_var[i] < 0);
        float delta_bar = powf(var[i], 0.5f) / cap_factor;

        mu[i] += delta_mu_sign * std::min(std::abs(delta_mu[i]), delta_bar);
        var[i] += delta_var_sign * std::min(std::abs(delta_var[i]), delta_bar);
        if (var[i] <= 0.0f) {
            var[i] = 1E-5f;
        }
    }
}

MultiheadAttentionV2::MultiheadAttentionV2(
    size_t embed_dim, size_t num_heads, size_t num_kv_heads, size_t seq_len_,
    bool bias, float gain_w, float gain_b, std::string init_method,
    std::string pos_emb, float rope_theta, size_t max_seq_len,
    bool use_causal_mask, int device_idx)
    : embed_dim(embed_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method),
      pos_emb(pos_emb),
      rope_theta(rope_theta),
      max_seq_len(max_seq_len),
      use_causal_mask(use_causal_mask) {
    this->input_size = embed_dim;
    this->output_size = embed_dim;
    this->seq_len = seq_len_;
    this->head_dim = embed_dim / num_heads;
    this->bias = bias;
    this->device_idx = device_idx;

    q_output_size = num_heads * this->head_dim;
    k_output_size = num_kv_heads * this->head_dim;
    v_output_size = num_kv_heads * this->head_dim;

    num_weights_q = embed_dim * q_output_size;
    num_weights_k = embed_dim * k_output_size;
    num_weights_v = embed_dim * v_output_size;

    num_biases_q = 0;
    num_biases_k = 0;
    num_biases_v = 0;
    if (this->bias) {
        num_biases_q = q_output_size;
        num_biases_k = k_output_size;
        num_biases_v = v_output_size;
    }

    // BaseLayer num_weights/num_biases set to 0 — we manage our own
    this->num_weights = 0;
    this->num_biases = 0;

    if (this->device.compare("cpu") == 0) {
        this->init_weight_bias();
    }

    if (this->training && this->device.compare("cpu") == 0) {
        this->allocate_param_delta();
    }

    remax_layer = std::make_unique<Remax>();

    if (this->pos_emb == "rope") {
        generate_rope_cache(this->max_seq_len, this->head_dim, this->rope_theta,
                            this->cos_cache, this->sin_cache);
    }
}

MultiheadAttentionV2::~MultiheadAttentionV2() {}

std::string MultiheadAttentionV2::get_layer_info() const {
    return "SelfAttentionV2(heads=" + std::to_string(this->num_heads) +
           ", kv_heads=" + std::to_string(this->num_kv_heads) +
           ", emb_size=" + std::to_string(this->embed_dim) + ")";
}

std::string MultiheadAttentionV2::get_layer_name() const {
    return "MultiheadAttentionV2";
}

LayerType MultiheadAttentionV2::get_layer_type() const {
    return LayerType::MultiheadAttention;
}

void MultiheadAttentionV2::init_weight_bias() {
    auto init_q = init_weight_bias_linear(
        this->init_method, this->gain_w, this->gain_b, this->embed_dim,
        q_output_size, num_weights_q, num_biases_q);
    mu_w_q = std::get<0>(init_q);
    var_w_q = std::get<1>(init_q);
    mu_b_q = std::get<2>(init_q);
    var_b_q = std::get<3>(init_q);

    auto init_k = init_weight_bias_linear(
        this->init_method, this->gain_w, this->gain_b, this->embed_dim,
        k_output_size, num_weights_k, num_biases_k);
    mu_w_k = std::get<0>(init_k);
    var_w_k = std::get<1>(init_k);
    mu_b_k = std::get<2>(init_k);
    var_b_k = std::get<3>(init_k);

    auto init_v = init_weight_bias_linear(
        this->init_method, this->gain_w, this->gain_b, this->embed_dim,
        v_output_size, num_weights_v, num_biases_v);
    mu_w_v = std::get<0>(init_v);
    var_w_v = std::get<1>(init_v);
    mu_b_v = std::get<2>(init_v);
    var_b_v = std::get<3>(init_v);
}

void MultiheadAttentionV2::allocate_param_delta() {
    delta_mu_w_q.resize(num_weights_q, 0.0f);
    delta_var_w_q.resize(num_weights_q, 0.0f);
    delta_mu_w_k.resize(num_weights_k, 0.0f);
    delta_var_w_k.resize(num_weights_k, 0.0f);
    delta_mu_w_v.resize(num_weights_v, 0.0f);
    delta_var_w_v.resize(num_weights_v, 0.0f);

    delta_mu_b_q.resize(num_biases_q, 0.0f);
    delta_var_b_q.resize(num_biases_q, 0.0f);
    delta_mu_b_k.resize(num_biases_k, 0.0f);
    delta_var_b_k.resize(num_biases_k, 0.0f);
    delta_mu_b_v.resize(num_biases_v, 0.0f);
    delta_var_b_v.resize(num_biases_v, 0.0f);
}

void MultiheadAttentionV2::update_weights() {
    capped_update(mu_w_q, var_w_q, delta_mu_w_q, delta_var_w_q,
                  this->cap_factor_update);
    capped_update(mu_w_k, var_w_k, delta_mu_w_k, delta_var_w_k,
                  this->cap_factor_update);
    capped_update(mu_w_v, var_w_v, delta_mu_w_v, delta_var_w_v,
                  this->cap_factor_update);
}

void MultiheadAttentionV2::update_biases() {
    if (this->bias) {
        capped_update(mu_b_q, var_b_q, delta_mu_b_q, delta_var_b_q,
                      this->cap_factor_update);
        capped_update(mu_b_k, var_b_k, delta_mu_b_k, delta_var_b_k,
                      this->cap_factor_update);
        capped_update(mu_b_v, var_b_v, delta_mu_b_v, delta_var_b_v,
                      this->cap_factor_update);
    }
}

void MultiheadAttentionV2::forward(BaseHiddenStates &input_states,
                                   BaseHiddenStates &output_states,
                                   BaseTempStates &temp_states) {
    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size * this->seq_len);
    int batch_seq = batch_size * this->seq_len;

    attn_states.set_size(batch_size, num_heads, this->seq_len, head_dim);

    // Resize projection buffers
    int q_proj_size = batch_seq * q_output_size;
    int k_proj_size = batch_seq * k_output_size;
    int v_proj_size = batch_seq * v_output_size;
    mu_q_proj.resize(q_proj_size);
    var_q_proj.resize(q_proj_size);
    mu_k_proj.resize(k_proj_size);
    var_k_proj.resize(k_proj_size);
    mu_v_proj.resize(v_proj_size);
    var_v_proj.resize(v_proj_size);

    // Separate Q, K, V linear projections
    linear_fwd_mean_var_mp(mu_w_q, var_w_q, mu_b_q, var_b_q, input_states.mu_a,
                           input_states.var_a, this->embed_dim, q_output_size,
                           batch_seq, this->bias, this->num_threads, mu_q_proj,
                           var_q_proj);

    linear_fwd_mean_var_mp(mu_w_k, var_w_k, mu_b_k, var_b_k, input_states.mu_a,
                           input_states.var_a, this->embed_dim, k_output_size,
                           batch_seq, this->bias, this->num_threads, mu_k_proj,
                           var_k_proj);

    linear_fwd_mean_var_mp(mu_w_v, var_w_v, mu_b_v, var_b_v, input_states.mu_a,
                           input_states.var_a, this->embed_dim, v_output_size,
                           batch_seq, this->bias, this->num_threads, mu_v_proj,
                           var_v_proj);

    // Reshape to [batch, heads, seq_len, head_dim]
    reshape_proj_to_heads(mu_q_proj, var_q_proj, batch_size, num_heads,
                          this->seq_len, head_dim, attn_states.mu_q,
                          attn_states.var_q);
    reshape_proj_to_heads(mu_k_proj, var_k_proj, batch_size, num_heads,
                          this->seq_len, head_dim, attn_states.mu_k,
                          attn_states.var_k);
    reshape_proj_to_heads(mu_v_proj, var_v_proj, batch_size, num_heads,
                          this->seq_len, head_dim, attn_states.mu_v,
                          attn_states.var_v);

    if (this->pos_emb == "rope") {
        apply_rope(attn_states.mu_q, attn_states.var_q, this->cos_cache,
                   this->sin_cache, batch_size, num_heads, this->seq_len,
                   head_dim, attn_states.mu_q_pe, attn_states.var_q_pe);

        apply_rope(attn_states.mu_k, attn_states.var_k, this->cos_cache,
                   this->sin_cache, batch_size, num_heads, this->seq_len,
                   head_dim, attn_states.mu_k_pe, attn_states.var_k_pe);

        query_key(attn_states.mu_q_pe, attn_states.var_q_pe,
                  attn_states.mu_k_pe, attn_states.var_k_pe, batch_size,
                  num_heads, this->seq_len, head_dim, attn_states.mu_qk,
                  attn_states.var_qk);
    } else {
        query_key(attn_states.mu_q, attn_states.var_q, attn_states.mu_k,
                  attn_states.var_k, batch_size, num_heads, this->seq_len,
                  head_dim, attn_states.mu_qk, attn_states.var_qk);
    }

    if (this->use_causal_mask) {
        causal_mask_pre_remax(attn_states.mu_qk, attn_states.var_qk, batch_size,
                              num_heads, this->seq_len, attn_states.mu_mqk,
                              attn_states.var_mqk);
    }

    // Apply Remax on query-key product
    int qk_size = batch_size * num_heads * this->seq_len * this->seq_len;
    remax_input.mu_a =
        this->use_causal_mask ? attn_states.mu_mqk : attn_states.mu_qk;
    remax_input.var_a =
        this->use_causal_mask ? attn_states.var_mqk : attn_states.var_qk;
    remax_input.block_size = batch_size * this->seq_len * this->num_heads;
    remax_input.actual_size = this->seq_len;

    remax_output.set_size(qk_size,
                          batch_size * this->seq_len * this->num_heads);

    if (this->debug) {
        std::printf("[attn-diag] MHAv2 pre-remax (rope=%s, mask=%d)\n",
                    this->pos_emb.c_str(), (int)this->use_causal_mask);
        print_magnitude_stats("W_q", mu_w_q, var_w_q);
        print_magnitude_stats("W_k", mu_w_k, var_w_k);
        print_magnitude_stats("W_v", mu_w_v, var_w_v);
        if (this->bias) {
            print_magnitude_stats("b_q", mu_b_q, var_b_q);
            print_magnitude_stats("b_k", mu_b_k, var_b_k);
            print_magnitude_stats("b_v", mu_b_v, var_b_v);
        }
        if (this->pos_emb == "rope") {
            print_magnitude_stats("Q(rope)", attn_states.mu_q_pe,
                                  attn_states.var_q_pe);
            print_magnitude_stats("K(rope)", attn_states.mu_k_pe,
                                  attn_states.var_k_pe);
        } else {
            print_magnitude_stats("Q", attn_states.mu_q, attn_states.var_q);
            print_magnitude_stats("K", attn_states.mu_k, attn_states.var_k);
        }
        print_magnitude_stats("V", attn_states.mu_v, attn_states.var_v);
        if (this->use_causal_mask) {
            print_magnitude_stats_causal("QK", remax_input.mu_a,
                                         remax_input.var_a, batch_size,
                                         num_heads, this->seq_len);
        } else {
            print_magnitude_stats("QK", remax_input.mu_a, remax_input.var_a);
        }
    }

    remax_layer->forward(remax_input, remax_output, remax_temp);

    if (this->use_causal_mask) {
        causal_mask_post_remax(remax_output.mu_a, remax_output.var_a,
                               remax_output.jcb, batch_size, num_heads,
                               this->seq_len);
    }

    if (this->debug) {
        if (this->use_causal_mask) {
            print_magnitude_stats_causal("att_score", remax_output.mu_a,
                                         remax_output.var_a, batch_size,
                                         num_heads, this->seq_len);
            print_magnitude_stats_causal("j_mqk", remax_output.jcb,
                                         remax_output.jcb, batch_size,
                                         num_heads, this->seq_len);
        } else {
            print_magnitude_stats("att_score", remax_output.mu_a,
                                  remax_output.var_a);
            print_magnitude_stats("j_mqk", remax_output.jcb, remax_output.jcb);
        }
    }

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
    output_states.seq_len = this->seq_len;
    output_states.actual_size = this->output_size;

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void MultiheadAttentionV2::backward(BaseDeltaStates &input_delta_states,
                                    BaseDeltaStates &output_delta_states,
                                    BaseTempStates &temp_states,
                                    bool state_udapte) {
    int batch_size = input_delta_states.block_size;
    attn_delta_states.set_size(batch_size, num_heads, this->seq_len, head_dim);
    int batch_seq = batch_size * this->seq_len;

    project_output_backward(
        input_delta_states.delta_mu, input_delta_states.delta_var, batch_size,
        this->num_heads, this->seq_len, this->head_dim,
        attn_delta_states.delta_mu_buffer, attn_delta_states.delta_var_buffer);

    mha_delta_value(attn_states.mu_att_score, attn_delta_states.delta_mu_buffer,
                    attn_delta_states.delta_var_buffer, batch_size,
                    this->num_heads, this->seq_len, this->head_dim,
                    attn_delta_states.delta_mu_v,
                    attn_delta_states.delta_var_v);

    mha_delta_score(attn_states.mu_v, attn_delta_states.delta_mu_buffer,
                    attn_delta_states.delta_var_buffer, batch_size,
                    this->num_heads, this->seq_len, this->head_dim,
                    attn_delta_states.delta_mu_att_score,
                    attn_delta_states.delta_var_att_score);

    if (this->pos_emb == "rope") {
        mha_delta_query(attn_states.var_q, attn_states.mu_k_pe,
                        attn_delta_states.delta_mu_att_score,
                        attn_delta_states.delta_var_att_score,
                        attn_states.j_mqk, batch_size, num_heads, this->seq_len,
                        this->head_dim, attn_delta_states.delta_mu_q_pe,
                        attn_delta_states.delta_var_q_pe);

        mha_delta_key(attn_states.var_k, attn_states.mu_q_pe,
                      attn_delta_states.delta_mu_att_score,
                      attn_delta_states.delta_var_att_score, attn_states.j_mqk,
                      batch_size, num_heads, this->seq_len, this->head_dim,
                      attn_delta_states.delta_mu_k_pe,
                      attn_delta_states.delta_var_k_pe);

        rope_backward(attn_delta_states.delta_mu_q_pe,
                      attn_delta_states.delta_var_q_pe, this->cos_cache,
                      this->sin_cache, batch_size, num_heads, this->seq_len,
                      this->head_dim, attn_delta_states.delta_mu_q,
                      attn_delta_states.delta_var_q);

        rope_backward(attn_delta_states.delta_mu_k_pe,
                      attn_delta_states.delta_var_k_pe, this->cos_cache,
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

    // Reshape Q/K/V deltas from [batch, heads, seq, hd] to [batch*seq,
    // heads*hd]
    int q_proj_size = batch_seq * q_output_size;
    int k_proj_size = batch_seq * k_output_size;
    int v_proj_size = batch_seq * v_output_size;

    std::vector<float> delta_mu_q_proj(q_proj_size);
    std::vector<float> delta_var_q_proj(q_proj_size);
    std::vector<float> delta_mu_k_proj(k_proj_size);
    std::vector<float> delta_var_k_proj(k_proj_size);
    std::vector<float> delta_mu_v_proj(v_proj_size);
    std::vector<float> delta_var_v_proj(v_proj_size);

    reshape_heads_to_proj(
        attn_delta_states.delta_mu_q, attn_delta_states.delta_var_q, batch_size,
        num_heads, this->seq_len, head_dim, delta_mu_q_proj, delta_var_q_proj);
    reshape_heads_to_proj(
        attn_delta_states.delta_mu_k, attn_delta_states.delta_var_k, batch_size,
        num_heads, this->seq_len, head_dim, delta_mu_k_proj, delta_var_k_proj);
    reshape_heads_to_proj(
        attn_delta_states.delta_mu_v, attn_delta_states.delta_var_v, batch_size,
        num_heads, this->seq_len, head_dim, delta_mu_v_proj, delta_var_v_proj);

    if (state_udapte) {
        // Compute delta_z for each projection and sum them
        int input_size = this->embed_dim;
        int out_size = batch_seq * input_size;

        std::vector<float> delta_mu_z_q(out_size, 0.0f);
        std::vector<float> delta_var_z_q(out_size, 0.0f);
        std::vector<float> delta_mu_z_k(out_size, 0.0f);
        std::vector<float> delta_var_z_k(out_size, 0.0f);
        std::vector<float> delta_mu_z_v(out_size, 0.0f);
        std::vector<float> delta_var_z_v(out_size, 0.0f);

        linear_bwd_fc_delta_z_mp(mu_w_q, this->bwd_states->jcb, delta_mu_q_proj,
                                 delta_var_q_proj, input_size, q_output_size,
                                 batch_seq, this->num_threads, delta_mu_z_q,
                                 delta_var_z_q);

        linear_bwd_fc_delta_z_mp(mu_w_k, this->bwd_states->jcb, delta_mu_k_proj,
                                 delta_var_k_proj, input_size, k_output_size,
                                 batch_seq, this->num_threads, delta_mu_z_k,
                                 delta_var_z_k);

        linear_bwd_fc_delta_z_mp(mu_w_v, this->bwd_states->jcb, delta_mu_v_proj,
                                 delta_var_v_proj, input_size, v_output_size,
                                 batch_seq, this->num_threads, delta_mu_z_v,
                                 delta_var_z_v);

        for (int i = 0; i < out_size; i++) {
            output_delta_states.delta_mu[i] =
                delta_mu_z_q[i] + delta_mu_z_k[i] + delta_mu_z_v[i];
            output_delta_states.delta_var[i] =
                delta_var_z_q[i] + delta_var_z_k[i] + delta_var_z_v[i];
        }
    }

    if (this->param_update) {
        linear_bwd_fc_delta_w_mp(
            var_w_q, this->bwd_states->mu_a, delta_mu_q_proj, delta_var_q_proj,
            this->embed_dim, q_output_size, batch_seq, this->num_threads,
            delta_mu_w_q, delta_var_w_q);

        linear_bwd_fc_delta_w_mp(
            var_w_k, this->bwd_states->mu_a, delta_mu_k_proj, delta_var_k_proj,
            this->embed_dim, k_output_size, batch_seq, this->num_threads,
            delta_mu_w_k, delta_var_w_k);

        linear_bwd_fc_delta_w_mp(
            var_w_v, this->bwd_states->mu_a, delta_mu_v_proj, delta_var_v_proj,
            this->embed_dim, v_output_size, batch_seq, this->num_threads,
            delta_mu_w_v, delta_var_w_v);

        if (this->bias) {
            linear_bwd_fc_delta_b_mp(
                var_b_q, delta_mu_q_proj, delta_var_q_proj, q_output_size,
                batch_seq, this->num_threads, delta_mu_b_q, delta_var_b_q);
            linear_bwd_fc_delta_b_mp(
                var_b_k, delta_mu_k_proj, delta_var_k_proj, k_output_size,
                batch_seq, this->num_threads, delta_mu_b_k, delta_var_b_k);
            linear_bwd_fc_delta_b_mp(
                var_b_v, delta_mu_v_proj, delta_var_v_proj, v_output_size,
                batch_seq, this->num_threads, delta_mu_b_v, delta_var_b_v);
        }
    }
}

AttentionScores MultiheadAttentionV2::get_attention_scores() {
    AttentionScores s;
    s.num_heads = (int)this->num_heads;
    s.timestep = (int)this->seq_len;
    int per_batch = s.num_heads * s.timestep * s.timestep;
    if (per_batch <= 0) return s;
    s.batch_size = (int)(attn_states.mu_att_score.size() / per_batch);
    s.mu = attn_states.mu_att_score;
    s.var = attn_states.var_att_score;
    return s;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> MultiheadAttentionV2::to_cuda(int device_idx) {
    this->device = "cuda";
    this->device_idx = device_idx;
    auto cuda_layer = std::make_unique<MultiheadAttentionV2Cuda>(
        this->embed_dim, this->num_heads, this->num_kv_heads, this->seq_len,
        this->bias, this->gain_w, this->gain_b, this->init_method,
        this->pos_emb, this->rope_theta, this->max_seq_len,
        this->use_causal_mask, device_idx);
    cuda_layer->copy_v2_params_from(*this);
    return cuda_layer;
}
#endif
