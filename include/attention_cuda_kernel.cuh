#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Forward Pass
// ---------------------------------------------------------------------------

// in_proj [batch_size*timestep, 3*num_heads*head_dim]  ->
// q,k,v each [batch_size, num_heads, timestep, head_dim]
__global__ void separate_in_proj_components_kernel(
    const float *mu_embs, const float *var_embs, int batch_size, int num_heads,
    int timestep, int head_dim, float *mu_q, float *var_q, float *mu_k,
    float *var_k, float *mu_v, float *var_v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * head_dim;
    if (idx >= total) return;
    int dim_idx = idx % head_dim;
    int timestep_idx = (idx / head_dim) % timestep;
    int head_idx = (idx / (timestep * head_dim)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * head_dim);

    int emb_size = num_heads * head_dim;
    int row_size = 3 * emb_size;
    int token_idx = batch_idx * timestep + timestep_idx;
    int head_off = head_idx * head_dim + dim_idx;
    int q_emb_idx = token_idx * row_size + head_off;
    int k_emb_idx = q_emb_idx + emb_size;
    int v_emb_idx = q_emb_idx + 2 * emb_size;

    mu_q[idx] = mu_embs[q_emb_idx];
    var_q[idx] = var_embs[q_emb_idx];
    mu_k[idx] = mu_embs[k_emb_idx];
    var_k[idx] = var_embs[k_emb_idx];
    mu_v[idx] = mu_embs[v_emb_idx];
    var_v[idx] = var_embs[v_emb_idx];
}

// Inverse: q,k,v -> in_proj.
__global__ void cat_in_proj_components_kernel(
    const float *mu_q, const float *var_q, const float *mu_k,
    const float *var_k, const float *mu_v, const float *var_v, int batch_size,
    int num_heads, int timestep, int head_dim, float *mu_embs,
    float *var_embs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * head_dim;
    if (idx >= total) return;
    int dim_idx = idx % head_dim;
    int timestep_idx = (idx / head_dim) % timestep;
    int head_idx = (idx / (timestep * head_dim)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * head_dim);

    int emb_size = num_heads * head_dim;
    int row_size = 3 * emb_size;
    int token_idx = batch_idx * timestep + timestep_idx;
    int head_off = head_idx * head_dim + dim_idx;
    int q_emb_idx = token_idx * row_size + head_off;
    int k_emb_idx = q_emb_idx + emb_size;
    int v_emb_idx = q_emb_idx + 2 * emb_size;

    mu_embs[q_emb_idx] = mu_q[idx];
    var_embs[q_emb_idx] = var_q[idx];
    mu_embs[k_emb_idx] = mu_k[idx];
    var_embs[k_emb_idx] = var_k[idx];
    mu_embs[v_emb_idx] = mu_v[idx];
    var_embs[v_emb_idx] = var_v[idx];
}

// QK^T with TAGI variance, scaled by 1/sqrt(head_dim).
// qk shape [batch_size, num_heads, timestep, timestep].
// TODO: optimize using shared memory
__global__ void query_key_kernel(const float *mu_q, const float *var_q,
                                 const float *mu_k, const float *var_k,
                                 int batch_size, int num_heads, int timestep,
                                 int head_dim, float *mu_qk, float *var_qk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * timestep;
    if (idx >= total) return;
    int key_timestep_idx = idx % timestep;
    int query_timestep_idx = (idx / timestep) % timestep;
    int head_idx = (idx / (timestep * timestep)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * timestep);

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int base = batch_idx * num_heads * timestep * head_dim +
               head_idx * timestep * head_dim;
    int q_off = base + query_timestep_idx * head_dim;
    int k_off = base + key_timestep_idx * head_dim;
    for (int dim_idx = 0; dim_idx < head_dim; dim_idx++) {
        float mq = mu_q[q_off + dim_idx];
        float vq = var_q[q_off + dim_idx];
        float mk = mu_k[k_off + dim_idx];
        float vk = var_k[k_off + dim_idx];
        sum_mu += mq * mk;
        sum_var += vq * vk + vq * mk * mk + vk * mq * mq;
    }
    float scale = rsqrtf((float)head_dim);
    mu_qk[idx] = sum_mu * scale;
    var_qk[idx] = sum_var * scale * scale;
}

// In-place causal mask before remax: positions with key > query get sentinels.
__global__ void apply_causal_mask_pre_remax_kernel(float *mu, float *var,
                                                   int batch_size,
                                                   int num_heads,
                                                   int timestep) {
    constexpr float MASK_MU = -1e8f;
    constexpr float MASK_VAR = 1e-6f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * timestep;
    if (idx >= total) return;
    int key_timestep_idx = idx % timestep;
    int query_timestep_idx = (idx / timestep) % timestep;
    if (key_timestep_idx > query_timestep_idx) {
        mu[idx] = MASK_MU;
        var[idx] = MASK_VAR;
    }
}

// In-place causal mask after remax: zero future positions in att/var/jcb.
__global__ void apply_causal_mask_post_remax_kernel(float *mu_att,
                                                    float *var_att, float *jcb,
                                                    int batch_size,
                                                    int num_heads,
                                                    int timestep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * timestep;
    if (idx >= total) return;
    int key_timestep_idx = idx % timestep;
    int query_timestep_idx = (idx / timestep) % timestep;
    if (key_timestep_idx > query_timestep_idx) {
        mu_att[idx] = 0.0f;
        var_att[idx] = 0.0f;
        jcb[idx] = 0.0f;
    }
}

// SV[b,h,q,d] = sum_k att[b,h,q,k] * V[b,h,k,d]  (TAGI variance).
// att in [batch_size,num_heads,timestep,timestep],
// v   in [batch_size,num_heads,timestep,head_dim],
// out in [batch_size,num_heads,timestep,head_dim].
__global__ void att_score_value_kernel(const float *mu_att,
                                       const float *var_att, const float *mu_v,
                                       const float *var_v, int batch_size,
                                       int num_heads, int timestep,
                                       int head_dim, float *mu_sv,
                                       float *var_sv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * head_dim;
    if (idx >= total) return;
    int dim_idx = idx % head_dim;
    int query_timestep_idx = (idx / head_dim) % timestep;
    int head_idx = (idx / (timestep * head_dim)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * head_dim);

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int a_base = batch_idx * num_heads * timestep * timestep +
                 head_idx * timestep * timestep + query_timestep_idx * timestep;
    int v_base = batch_idx * num_heads * timestep * head_dim +
                 head_idx * timestep * head_dim;
    for (int key_timestep_idx = 0; key_timestep_idx < timestep;
         key_timestep_idx++) {
        float ma = mu_att[a_base + key_timestep_idx];
        float va = var_att[a_base + key_timestep_idx];
        float mb = mu_v[v_base + key_timestep_idx * head_dim + dim_idx];
        float vb = var_v[v_base + key_timestep_idx * head_dim + dim_idx];
        sum_mu += ma * mb;
        sum_var += va * vb + va * mb * mb + vb * ma * ma;
    }
    mu_sv[idx] = sum_mu;
    var_sv[idx] = sum_var;
}

// [batch_size, num_heads, timestep, head_dim] ->
// [batch_size, timestep, num_heads, head_dim] (forward output projection).
__global__ void project_output_forward_kernel(const float *mu_in,
                                              const float *var_in,
                                              int batch_size, int num_heads,
                                              int timestep, int head_dim,
                                              float *mu_out, float *var_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * timestep * num_heads * head_dim;
    if (idx >= total) return;
    int dim_idx = idx % head_dim;
    int head_idx = (idx / head_dim) % num_heads;
    int timestep_idx = (idx / (num_heads * head_dim)) % timestep;
    int batch_idx = idx / (timestep * num_heads * head_dim);

    int in_idx = batch_idx * num_heads * timestep * head_dim +
                 head_idx * timestep * head_dim + timestep_idx * head_dim +
                 dim_idx;
    mu_out[idx] = mu_in[in_idx];
    var_out[idx] = var_in[in_idx];
}

// Inverse: [batch_size, timestep, num_heads, head_dim] ->
// [batch_size, num_heads, timestep, head_dim] (used at start of backward).
__global__ void project_output_backward_kernel(const float *mu_in,
                                               const float *var_in,
                                               int batch_size, int num_heads,
                                               int timestep, int head_dim,
                                               float *mu_out, float *var_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * head_dim;
    if (idx >= total) return;
    int dim_idx = idx % head_dim;
    int timestep_idx = (idx / head_dim) % timestep;
    int head_idx = (idx / (timestep * head_dim)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * head_dim);

    int in_idx = batch_idx * timestep * num_heads * head_dim +
                 timestep_idx * num_heads * head_dim + head_idx * head_dim +
                 dim_idx;
    mu_out[idx] = mu_in[in_idx];
    var_out[idx] = var_in[in_idx];
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------
__global__ void mha_delta_score_kernel(const float *mu_v, const float *delta_mu,
                                       const float *delta_var, int batch_size,
                                       int num_heads, int timestep,
                                       int head_dim, float *delta_mu_s,
                                       float *delta_var_s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * timestep;
    if (idx >= total) return;
    int key_timestep_idx = idx % timestep;
    int query_timestep_idx = (idx / timestep) % timestep;
    int head_idx = (idx / (timestep * timestep)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * timestep);

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int base = batch_idx * num_heads * timestep * head_dim +
               head_idx * timestep * head_dim;
    int v_off = base + key_timestep_idx * head_dim;
    int d_off = base + query_timestep_idx * head_dim;
    for (int dim_idx = 0; dim_idx < head_dim; dim_idx++) {
        float mv = mu_v[v_off + dim_idx];
        sum_mu += mv * delta_mu[d_off + dim_idx];
        sum_var += mv * delta_var[d_off + dim_idx] * mv;
    }
    delta_mu_s[idx] = sum_mu;
    delta_var_s[idx] = sum_var;
}

// Remax cross-covariance correction: d[q,k] -= sum_l a[q,l] * d[q,l] per row.
__global__ void center_delta_score_kernel(const float *mu_att,
                                          float *delta_mu_s, int num_rows,
                                          int timestep) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    int base = row * timestep;
    float sum = 0.0f;
    for (int k = 0; k < timestep; k++) {
        sum += mu_att[base + k] * delta_mu_s[base + k];
    }
    for (int k = 0; k < timestep; k++) {
        delta_mu_s[base + k] -= sum;
    }
}

__global__ void mha_delta_value_kernel(const float *mu_s, const float *delta_mu,
                                       const float *delta_var, int batch_size,
                                       int num_heads, int timestep,
                                       int head_dim, float *delta_mu_v,
                                       float *delta_var_v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * head_dim;
    if (idx >= total) return;
    int dim_idx = idx % head_dim;
    int value_timestep_idx = (idx / head_dim) % timestep;
    int head_idx = (idx / (timestep * head_dim)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * head_dim);

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int s_base = batch_idx * num_heads * timestep * timestep +
                 head_idx * timestep * timestep;
    int v_base = batch_idx * num_heads * timestep * head_dim +
                 head_idx * timestep * head_dim;
    for (int query_timestep_idx = 0; query_timestep_idx < timestep;
         query_timestep_idx++) {
        float ms =
            mu_s[s_base + query_timestep_idx * timestep + value_timestep_idx];
        sum_mu +=
            ms * delta_mu[v_base + query_timestep_idx * head_dim + dim_idx];
        sum_var += ms *
                   delta_var[v_base + query_timestep_idx * head_dim + dim_idx] *
                   ms;
    }
    delta_mu_v[idx] = sum_mu;
    delta_var_v[idx] = sum_var;
}

__global__ void mha_delta_query_kernel(
    const float *mu_k, const float *delta_mu_s, const float *delta_var_s,
    const float *jcb_mqk, int batch_size, int num_heads, int timestep,
    int head_dim, float *delta_mu_q, float *delta_var_q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * head_dim;
    if (idx >= total) return;
    int dim_idx = idx % head_dim;
    int query_timestep_idx = (idx / head_dim) % timestep;
    int head_idx = (idx / (timestep * head_dim)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * head_dim);

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int k_base = batch_idx * num_heads * timestep * head_dim +
                 head_idx * timestep * head_dim;
    int s_base = batch_idx * num_heads * timestep * timestep +
                 head_idx * timestep * timestep + query_timestep_idx * timestep;
    for (int key_timestep_idx = 0; key_timestep_idx < timestep;
         key_timestep_idx++) {
        float mk = mu_k[k_base + key_timestep_idx * head_dim + dim_idx];
        float jc = jcb_mqk[s_base + key_timestep_idx];
        sum_mu += mk * delta_mu_s[s_base + key_timestep_idx] * jc;
        sum_var += mk * mk * delta_var_s[s_base + key_timestep_idx] * jc * jc;
    }
    float scale = rsqrtf((float)head_dim);
    delta_mu_q[idx] = sum_mu * scale;
    delta_var_q[idx] = sum_var * scale * scale;
}

__global__ void mha_delta_key_kernel(const float *mu_q, const float *delta_mu_s,
                                     const float *delta_var_s,
                                     const float *jcb_mqk, int batch_size,
                                     int num_heads, int timestep, int head_dim,
                                     float *delta_mu_k, float *delta_var_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * head_dim;
    if (idx >= total) return;
    int dim_idx = idx % head_dim;
    int key_timestep_idx = (idx / head_dim) % timestep;
    int head_idx = (idx / (timestep * head_dim)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * head_dim);

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int q_base = batch_idx * num_heads * timestep * head_dim +
                 head_idx * timestep * head_dim;
    int s_base = batch_idx * num_heads * timestep * timestep +
                 head_idx * timestep * timestep;
    for (int query_timestep_idx = 0; query_timestep_idx < timestep;
         query_timestep_idx++) {
        float mq = mu_q[q_base + query_timestep_idx * head_dim + dim_idx];
        int s_idx = s_base + query_timestep_idx * timestep + key_timestep_idx;
        float jc = jcb_mqk[s_idx];
        sum_mu += mq * delta_mu_s[s_idx] * jc;
        sum_var += mq * mq * delta_var_s[s_idx] * jc * jc;
    }
    float scale = rsqrtf((float)head_dim);
    delta_mu_k[idx] = sum_mu * scale;
    delta_var_k[idx] = sum_var * scale * scale;
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

__global__ void apply_rope_kernel(const float *mu_in, const float *var_in,
                                  const float *cos_cache,
                                  const float *sin_cache, int batch_size,
                                  int num_heads, int timestep, int head_dim,
                                  float *mu_out, float *var_out) {
    int half = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * half;
    if (idx >= total) return;
    int dim_pair_idx = idx % half;
    int timestep_idx = (idx / half) % timestep;
    int head_idx = (idx / (timestep * half)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * half);

    int in_idx = batch_idx * num_heads * timestep * head_dim +
                 head_idx * timestep * head_dim + timestep_idx * head_dim +
                 2 * dim_pair_idx;
    int cache_idx = timestep_idx * half + dim_pair_idx;

    float mu_x1 = mu_in[in_idx];
    float mu_x2 = mu_in[in_idx + 1];
    float var_x1 = var_in[in_idx];
    float var_x2 = var_in[in_idx + 1];
    float cv = cos_cache[cache_idx];
    float sv = sin_cache[cache_idx];

    mu_out[in_idx] = mu_x1 * cv - mu_x2 * sv;
    mu_out[in_idx + 1] = mu_x1 * sv + mu_x2 * cv;
    var_out[in_idx] = var_x1 * cv * cv + var_x2 * sv * sv;
    var_out[in_idx + 1] = var_x1 * sv * sv + var_x2 * cv * cv;
}

__global__ void rope_backward_kernel(
    const float *delta_mu_in, const float *delta_var_in, const float *cos_cache,
    const float *sin_cache, int batch_size, int num_heads, int timestep,
    int head_dim, float *delta_mu_out, float *delta_var_out) {
    int half = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * half;
    if (idx >= total) return;
    int dim_pair_idx = idx % half;
    int timestep_idx = (idx / half) % timestep;
    int head_idx = (idx / (timestep * half)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * half);

    int in_idx = batch_idx * num_heads * timestep * head_dim +
                 head_idx * timestep * head_dim + timestep_idx * head_dim +
                 2 * dim_pair_idx;
    int cache_idx = timestep_idx * half + dim_pair_idx;

    float dmu_y1 = delta_mu_in[in_idx];
    float dmu_y2 = delta_mu_in[in_idx + 1];
    float dvar_y1 = delta_var_in[in_idx];
    float dvar_y2 = delta_var_in[in_idx + 1];
    float cv = cos_cache[cache_idx];
    float sv = sin_cache[cache_idx];

    delta_mu_out[in_idx] = dmu_y1 * cv + dmu_y2 * sv;
    delta_mu_out[in_idx + 1] = -dmu_y1 * sv + dmu_y2 * cv;
    delta_var_out[in_idx] = dvar_y1 * cv * cv + dvar_y2 * sv * sv;
    delta_var_out[in_idx + 1] = dvar_y1 * sv * sv + dvar_y2 * cv * cv;
}

// [batch_size*timestep, num_heads*head_dim] ->
// [batch_size, num_heads, timestep, head_dim].
__global__ void reshape_proj_to_heads_kernel(const float *mu_proj,
                                             const float *var_proj,
                                             int batch_size, int num_heads,
                                             int timestep, int head_dim,
                                             float *mu_out, float *var_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * head_dim;
    if (idx >= total) return;
    int dim_idx = idx % head_dim;
    int timestep_idx = (idx / head_dim) % timestep;
    int head_idx = (idx / (timestep * head_dim)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * head_dim);

    int emb_size = num_heads * head_dim;
    int token_idx = batch_idx * timestep + timestep_idx;
    int proj_idx = token_idx * emb_size + head_idx * head_dim + dim_idx;
    mu_out[idx] = mu_proj[proj_idx];
    var_out[idx] = var_proj[proj_idx];
}

// Inverse of reshape_proj_to_heads_kernel.
__global__ void reshape_heads_to_proj_kernel(const float *mu_heads,
                                             const float *var_heads,
                                             int batch_size, int num_heads,
                                             int timestep, int head_dim,
                                             float *mu_out, float *var_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * timestep * head_dim;
    if (idx >= total) return;
    int dim_idx = idx % head_dim;
    int timestep_idx = (idx / head_dim) % timestep;
    int head_idx = (idx / (timestep * head_dim)) % num_heads;
    int batch_idx = idx / (num_heads * timestep * head_dim);

    int emb_size = num_heads * head_dim;
    int token_idx = batch_idx * timestep + timestep_idx;
    int proj_idx = token_idx * emb_size + head_idx * head_dim + dim_idx;
    mu_out[proj_idx] = mu_heads[idx];
    var_out[proj_idx] = var_heads[idx];
}

// dst += src (both mu and var) used by V2 to sum delta_z from Q,K,V.
__global__ void add_delta_inplace_kernel(const float *src_mu,
                                         const float *src_var, int num_elements,
                                         float *dst_mu, float *dst_var) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    dst_mu[idx] += src_mu[idx];
    dst_var[idx] += src_var[idx];
}
