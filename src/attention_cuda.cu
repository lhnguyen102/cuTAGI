#include <vector>

#include "../include/attention_cuda.cuh"
#include "../include/attention_cuda_kernel.cuh"
#include "../include/cuda_error_checking.cuh"
#include "../include/linear_layer_cuda.cuh"
#include "../include/param_init.h"

namespace {
constexpr int THREADS = 256;

inline int blocks_for(size_t total) { return (total + THREADS - 1) / THREADS; }

inline void alloc_pair(float **d_mu, float **d_var, size_t n) {
    CHECK_CUDA_ERROR(cudaMalloc((void **)d_mu, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)d_var, n * sizeof(float)));
}

inline void free_buffer(float **d_buf) {
    cudaFree(*d_buf);
    *d_buf = nullptr;
}

inline void free_pair(float **d_mu, float **d_var) {
    free_buffer(d_mu);
    free_buffer(d_var);
}

inline void device_to_host(std::vector<float> &dst, const float *src,
                           size_t n) {
    dst.resize(n);
    cudaMemcpy(dst.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost);
}
}  // namespace

////////////////////////////////////////////////////////////////////////////////
// AttentionStatesCuda / AttentionDeltaStatesCuda
////////////////////////////////////////////////////////////////////////////////
AttentionStatesCuda::~AttentionStatesCuda() { this->deallocate(); }

void AttentionStatesCuda::set_size(int batch_size, int num_heads, int timestep,
                                   int head_dim, bool use_rope) {
    if (batch_size == allocated_batch_size &&
        num_heads == allocated_num_heads && timestep == allocated_timestep &&
        head_dim == allocated_head_dim && use_rope == allocated_rope) {
        return;
    }
    this->deallocate();

    size_t comp_size = (size_t)batch_size * num_heads * timestep * head_dim;
    size_t qk_size = (size_t)batch_size * num_heads * timestep * timestep;
    alloc_pair(&d_mu_q, &d_var_q, comp_size);
    alloc_pair(&d_mu_k, &d_var_k, comp_size);
    alloc_pair(&d_mu_v, &d_var_v, comp_size);
    alloc_pair(&d_mu_qk, &d_var_qk, qk_size);
    alloc_pair(&d_mu_att_score, &d_var_att_score, qk_size);
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_j_mqk, qk_size * sizeof(float)));
    alloc_pair(&d_mu_sv, &d_var_sv, comp_size);
    if (use_rope) {
        alloc_pair(&d_mu_q_rope, &d_var_q_rope, comp_size);
        alloc_pair(&d_mu_k_rope, &d_var_k_rope, comp_size);
    }

    allocated_batch_size = batch_size;
    allocated_num_heads = num_heads;
    allocated_timestep = timestep;
    allocated_head_dim = head_dim;
    allocated_rope = use_rope;
}

void AttentionStatesCuda::deallocate() {
    free_pair(&d_mu_q, &d_var_q);
    free_pair(&d_mu_k, &d_var_k);
    free_pair(&d_mu_v, &d_var_v);
    free_pair(&d_mu_q_rope, &d_var_q_rope);
    free_pair(&d_mu_k_rope, &d_var_k_rope);
    free_pair(&d_mu_qk, &d_var_qk);
    free_pair(&d_mu_att_score, &d_var_att_score);
    free_buffer(&d_j_mqk);
    free_pair(&d_mu_sv, &d_var_sv);
    allocated_batch_size = allocated_num_heads = allocated_timestep =
        allocated_head_dim = 0;
    allocated_rope = false;
}

AttentionDeltaStatesCuda::~AttentionDeltaStatesCuda() { this->deallocate(); }

void AttentionDeltaStatesCuda::set_size(int batch_size, int num_heads,
                                        int timestep, int head_dim,
                                        bool use_rope) {
    if (batch_size == allocated_batch_size &&
        num_heads == allocated_num_heads && timestep == allocated_timestep &&
        head_dim == allocated_head_dim && use_rope == allocated_rope) {
        return;
    }
    this->deallocate();

    size_t comp_size = (size_t)batch_size * num_heads * timestep * head_dim;
    size_t qk_size = (size_t)batch_size * num_heads * timestep * timestep;
    alloc_pair(&d_delta_mu_buffer, &d_delta_var_buffer, comp_size);
    alloc_pair(&d_delta_mu_v, &d_delta_var_v, comp_size);
    alloc_pair(&d_delta_mu_att_score, &d_delta_var_att_score, qk_size);
    alloc_pair(&d_delta_mu_q, &d_delta_var_q, comp_size);
    alloc_pair(&d_delta_mu_k, &d_delta_var_k, comp_size);
    if (use_rope) {
        alloc_pair(&d_delta_mu_q_rope, &d_delta_var_q_rope, comp_size);
        alloc_pair(&d_delta_mu_k_rope, &d_delta_var_k_rope, comp_size);
    }

    allocated_batch_size = batch_size;
    allocated_num_heads = num_heads;
    allocated_timestep = timestep;
    allocated_head_dim = head_dim;
    allocated_rope = use_rope;
}

void AttentionDeltaStatesCuda::deallocate() {
    free_pair(&d_delta_mu_buffer, &d_delta_var_buffer);
    free_pair(&d_delta_mu_v, &d_delta_var_v);
    free_pair(&d_delta_mu_att_score, &d_delta_var_att_score);
    free_pair(&d_delta_mu_q, &d_delta_var_q);
    free_pair(&d_delta_mu_k, &d_delta_var_k);
    free_pair(&d_delta_mu_q_rope, &d_delta_var_q_rope);
    free_pair(&d_delta_mu_k_rope, &d_delta_var_k_rope);
    allocated_batch_size = allocated_num_heads = allocated_timestep =
        allocated_head_dim = 0;
    allocated_rope = false;
}

////////////////////////////////////////////////////////////////////////////////
// MultiheadAttentionCuda
////////////////////////////////////////////////////////////////////////////////
MultiheadAttentionCuda::MultiheadAttentionCuda(
    size_t embed_dim, size_t num_heads, size_t num_kv_heads, size_t seq_len_,
    bool bias, float gain_w, float gain_b, std::string init_method,
    bool use_rope, float rope_theta, size_t max_seq_len, bool use_causal_mask,
    bool center_score_delta, int device_idx)
    : embed_dim(embed_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method),
      use_rope(use_rope),
      rope_theta(rope_theta),
      max_seq_len(max_seq_len),
      use_causal_mask(use_causal_mask),
      center_score_delta(center_score_delta) {
    check_attention_config(embed_dim, num_heads, num_kv_heads, seq_len_,
                           use_rope, max_seq_len);
    this->input_size = embed_dim;
    this->output_size = embed_dim;
    this->seq_len = seq_len_;
    this->head_dim = embed_dim / num_heads;
    this->bias = bias;
    this->device_idx = device_idx;
    this->num_weights = embed_dim * 3 * embed_dim;
    this->num_biases = this->bias ? 3 * embed_dim : 0;

    if (this->training) {
        this->allocate_param_delta();
    }

    remax_layer = std::make_unique<RemaxCuda>();

    if (this->use_rope) {
        std::vector<float> cos_cache, sin_cache;
        generate_rope_cache(this->max_seq_len, this->head_dim, this->rope_theta,
                            cos_cache, sin_cache);
        size_t bytes = cos_cache.size() * sizeof(float);
        cudaSetDevice(this->device_idx);
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cos_cache, bytes));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sin_cache, bytes));
        CHECK_CUDA_ERROR(cudaMemcpy(d_cos_cache, cos_cache.data(), bytes,
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_sin_cache, sin_cache.data(), bytes,
                                    cudaMemcpyHostToDevice));
    }
}

MultiheadAttentionCuda::~MultiheadAttentionCuda() {
    cudaFree(d_cos_cache);
    cudaFree(d_sin_cache);
}

std::string MultiheadAttentionCuda::get_layer_info() const {
    return "SelfAttention(heads=" + std::to_string(this->num_heads) +
           ", kv_heads=" + std::to_string(this->num_kv_heads) +
           ", emb_size=" + std::to_string(this->embed_dim) + ")";
}

std::string MultiheadAttentionCuda::get_layer_name() const {
    return "MultiheadAttentionCuda";
}

LayerType MultiheadAttentionCuda::get_layer_type() const {
    return LayerType::MultiheadAttention;
}

void MultiheadAttentionCuda::init_weight_bias()
/*Same prior as MultiheadAttention::init_weight_bias.
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_linear(this->init_method, this->gain_w, this->gain_b,
                                this->embed_dim, this->embed_dim,
                                this->num_weights, this->num_biases);
    this->allocate_param_memory();
    this->params_to_device();
}

void MultiheadAttentionCuda::forward(BaseHiddenStates &input_states,
                                     BaseHiddenStates &output_states,
                                     BaseTempStates &temp_states) {
    HiddenStateCuda *cu_in = dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_out = dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = cu_in->block_size;
    int timestep = this->seq_len;
    int num_heads = this->num_heads;
    int head_dim = this->head_dim;
    int batch_seq = batch_size * timestep;
    int qkv_size = 3 * this->embed_dim;
    size_t comp_size = (size_t)batch_seq * this->embed_dim;
    size_t qk_size = (size_t)batch_size * num_heads * timestep * timestep;
    size_t num_rows = (size_t)batch_size * num_heads * timestep;

    this->set_cap_factor_udapte(batch_size);
    attn_states.set_size(batch_size, num_heads, timestep, head_dim,
                         this->use_rope);

    in_proj_buffer.set_size((size_t)batch_seq * qkv_size, batch_seq);
    HiddenStateCuda *p_proj = &in_proj_buffer;
    linear_forward_cuda(cu_in, p_proj, this->d_mu_w, this->d_var_w,
                        this->d_mu_b, this->d_var_b, this->embed_dim, qkv_size,
                        batch_seq, this->bias);

    separate_in_proj_components_kernel<<<blocks_for(comp_size), THREADS>>>(
        in_proj_buffer.d_mu_a, in_proj_buffer.d_var_a, batch_size, num_heads,
        timestep, head_dim, attn_states.d_mu_q, attn_states.d_var_q,
        attn_states.d_mu_k, attn_states.d_var_k, attn_states.d_mu_v,
        attn_states.d_var_v);

    float *mu_q = attn_states.d_mu_q, *var_q = attn_states.d_var_q;
    float *mu_k = attn_states.d_mu_k, *var_k = attn_states.d_var_k;
    if (this->use_rope) {
        apply_rope_kernel<<<blocks_for(comp_size / 2), THREADS>>>(
            attn_states.d_mu_q, attn_states.d_var_q, d_cos_cache, d_sin_cache,
            batch_size, num_heads, timestep, head_dim, attn_states.d_mu_q_rope,
            attn_states.d_var_q_rope);
        apply_rope_kernel<<<blocks_for(comp_size / 2), THREADS>>>(
            attn_states.d_mu_k, attn_states.d_var_k, d_cos_cache, d_sin_cache,
            batch_size, num_heads, timestep, head_dim, attn_states.d_mu_k_rope,
            attn_states.d_var_k_rope);
        mu_q = attn_states.d_mu_q_rope;
        var_q = attn_states.d_var_q_rope;
        mu_k = attn_states.d_mu_k_rope;
        var_k = attn_states.d_var_k_rope;
    }

    query_key_kernel<<<blocks_for(qk_size), THREADS>>>(
        mu_q, var_q, mu_k, var_k, batch_size, num_heads, timestep, head_dim,
        attn_states.d_mu_qk, attn_states.d_var_qk);

    remax_input.set_size(qk_size, num_rows);
    remax_input.seq_len = 1;
    cudaMemcpy(remax_input.d_mu_a, attn_states.d_mu_qk, qk_size * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(remax_input.d_var_a, attn_states.d_var_qk,
               qk_size * sizeof(float), cudaMemcpyDeviceToDevice);
    if (this->use_causal_mask) {
        apply_causal_mask_pre_remax_kernel<<<blocks_for(qk_size), THREADS>>>(
            remax_input.d_mu_a, remax_input.d_var_a, batch_size, num_heads,
            timestep);
    }

    remax_output.set_size(qk_size, num_rows);
    remax_layer->forward(remax_input, remax_output, remax_temp);
    if (this->use_causal_mask) {
        apply_causal_mask_post_remax_kernel<<<blocks_for(qk_size), THREADS>>>(
            remax_output.d_mu_a, remax_output.d_var_a, remax_output.d_jcb,
            batch_size, num_heads, timestep);
    }
    cudaMemcpy(attn_states.d_mu_att_score, remax_output.d_mu_a,
               qk_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(attn_states.d_var_att_score, remax_output.d_var_a,
               qk_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(attn_states.d_j_mqk, remax_output.d_jcb, qk_size * sizeof(float),
               cudaMemcpyDeviceToDevice);

    att_score_value_kernel<<<blocks_for(comp_size), THREADS>>>(
        attn_states.d_mu_att_score, attn_states.d_var_att_score,
        attn_states.d_mu_v, attn_states.d_var_v, batch_size, num_heads,
        timestep, head_dim, attn_states.d_mu_sv, attn_states.d_var_sv);

    project_output_forward_kernel<<<blocks_for(comp_size), THREADS>>>(
        attn_states.d_mu_sv, attn_states.d_var_sv, batch_size, num_heads,
        timestep, head_dim, cu_out->d_mu_a, cu_out->d_var_a);
    CHECK_LAST_CUDA_ERROR();

    cu_out->width = this->out_width;
    cu_out->height = this->out_height;
    cu_out->depth = this->out_channels;
    cu_out->block_size = batch_size;
    cu_out->seq_len = this->seq_len;
    cu_out->actual_size = this->output_size;

    if (this->training) {
        this->store_states_for_training_cuda(*cu_in, *cu_out);
    }
}

void MultiheadAttentionCuda::backward(BaseDeltaStates &input_delta_states,
                                      BaseDeltaStates &output_delta_states,
                                      BaseTempStates &temp_states,
                                      bool state_udapte) {
    DeltaStateCuda *cu_in_delta =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_out_delta =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    int batch_size = cu_in_delta->block_size;
    int timestep = this->seq_len;
    int num_heads = this->num_heads;
    int head_dim = this->head_dim;
    int batch_seq = batch_size * timestep;
    int qkv_size = 3 * this->embed_dim;
    size_t comp_size = (size_t)batch_seq * this->embed_dim;
    size_t qk_size = (size_t)batch_size * num_heads * timestep * timestep;

    attn_delta_states.set_size(batch_size, num_heads, timestep, head_dim,
                               this->use_rope);

    project_output_backward_kernel<<<blocks_for(comp_size), THREADS>>>(
        cu_in_delta->d_delta_mu, cu_in_delta->d_delta_var, batch_size,
        num_heads, timestep, head_dim, attn_delta_states.d_delta_mu_buffer,
        attn_delta_states.d_delta_var_buffer);

    mha_delta_value_kernel<<<blocks_for(comp_size), THREADS>>>(
        attn_states.d_mu_att_score, attn_delta_states.d_delta_mu_buffer,
        attn_delta_states.d_delta_var_buffer, batch_size, num_heads, timestep,
        head_dim, attn_delta_states.d_delta_mu_v,
        attn_delta_states.d_delta_var_v);

    mha_delta_score_kernel<<<blocks_for(qk_size), THREADS>>>(
        attn_states.d_mu_v, attn_delta_states.d_delta_mu_buffer,
        attn_delta_states.d_delta_var_buffer, batch_size, num_heads, timestep,
        head_dim, attn_delta_states.d_delta_mu_att_score,
        attn_delta_states.d_delta_var_att_score);

    if (this->center_score_delta) {
        int num_rows = batch_size * num_heads * timestep;
        center_delta_score_kernel<<<blocks_for(num_rows), THREADS>>>(
            attn_states.d_mu_att_score, num_rows, timestep,
            attn_delta_states.d_delta_mu_att_score);
    }

    float *mu_q = attn_states.d_mu_q, *mu_k = attn_states.d_mu_k;
    float *delta_mu_q = attn_delta_states.d_delta_mu_q;
    float *delta_var_q = attn_delta_states.d_delta_var_q;
    float *delta_mu_k = attn_delta_states.d_delta_mu_k;
    float *delta_var_k = attn_delta_states.d_delta_var_k;
    if (this->use_rope) {
        mu_q = attn_states.d_mu_q_rope;
        mu_k = attn_states.d_mu_k_rope;
        delta_mu_q = attn_delta_states.d_delta_mu_q_rope;
        delta_var_q = attn_delta_states.d_delta_var_q_rope;
        delta_mu_k = attn_delta_states.d_delta_mu_k_rope;
        delta_var_k = attn_delta_states.d_delta_var_k_rope;
    }
    mha_delta_query_kernel<<<blocks_for(comp_size), THREADS>>>(
        mu_k, attn_delta_states.d_delta_mu_att_score,
        attn_delta_states.d_delta_var_att_score, attn_states.d_j_mqk,
        batch_size, num_heads, timestep, head_dim, delta_mu_q, delta_var_q);
    mha_delta_key_kernel<<<blocks_for(comp_size), THREADS>>>(
        mu_q, attn_delta_states.d_delta_mu_att_score,
        attn_delta_states.d_delta_var_att_score, attn_states.d_j_mqk,
        batch_size, num_heads, timestep, head_dim, delta_mu_k, delta_var_k);
    if (this->use_rope) {
        rope_backward_kernel<<<blocks_for(comp_size / 2), THREADS>>>(
            delta_mu_q, delta_var_q, d_cos_cache, d_sin_cache, batch_size,
            num_heads, timestep, head_dim, attn_delta_states.d_delta_mu_q,
            attn_delta_states.d_delta_var_q);
        rope_backward_kernel<<<blocks_for(comp_size / 2), THREADS>>>(
            delta_mu_k, delta_var_k, d_cos_cache, d_sin_cache, batch_size,
            num_heads, timestep, head_dim, attn_delta_states.d_delta_mu_k,
            attn_delta_states.d_delta_var_k);
    }

    d_in_proj_buffer.set_size((size_t)batch_seq * qkv_size, batch_seq);
    cat_in_proj_components_kernel<<<blocks_for(comp_size), THREADS>>>(
        attn_delta_states.d_delta_mu_q, attn_delta_states.d_delta_var_q,
        attn_delta_states.d_delta_mu_k, attn_delta_states.d_delta_var_k,
        attn_delta_states.d_delta_mu_v, attn_delta_states.d_delta_var_v,
        batch_size, num_heads, timestep, head_dim, d_in_proj_buffer.d_delta_mu,
        d_in_proj_buffer.d_delta_var);

    BackwardStateCuda *cu_bwd =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());
    DeltaStateCuda *p_in_dproj = &d_in_proj_buffer;
    if (state_udapte) {
        linear_state_backward_cuda(p_in_dproj, cu_out_delta, cu_bwd,
                                   this->d_mu_w, this->embed_dim, qkv_size,
                                   batch_seq);
        cu_out_delta->seq_len = this->seq_len;
    }

    if (this->param_update) {
        linear_weight_backward_cuda(
            p_in_dproj, cu_out_delta, cu_bwd, this->d_var_w, this->embed_dim,
            qkv_size, batch_seq, this->d_delta_mu_w, this->d_delta_var_w);
        if (this->bias) {
            unsigned int blocks_b = (qkv_size + 16 - 1) / 16;
            linear_bwd_delta_b<<<dim3(1, blocks_b), dim3(16, 16)>>>(
                this->d_var_b, d_in_proj_buffer.d_delta_mu,
                d_in_proj_buffer.d_delta_var, this->embed_dim, qkv_size,
                batch_seq, this->d_delta_mu_b, this->d_delta_var_b);
        }
    }
    CHECK_LAST_CUDA_ERROR();
}

std::unique_ptr<BaseLayer> MultiheadAttentionCuda::to_host() {
    this->params_to_host();
    auto host = std::make_unique<MultiheadAttention>(
        this->embed_dim, this->num_heads, this->num_kv_heads, this->seq_len,
        this->bias, this->gain_w, this->gain_b, this->init_method,
        this->use_rope, this->rope_theta, this->max_seq_len,
        this->use_causal_mask, this->center_score_delta, this->device_idx);
    host->mu_w = this->mu_w;
    host->var_w = this->var_w;
    host->mu_b = this->mu_b;
    host->var_b = this->var_b;
    return host;
}

AttentionScores MultiheadAttentionCuda::get_attention_scores() const {
    AttentionScores scores;
    scores.batch_size = attn_states.allocated_batch_size;
    scores.num_heads = attn_states.allocated_num_heads;
    scores.timestep = attn_states.allocated_timestep;
    size_t qk_size = (size_t)scores.batch_size * scores.num_heads *
                     scores.timestep * scores.timestep;
    device_to_host(scores.mu_qk, attn_states.d_mu_qk, qk_size);
    device_to_host(scores.var_qk, attn_states.d_var_qk, qk_size);
    device_to_host(scores.mu_att, attn_states.d_mu_att_score, qk_size);
    device_to_host(scores.var_att, attn_states.d_var_att_score, qk_size);
    return scores;
}
