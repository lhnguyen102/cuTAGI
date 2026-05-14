#include <cmath>
#include <cstdio>
#include <vector>

#include "../include/attention_cuda.cuh"
#include "../include/attention_cuda_kernel.cuh"
#include "../include/cuda_error_checking.cuh"
#include "../include/custom_logger.h"
#include "../include/linear_layer_cuda.cuh"
#include "../include/param_init.h"

////////////////////////////////////////////////////////////////////////////////
// AttentionStatesCuda / AttentionDeltaStatesCuda
////////////////////////////////////////////////////////////////////////////////
namespace {
inline void realloc_pair(float **d_mu, float **d_var, size_t n) {
    if (*d_mu) cudaFree(*d_mu);
    if (*d_var) cudaFree(*d_var);
    CHECK_CUDA_ERROR(cudaMalloc((void **)d_mu, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)d_var, n * sizeof(float)));
}

inline void realloc_single(float **d_buf, size_t n) {
    if (*d_buf) cudaFree(*d_buf);
    CHECK_CUDA_ERROR(cudaMalloc((void **)d_buf, n * sizeof(float)));
}

inline void free_pair(float **d_mu, float **d_var) {
    if (*d_mu) {
        cudaFree(*d_mu);
        *d_mu = nullptr;
    }
    if (*d_var) {
        cudaFree(*d_var);
        *d_var = nullptr;
    }
}

inline void free_single(float **d_buf) {
    if (*d_buf) {
        cudaFree(*d_buf);
        *d_buf = nullptr;
    }
}
}  // namespace

AttentionStatesCuda::~AttentionStatesCuda() { this->deallocate(); }

void AttentionStatesCuda::set_size(int batch_size, int num_heads, int timestep,
                                   int head_dim, bool need_pe) {
    bool same_shape =
        (batch_size == allocated_batch_size &&
         num_heads == allocated_num_heads && timestep == allocated_timestep &&
         head_dim == allocated_head_dim);
    bool same_pe = (alloc_pe == need_pe);
    if (same_shape && same_pe) return;

    cudaDeviceSynchronize();
    this->deallocate();

    size_t comp_size = (size_t)batch_size * num_heads * timestep * head_dim;
    size_t qk_size = (size_t)batch_size * num_heads * timestep * timestep;

    realloc_pair(&d_mu_q, &d_var_q, comp_size);
    realloc_pair(&d_mu_k, &d_var_k, comp_size);
    realloc_pair(&d_mu_v, &d_var_v, comp_size);
    realloc_pair(&d_mu_qk, &d_var_qk, qk_size);
    realloc_pair(&d_mu_att_score, &d_var_att_score, qk_size);
    realloc_single(&d_j_mqk, qk_size);
    realloc_pair(&d_mu_sv, &d_var_sv, comp_size);
    if (need_pe) {
        realloc_pair(&d_mu_q_pe, &d_var_q_pe, comp_size);
        realloc_pair(&d_mu_k_pe, &d_var_k_pe, comp_size);
    }

    allocated_batch_size = batch_size;
    allocated_num_heads = num_heads;
    allocated_timestep = timestep;
    allocated_head_dim = head_dim;
    alloc_pe = need_pe;
}

void AttentionStatesCuda::deallocate() {
    free_pair(&d_mu_q, &d_var_q);
    free_pair(&d_mu_k, &d_var_k);
    free_pair(&d_mu_v, &d_var_v);
    free_pair(&d_mu_q_pe, &d_var_q_pe);
    free_pair(&d_mu_k_pe, &d_var_k_pe);
    free_pair(&d_mu_qk, &d_var_qk);
    free_pair(&d_mu_att_score, &d_var_att_score);
    free_single(&d_j_mqk);
    free_pair(&d_mu_sv, &d_var_sv);
    allocated_batch_size = allocated_num_heads = allocated_timestep =
        allocated_head_dim = 0;
    alloc_pe = false;
}

AttentionDeltaStatesCuda::~AttentionDeltaStatesCuda() { this->deallocate(); }

void AttentionDeltaStatesCuda::set_size(int batch_size, int num_heads,
                                        int timestep, int head_dim,
                                        bool need_pe) {
    bool same_shape =
        (batch_size == allocated_batch_size &&
         num_heads == allocated_num_heads && timestep == allocated_timestep &&
         head_dim == allocated_head_dim);
    bool same_pe = (alloc_pe == need_pe);
    if (same_shape && same_pe) return;

    cudaDeviceSynchronize();
    this->deallocate();

    size_t comp_size = (size_t)batch_size * num_heads * timestep * head_dim;
    size_t qk_size = (size_t)batch_size * num_heads * timestep * timestep;

    realloc_pair(&d_delta_mu_buffer, &d_delta_var_buffer, comp_size);
    realloc_pair(&d_delta_mu_v, &d_delta_var_v, comp_size);
    realloc_pair(&d_delta_mu_att_score, &d_delta_var_att_score, qk_size);
    realloc_pair(&d_delta_mu_q, &d_delta_var_q, comp_size);
    realloc_pair(&d_delta_mu_k, &d_delta_var_k, comp_size);
    if (need_pe) {
        realloc_pair(&d_delta_mu_q_pe, &d_delta_var_q_pe, comp_size);
        realloc_pair(&d_delta_mu_k_pe, &d_delta_var_k_pe, comp_size);
    }

    allocated_batch_size = batch_size;
    allocated_num_heads = num_heads;
    allocated_timestep = timestep;
    allocated_head_dim = head_dim;
    alloc_pe = need_pe;
}

void AttentionDeltaStatesCuda::deallocate() {
    free_pair(&d_delta_mu_buffer, &d_delta_var_buffer);
    free_pair(&d_delta_mu_v, &d_delta_var_v);
    free_pair(&d_delta_mu_att_score, &d_delta_var_att_score);
    free_pair(&d_delta_mu_q, &d_delta_var_q);
    free_pair(&d_delta_mu_k, &d_delta_var_k);
    free_pair(&d_delta_mu_q_pe, &d_delta_var_q_pe);
    free_pair(&d_delta_mu_k_pe, &d_delta_var_k_pe);
    allocated_batch_size = allocated_num_heads = allocated_timestep =
        allocated_head_dim = 0;
    alloc_pe = false;
}

////////////////////////////////////////////////////////////////////////////////
// Local helpers
////////////////////////////////////////////////////////////////////////////////
namespace {
constexpr int THREADS = 256;

inline int blocks_for(int total) { return (total + THREADS - 1) / THREADS; }

inline void ensure_delta_size(DeltaStateCuda &d, size_t new_size,
                              size_t block_size) {
    if (new_size > d.size) {
        cudaDeviceSynchronize();
        if (d.d_delta_mu) {
            cudaFree(d.d_delta_mu);
            d.d_delta_mu = nullptr;
        }
        if (d.d_delta_var) {
            cudaFree(d.d_delta_var);
            d.d_delta_var = nullptr;
        }
        d.size = new_size;
        cudaMalloc(&d.d_delta_mu, new_size * sizeof(float));
        cudaMalloc(&d.d_delta_var, new_size * sizeof(float));
    }
    d.block_size = block_size;
    d.actual_size = new_size / block_size;
}

inline void d2h(std::vector<float> &dst, const float *src, size_t n) {
    dst.resize(n);
    cudaMemcpy(dst.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost);
}

// Copy three buffers (d_mu_a, d_var_a, d_jcb) D2D between two HiddenStateCuda.
inline void copy_remax_output(HiddenStateCuda &src, float *dst_mu,
                              float *dst_var, float *dst_jcb, size_t n) {
    cudaMemcpy(dst_mu, src.d_mu_a, n * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst_var, src.d_var_a, n * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst_jcb, src.d_jcb, n * sizeof(float), cudaMemcpyDeviceToDevice);
}
}  // namespace

////////////////////////////////////////////////////////////////////////////////
// MultiheadAttentionCuda
////////////////////////////////////////////////////////////////////////////////
MultiheadAttentionCuda::MultiheadAttentionCuda(
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
    this->num_weights =
        embed_dim * ((num_heads + 2 * num_kv_heads) * this->head_dim);
    this->num_biases = 0;
    if (this->bias) {
        this->num_biases = (num_heads + 2 * num_kv_heads) * this->head_dim;
    }

    if (this->training) {
        this->allocate_param_delta();
    }

    remax_layer = std::make_unique<RemaxCuda>();

    if (this->pos_emb == "rope") {
        generate_rope_cache(this->max_seq_len, this->head_dim, this->rope_theta,
                            this->cos_cache, this->sin_cache);
        size_t bytes = this->cos_cache.size() * sizeof(float);
        cudaSetDevice(this->device_idx);
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cos_cache, bytes));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sin_cache, bytes));
        CHECK_CUDA_ERROR(cudaMemcpy(d_cos_cache, this->cos_cache.data(), bytes,
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_sin_cache, this->sin_cache.data(), bytes,
                                    cudaMemcpyHostToDevice));
    }
}

MultiheadAttentionCuda::~MultiheadAttentionCuda() {
    if (d_cos_cache) cudaFree(d_cos_cache);
    if (d_sin_cache) cudaFree(d_sin_cache);
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

void MultiheadAttentionCuda::init_weight_bias() {
    int qkv_output = (num_heads + 2 * num_kv_heads) * head_dim;
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_linear(this->init_method, this->gain_w, this->gain_b,
                                this->embed_dim, qkv_output, this->num_weights,
                                this->num_biases);
    this->allocate_param_memory();
    this->params_to_device();
}

void MultiheadAttentionCuda::forward(BaseHiddenStates &input_states,
                                     BaseHiddenStates &output_states,
                                     BaseTempStates &temp_states) {
    HiddenStateCuda *cu_in = dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_out = dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = cu_in->block_size;
    int timestep = (int)this->seq_len;
    int num_heads = (int)this->num_heads;
    int head_dim = (int)this->head_dim;
    int batch_seq = batch_size * timestep;

    if (batch_seq <= 0 || num_heads <= 0 || head_dim <= 0) return;

    this->set_cap_factor_udapte(batch_seq);

    bool need_pe = (this->pos_emb == "rope");
    attn_states.set_size(batch_size, num_heads, timestep, head_dim, need_pe);

    size_t comp = (size_t)batch_size * num_heads * timestep * head_dim;
    size_t qk = (size_t)batch_size * num_heads * timestep * timestep;
    int qkv_output = (int)((num_heads + 2 * num_kv_heads) * head_dim);

    // QKV linear projection -> in_proj_buffer.
    in_proj_buffer.set_size((size_t)batch_seq * qkv_output, batch_seq);
    in_proj_buffer.actual_size = qkv_output;
    in_proj_buffer.block_size = batch_seq;

    HiddenStateCuda *p_in = cu_in;
    HiddenStateCuda *p_proj = &in_proj_buffer;
    linear_forward_cuda(p_in, p_proj, this->d_mu_w, this->d_var_w, this->d_mu_b,
                        this->d_var_b, this->embed_dim, (size_t)qkv_output,
                        batch_seq, this->bias);

    // Split in_proj into Q,K,V.
    separate_in_proj_components_kernel<<<blocks_for((int)comp), THREADS>>>(
        in_proj_buffer.d_mu_a, in_proj_buffer.d_var_a, batch_size, num_heads,
        timestep, head_dim, attn_states.d_mu_q, attn_states.d_var_q,
        attn_states.d_mu_k, attn_states.d_var_k, attn_states.d_mu_v,
        attn_states.d_var_v);

    // Optional RoPE on Q and K.
    float *q_for_qk_mu = attn_states.d_mu_q;
    float *q_for_qk_var = attn_states.d_var_q;
    float *k_for_qk_mu = attn_states.d_mu_k;
    float *k_for_qk_var = attn_states.d_var_k;
    if (need_pe) {
        int rope_total = batch_size * num_heads * timestep * (head_dim / 2);
        apply_rope_kernel<<<blocks_for(rope_total), THREADS>>>(
            attn_states.d_mu_q, attn_states.d_var_q, d_cos_cache, d_sin_cache,
            batch_size, num_heads, timestep, head_dim, attn_states.d_mu_q_pe,
            attn_states.d_var_q_pe);
        apply_rope_kernel<<<blocks_for(rope_total), THREADS>>>(
            attn_states.d_mu_k, attn_states.d_var_k, d_cos_cache, d_sin_cache,
            batch_size, num_heads, timestep, head_dim, attn_states.d_mu_k_pe,
            attn_states.d_var_k_pe);
        q_for_qk_mu = attn_states.d_mu_q_pe;
        q_for_qk_var = attn_states.d_var_q_pe;
        k_for_qk_mu = attn_states.d_mu_k_pe;
        k_for_qk_var = attn_states.d_var_k_pe;
    }

    // QK^T scaled.
    query_key_kernel<<<blocks_for((int)qk), THREADS>>>(
        q_for_qk_mu, q_for_qk_var, k_for_qk_mu, k_for_qk_var, batch_size,
        num_heads, timestep, head_dim, attn_states.d_mu_qk,
        attn_states.d_var_qk);

    // Build remax_input by D2D copy from QK; mask in-place if causal.
    remax_input.set_size(qk, (size_t)batch_size * num_heads * timestep);
    remax_input.actual_size = timestep;
    remax_input.block_size = (size_t)batch_size * num_heads * timestep;
    remax_input.seq_len = 1;

    cudaMemcpy(remax_input.d_mu_a, attn_states.d_mu_qk, qk * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(remax_input.d_var_a, attn_states.d_var_qk, qk * sizeof(float),
               cudaMemcpyDeviceToDevice);

    if (this->use_causal_mask) {
        apply_causal_mask_pre_remax_kernel<<<blocks_for((int)qk), THREADS>>>(
            remax_input.d_mu_a, remax_input.d_var_a, batch_size, num_heads,
            timestep);
    }

    // Optional debug stats (matches CPU path: D2H + reuse of CPU printers).
    bool fire = this->debug &&
                (this->_debug_step % std::max(1, this->debug_interval) == 0);
    if (fire) {
        std::printf("[attn-diag] MHACuda forward step=%d (rope=%s, mask=%d)\n",
                    this->_debug_step, this->pos_emb.c_str(),
                    (int)this->use_causal_mask);
        std::vector<float> h_mu, h_var;
        d2h(h_mu, this->d_mu_w, this->num_weights);
        d2h(h_var, this->d_var_w, this->num_weights);
        print_magnitude_stats("W_qkv", h_mu, h_var);
        if (this->bias) {
            d2h(h_mu, this->d_mu_b, this->num_biases);
            d2h(h_var, this->d_var_b, this->num_biases);
            print_magnitude_stats("b_qkv", h_mu, h_var);
        }
        if (need_pe) {
            d2h(h_mu, attn_states.d_mu_q_pe, comp);
            d2h(h_var, attn_states.d_var_q_pe, comp);
            print_magnitude_stats("Q(rope)", h_mu, h_var);
            d2h(h_mu, attn_states.d_mu_k_pe, comp);
            d2h(h_var, attn_states.d_var_k_pe, comp);
            print_magnitude_stats("K(rope)", h_mu, h_var);
        } else {
            d2h(h_mu, attn_states.d_mu_q, comp);
            d2h(h_var, attn_states.d_var_q, comp);
            print_magnitude_stats("Q", h_mu, h_var);
            d2h(h_mu, attn_states.d_mu_k, comp);
            d2h(h_var, attn_states.d_var_k, comp);
            print_magnitude_stats("K", h_mu, h_var);
        }
        d2h(h_mu, attn_states.d_mu_v, comp);
        d2h(h_var, attn_states.d_var_v, comp);
        print_magnitude_stats("V", h_mu, h_var);
        d2h(h_mu, remax_input.d_mu_a, qk);
        d2h(h_var, remax_input.d_var_a, qk);
        if (this->use_causal_mask) {
            print_magnitude_stats_causal("QK", h_mu, h_var, batch_size,
                                         num_heads, timestep);
        } else {
            print_magnitude_stats("QK", h_mu, h_var);
        }
    }

    // Remax.
    remax_output.set_size(qk, (size_t)batch_size * num_heads * timestep);
    remax_output.actual_size = timestep;
    remax_output.block_size = (size_t)batch_size * num_heads * timestep;
    remax_output.seq_len = 1;
    remax_layer->forward(remax_input, remax_output, remax_temp);

    // Apply post-remax causal mask in-place on remax_output.
    if (this->use_causal_mask) {
        apply_causal_mask_post_remax_kernel<<<blocks_for((int)qk), THREADS>>>(
            remax_output.d_mu_a, remax_output.d_var_a, remax_output.d_jcb,
            batch_size, num_heads, timestep);
    }

    // Copy attention scores + jcb into attn_states (used in backward).
    copy_remax_output(remax_output, attn_states.d_mu_att_score,
                      attn_states.d_var_att_score, attn_states.d_j_mqk, qk);

    if (fire) {
        std::vector<float> h_mu, h_var;
        d2h(h_mu, remax_layer->d_mu_m, qk);
        d2h(h_var, remax_layer->d_var_m, qk);
        print_magnitude_stats("remax.M", h_mu, h_var);
        d2h(h_mu, remax_layer->d_mu_log_m, qk);
        d2h(h_var, remax_layer->d_var_log_m, qk);
        print_magnitude_stats("remax.logM", h_mu, h_var);
        d2h(h_mu, attn_states.d_mu_att_score, qk);
        d2h(h_var, attn_states.d_var_att_score, qk);
        if (this->use_causal_mask) {
            print_magnitude_stats_causal("att_score", h_mu, h_var, batch_size,
                                         num_heads, timestep);
        } else {
            print_magnitude_stats("att_score", h_mu, h_var);
        }
        d2h(h_mu, attn_states.d_j_mqk, qk);
        if (this->use_causal_mask) {
            print_magnitude_stats_causal("j_mqk", h_mu, h_mu, batch_size,
                                         num_heads, timestep);
        } else {
            print_magnitude_stats("j_mqk", h_mu, h_mu);
        }
    }

    // SV = att_score @ V.
    att_score_value_kernel<<<blocks_for((int)comp), THREADS>>>(
        attn_states.d_mu_att_score, attn_states.d_var_att_score,
        attn_states.d_mu_v, attn_states.d_var_v, batch_size, num_heads,
        timestep, head_dim, attn_states.d_mu_sv, attn_states.d_var_sv);

    // Project [B,H,T,D] -> [B,T,H,D] into output.
    // (B=batch_size, H=num_heads, T=timestep, D=head_dim).
    project_output_forward_kernel<<<blocks_for((int)comp), THREADS>>>(
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
    this->_debug_step++;
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
    int timestep = (int)this->seq_len;
    int num_heads = (int)this->num_heads;
    int head_dim = (int)this->head_dim;
    int batch_seq = batch_size * timestep;

    if (batch_seq <= 0 || num_heads <= 0 || head_dim <= 0) return;

    bool need_pe = (this->pos_emb == "rope");
    attn_delta_states.set_size(batch_size, num_heads, timestep, head_dim,
                               need_pe);

    size_t comp = (size_t)batch_size * num_heads * timestep * head_dim;
    size_t qk = (size_t)batch_size * num_heads * timestep * timestep;
    int qkv_output = (int)((num_heads + 2 * num_kv_heads) * head_dim);

    // Project [B,T,H,D] -> [B,H,T,D] into delta_buffer.
    // (B=batch_size, T=timestep, H=num_heads, D=head_dim).
    project_output_backward_kernel<<<blocks_for((int)comp), THREADS>>>(
        cu_in_delta->d_delta_mu, cu_in_delta->d_delta_var, batch_size,
        num_heads, timestep, head_dim, attn_delta_states.d_delta_mu_buffer,
        attn_delta_states.d_delta_var_buffer);

    // delta_v.
    mha_delta_value_kernel<<<blocks_for((int)comp), THREADS>>>(
        attn_states.d_mu_att_score, attn_delta_states.d_delta_mu_buffer,
        attn_delta_states.d_delta_var_buffer, batch_size, num_heads, timestep,
        head_dim, attn_delta_states.d_delta_mu_v,
        attn_delta_states.d_delta_var_v);

    // delta_att_score.
    mha_delta_score_kernel<<<blocks_for((int)qk), THREADS>>>(
        attn_states.d_mu_v, attn_delta_states.d_delta_mu_buffer,
        attn_delta_states.d_delta_var_buffer, batch_size, num_heads, timestep,
        head_dim, attn_delta_states.d_delta_mu_att_score,
        attn_delta_states.d_delta_var_att_score);

    // delta_q, delta_k (rope or non-rope path).
    if (need_pe) {
        mha_delta_query_kernel<<<blocks_for((int)comp), THREADS>>>(
            attn_states.d_mu_k_pe, attn_delta_states.d_delta_mu_att_score,
            attn_delta_states.d_delta_var_att_score, attn_states.d_j_mqk,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_q_pe,
            attn_delta_states.d_delta_var_q_pe);
        mha_delta_key_kernel<<<blocks_for((int)comp), THREADS>>>(
            attn_states.d_mu_q_pe, attn_delta_states.d_delta_mu_att_score,
            attn_delta_states.d_delta_var_att_score, attn_states.d_j_mqk,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_k_pe,
            attn_delta_states.d_delta_var_k_pe);
        int rope_total = batch_size * num_heads * timestep * (head_dim / 2);
        rope_backward_kernel<<<blocks_for(rope_total), THREADS>>>(
            attn_delta_states.d_delta_mu_q_pe,
            attn_delta_states.d_delta_var_q_pe, d_cos_cache, d_sin_cache,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_q, attn_delta_states.d_delta_var_q);
        rope_backward_kernel<<<blocks_for(rope_total), THREADS>>>(
            attn_delta_states.d_delta_mu_k_pe,
            attn_delta_states.d_delta_var_k_pe, d_cos_cache, d_sin_cache,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_k, attn_delta_states.d_delta_var_k);
    } else {
        mha_delta_query_kernel<<<blocks_for((int)comp), THREADS>>>(
            attn_states.d_mu_k, attn_delta_states.d_delta_mu_att_score,
            attn_delta_states.d_delta_var_att_score, attn_states.d_j_mqk,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_q, attn_delta_states.d_delta_var_q);
        mha_delta_key_kernel<<<blocks_for((int)comp), THREADS>>>(
            attn_states.d_mu_q, attn_delta_states.d_delta_mu_att_score,
            attn_delta_states.d_delta_var_att_score, attn_states.d_j_mqk,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_k, attn_delta_states.d_delta_var_k);
    }

    // Concat Q,K,V deltas into in_proj layout (using d_in_proj_buffer).
    ensure_delta_size(d_in_proj_buffer, (size_t)batch_seq * qkv_output,
                      batch_seq);
    cat_in_proj_components_kernel<<<blocks_for((int)comp), THREADS>>>(
        attn_delta_states.d_delta_mu_q, attn_delta_states.d_delta_var_q,
        attn_delta_states.d_delta_mu_k, attn_delta_states.d_delta_var_k,
        attn_delta_states.d_delta_mu_v, attn_delta_states.d_delta_var_v,
        batch_size, num_heads, timestep, head_dim, d_in_proj_buffer.d_delta_mu,
        d_in_proj_buffer.d_delta_var);

    BackwardStateCuda *cu_bwd =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());

    // State and weight gradients via the existing linear CUDA wrappers.
    DeltaStateCuda *p_in_dproj = &d_in_proj_buffer;
    if (state_udapte) {
        DeltaStateCuda *p_out = cu_out_delta;
        linear_state_backward_cuda(p_in_dproj, p_out, cu_bwd, this->d_mu_w,
                                   this->embed_dim, (size_t)qkv_output,
                                   batch_seq);
        cu_out_delta->seq_len = timestep;
    }

    if (this->param_update) {
        DeltaStateCuda *p_out = cu_out_delta;
        linear_weight_backward_cuda(p_in_dproj, p_out, cu_bwd, this->d_var_w,
                                    this->embed_dim, (size_t)qkv_output,
                                    batch_seq, this->d_delta_mu_w,
                                    this->d_delta_var_w);

        if (this->bias) {
            unsigned int blk_b = (qkv_output + 16 - 1) / 16;
            dim3 grid_b(1, blk_b);
            dim3 block_b(16, 16);
            linear_bwd_delta_b<<<grid_b, block_b>>>(
                this->d_var_b, d_in_proj_buffer.d_delta_mu,
                d_in_proj_buffer.d_delta_var, this->embed_dim,
                (size_t)qkv_output, batch_seq, this->d_delta_mu_b,
                this->d_delta_var_b);
        }
    }

    int prev_step = this->_debug_step - 1;
    bool fire = this->debug && prev_step >= 0 &&
                (prev_step % std::max(1, this->debug_interval) == 0);
    if (fire) {
        std::vector<float> h_mu, h_var;
        std::printf("[attn-diag] MHACuda backward step=%d\n", prev_step);
        d2h(h_mu, this->d_delta_mu_w, this->num_weights);
        d2h(h_var, this->d_delta_var_w, this->num_weights);
        print_magnitude_stats("dW_qkv", h_mu, h_var);
        d2h(h_mu, d_in_proj_buffer.d_delta_mu, (size_t)batch_seq * qkv_output);
        d2h(h_var, d_in_proj_buffer.d_delta_var,
            (size_t)batch_seq * qkv_output);
        print_magnitude_stats("d_in_proj", h_mu, h_var);
    }

    CHECK_LAST_CUDA_ERROR();
}

std::unique_ptr<BaseLayer> MultiheadAttentionCuda::to_host() {
    auto host = std::make_unique<MultiheadAttention>(
        this->embed_dim, this->num_heads, this->num_kv_heads, this->seq_len,
        this->bias, this->gain_w, this->gain_b, this->init_method,
        this->pos_emb, this->rope_theta, this->max_seq_len,
        this->use_causal_mask);
    this->params_to_host();
    host->mu_w = this->mu_w;
    host->var_w = this->var_w;
    host->mu_b = this->mu_b;
    host->var_b = this->var_b;
    return host;
}

AttentionScores MultiheadAttentionCuda::get_attention_scores() {
    AttentionScores s;
    s.batch_size = attn_states.allocated_batch_size;
    s.num_heads = attn_states.allocated_num_heads;
    s.timestep = attn_states.allocated_timestep;
    if (s.batch_size <= 0 || s.num_heads <= 0 || s.timestep <= 0) return s;
    size_t total = (size_t)s.batch_size * s.num_heads * s.timestep * s.timestep;
    s.mu.resize(total);
    s.var.resize(total);
    cudaMemcpy(s.mu.data(), attn_states.d_mu_att_score, total * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(s.var.data(), attn_states.d_var_att_score, total * sizeof(float),
               cudaMemcpyDeviceToHost);
    return s;
}

////////////////////////////////////////////////////////////////////////////////
// MultiheadAttentionV2Cuda
////////////////////////////////////////////////////////////////////////////////
MultiheadAttentionV2Cuda::MultiheadAttentionV2Cuda(
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
    num_biases_q = this->bias ? q_output_size : 0;
    num_biases_k = this->bias ? k_output_size : 0;
    num_biases_v = this->bias ? v_output_size : 0;

    // BaseLayer's num_weights/num_biases are unused here.
    this->num_weights = 0;
    this->num_biases = 0;

    if (this->training) {
        this->allocate_param_delta();
    }

    // V2 manages its own Q/K/V param buffers and doesn't go through
    // BaseLayerCuda::allocate_param_memory, so we must allocate
    // `d_neg_var_count` ourselves; update_weights uses it.
    cudaSetDevice(this->device_idx);
    CHECK_CUDA_ERROR(cudaMalloc((void **)&this->d_neg_var_count, sizeof(int)));

    remax_layer = std::make_unique<RemaxCuda>();

    if (this->pos_emb == "rope") {
        generate_rope_cache(this->max_seq_len, this->head_dim, this->rope_theta,
                            this->cos_cache, this->sin_cache);
        size_t bytes = this->cos_cache.size() * sizeof(float);
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cos_cache, bytes));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sin_cache, bytes));
        CHECK_CUDA_ERROR(cudaMemcpy(d_cos_cache, this->cos_cache.data(), bytes,
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_sin_cache, this->sin_cache.data(), bytes,
                                    cudaMemcpyHostToDevice));
    }
}

MultiheadAttentionV2Cuda::~MultiheadAttentionV2Cuda() {
    free_pair(&d_mu_w_q, &d_var_w_q);
    free_pair(&d_mu_w_k, &d_var_w_k);
    free_pair(&d_mu_w_v, &d_var_w_v);
    free_pair(&d_mu_b_q, &d_var_b_q);
    free_pair(&d_mu_b_k, &d_var_b_k);
    free_pair(&d_mu_b_v, &d_var_b_v);
    free_pair(&d_delta_mu_w_q, &d_delta_var_w_q);
    free_pair(&d_delta_mu_w_k, &d_delta_var_w_k);
    free_pair(&d_delta_mu_w_v, &d_delta_var_w_v);
    free_pair(&d_delta_mu_b_q, &d_delta_var_b_q);
    free_pair(&d_delta_mu_b_k, &d_delta_var_b_k);
    free_pair(&d_delta_mu_b_v, &d_delta_var_b_v);
    if (d_cos_cache) cudaFree(d_cos_cache);
    if (d_sin_cache) cudaFree(d_sin_cache);
}

std::string MultiheadAttentionV2Cuda::get_layer_info() const {
    return "SelfAttentionV2(heads=" + std::to_string(this->num_heads) +
           ", kv_heads=" + std::to_string(this->num_kv_heads) +
           ", emb_size=" + std::to_string(this->embed_dim) + ")";
}

std::string MultiheadAttentionV2Cuda::get_layer_name() const {
    return "MultiheadAttentionV2Cuda";
}

LayerType MultiheadAttentionV2Cuda::get_layer_type() const {
    return LayerType::MultiheadAttention;
}

void MultiheadAttentionV2Cuda::init_weight_bias() {
    auto init_q = init_weight_bias_linear(
        this->init_method, this->gain_w, this->gain_b, this->embed_dim,
        q_output_size, num_weights_q, num_biases_q);
    auto init_k = init_weight_bias_linear(
        this->init_method, this->gain_w, this->gain_b, this->embed_dim,
        k_output_size, num_weights_k, num_biases_k);
    auto init_v = init_weight_bias_linear(
        this->init_method, this->gain_w, this->gain_b, this->embed_dim,
        v_output_size, num_weights_v, num_biases_v);
    mu_w_q = std::get<0>(init_q);
    var_w_q = std::get<1>(init_q);
    mu_b_q = std::get<2>(init_q);
    var_b_q = std::get<3>(init_q);
    mu_w_k = std::get<0>(init_k);
    var_w_k = std::get<1>(init_k);
    mu_b_k = std::get<2>(init_k);
    var_b_k = std::get<3>(init_k);
    mu_w_v = std::get<0>(init_v);
    var_w_v = std::get<1>(init_v);
    mu_b_v = std::get<2>(init_v);
    var_b_v = std::get<3>(init_v);

    cudaSetDevice(this->device_idx);
    realloc_pair(&d_mu_w_q, &d_var_w_q, num_weights_q);
    realloc_pair(&d_mu_w_k, &d_var_w_k, num_weights_k);
    realloc_pair(&d_mu_w_v, &d_var_w_v, num_weights_v);
    if (this->bias) {
        realloc_pair(&d_mu_b_q, &d_var_b_q, num_biases_q);
        realloc_pair(&d_mu_b_k, &d_var_b_k, num_biases_k);
        realloc_pair(&d_mu_b_v, &d_var_b_v, num_biases_v);
    }
    this->params_to_device();
}

void MultiheadAttentionV2Cuda::allocate_param_delta() {
    cudaSetDevice(this->device_idx);
    delta_mu_w_q.assign(num_weights_q, 0.0f);
    delta_var_w_q.assign(num_weights_q, 0.0f);
    delta_mu_w_k.assign(num_weights_k, 0.0f);
    delta_var_w_k.assign(num_weights_k, 0.0f);
    delta_mu_w_v.assign(num_weights_v, 0.0f);
    delta_var_w_v.assign(num_weights_v, 0.0f);
    realloc_pair(&d_delta_mu_w_q, &d_delta_var_w_q, num_weights_q);
    realloc_pair(&d_delta_mu_w_k, &d_delta_var_w_k, num_weights_k);
    realloc_pair(&d_delta_mu_w_v, &d_delta_var_w_v, num_weights_v);
    if (this->bias) {
        delta_mu_b_q.assign(num_biases_q, 0.0f);
        delta_var_b_q.assign(num_biases_q, 0.0f);
        delta_mu_b_k.assign(num_biases_k, 0.0f);
        delta_var_b_k.assign(num_biases_k, 0.0f);
        delta_mu_b_v.assign(num_biases_v, 0.0f);
        delta_var_b_v.assign(num_biases_v, 0.0f);
        realloc_pair(&d_delta_mu_b_q, &d_delta_var_b_q, num_biases_q);
        realloc_pair(&d_delta_mu_b_k, &d_delta_var_b_k, num_biases_k);
        realloc_pair(&d_delta_mu_b_v, &d_delta_var_b_v, num_biases_v);
    }
}

void MultiheadAttentionV2Cuda::params_to_device() {
    cudaSetDevice(this->device_idx);
    cudaMemcpy(d_mu_w_q, mu_w_q.data(), num_weights_q * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_w_q, var_w_q.data(), num_weights_q * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu_w_k, mu_w_k.data(), num_weights_k * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_w_k, var_w_k.data(), num_weights_k * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu_w_v, mu_w_v.data(), num_weights_v * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_w_v, var_w_v.data(), num_weights_v * sizeof(float),
               cudaMemcpyHostToDevice);
    if (this->bias) {
        cudaMemcpy(d_mu_b_q, mu_b_q.data(), num_biases_q * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_var_b_q, var_b_q.data(), num_biases_q * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_mu_b_k, mu_b_k.data(), num_biases_k * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_var_b_k, var_b_k.data(), num_biases_k * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_mu_b_v, mu_b_v.data(), num_biases_v * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_var_b_v, var_b_v.data(), num_biases_v * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
    CHECK_LAST_CUDA_ERROR();
}

void MultiheadAttentionV2Cuda::params_to_host() {
    cudaSetDevice(this->device_idx);
    mu_w_q.resize(num_weights_q);
    var_w_q.resize(num_weights_q);
    mu_w_k.resize(num_weights_k);
    var_w_k.resize(num_weights_k);
    mu_w_v.resize(num_weights_v);
    var_w_v.resize(num_weights_v);
    cudaMemcpy(mu_w_q.data(), d_mu_w_q, num_weights_q * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(var_w_q.data(), d_var_w_q, num_weights_q * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(mu_w_k.data(), d_mu_w_k, num_weights_k * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(var_w_k.data(), d_var_w_k, num_weights_k * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(mu_w_v.data(), d_mu_w_v, num_weights_v * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(var_w_v.data(), d_var_w_v, num_weights_v * sizeof(float),
               cudaMemcpyDeviceToHost);
    if (this->bias) {
        mu_b_q.resize(num_biases_q);
        var_b_q.resize(num_biases_q);
        mu_b_k.resize(num_biases_k);
        var_b_k.resize(num_biases_k);
        mu_b_v.resize(num_biases_v);
        var_b_v.resize(num_biases_v);
        cudaMemcpy(mu_b_q.data(), d_mu_b_q, num_biases_q * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(var_b_q.data(), d_var_b_q, num_biases_q * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(mu_b_k.data(), d_mu_b_k, num_biases_k * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(var_b_k.data(), d_var_b_k, num_biases_k * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(mu_b_v.data(), d_mu_b_v, num_biases_v * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(var_b_v.data(), d_var_b_v, num_biases_v * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    CHECK_LAST_CUDA_ERROR();
}

// Forward declarations for the existing capped-update kernels in
// base_layer_cuda.cu — used to mirror CPU `capped_update` per Q/K/V tensor.
__global__ void device_weight_update(float const *delta_mu_w,
                                     float const *delta_var_w,
                                     float cap_factor_udapte, size_t size,
                                     float *mu_w, float *var_w,
                                     int *negative_var_count);
__global__ void device_bias_update(float const *delta_mu_b,
                                   float const *delta_var_b,
                                   float cap_factor_udapte, size_t size,
                                   float *mu_b, float *var_b);

namespace {
inline void capped_update_param(float *d_mu, float *d_var,
                                const float *d_delta_mu,
                                const float *d_delta_var, size_t n,
                                float cap_factor, int *d_neg_count, bool is_w) {
    constexpr int THR = 256;
    unsigned int blk = (n + THR - 1) / THR;
    if (is_w) {
        device_weight_update<<<blk, THR>>>(d_delta_mu, d_delta_var, cap_factor,
                                           n, d_mu, d_var, d_neg_count);
    } else {
        device_bias_update<<<blk, THR>>>(d_delta_mu, d_delta_var, cap_factor, n,
                                         d_mu, d_var);
    }
}
}  // namespace

void MultiheadAttentionV2Cuda::update_weights() {
    cudaSetDevice(this->device_idx);
    this->neg_var_w_counter = 0;
    int zero = 0;
    cudaMemcpy(this->d_neg_var_count, &zero, sizeof(int),
               cudaMemcpyHostToDevice);
    capped_update_param(d_mu_w_q, d_var_w_q, d_delta_mu_w_q, d_delta_var_w_q,
                        num_weights_q, this->cap_factor_update,
                        this->d_neg_var_count, true);
    capped_update_param(d_mu_w_k, d_var_w_k, d_delta_mu_w_k, d_delta_var_w_k,
                        num_weights_k, this->cap_factor_update,
                        this->d_neg_var_count, true);
    capped_update_param(d_mu_w_v, d_var_w_v, d_delta_mu_w_v, d_delta_var_w_v,
                        num_weights_v, this->cap_factor_update,
                        this->d_neg_var_count, true);
    cudaMemcpy(&this->neg_var_w_counter, this->d_neg_var_count, sizeof(int),
               cudaMemcpyDeviceToHost);
    CHECK_LAST_CUDA_ERROR();
}

void MultiheadAttentionV2Cuda::update_biases() {
    if (!this->bias) return;
    cudaSetDevice(this->device_idx);
    capped_update_param(d_mu_b_q, d_var_b_q, d_delta_mu_b_q, d_delta_var_b_q,
                        num_biases_q, this->cap_factor_update, nullptr, false);
    capped_update_param(d_mu_b_k, d_var_b_k, d_delta_mu_b_k, d_delta_var_b_k,
                        num_biases_k, this->cap_factor_update, nullptr, false);
    capped_update_param(d_mu_b_v, d_var_b_v, d_delta_mu_b_v, d_delta_var_b_v,
                        num_biases_v, this->cap_factor_update, nullptr, false);
    CHECK_LAST_CUDA_ERROR();
}

void MultiheadAttentionV2Cuda::forward(BaseHiddenStates &input_states,
                                       BaseHiddenStates &output_states,
                                       BaseTempStates &temp_states) {
    HiddenStateCuda *cu_in = dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_out = dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = cu_in->block_size;
    int timestep = (int)this->seq_len;
    int num_heads = (int)this->num_heads;
    int head_dim = (int)this->head_dim;
    int batch_seq = batch_size * timestep;

    if (batch_seq <= 0 || num_heads <= 0 || head_dim <= 0) return;

    this->set_cap_factor_udapte(batch_seq);

    bool need_pe = (this->pos_emb == "rope");
    attn_states.set_size(batch_size, num_heads, timestep, head_dim, need_pe);

    size_t comp = (size_t)batch_size * num_heads * timestep * head_dim;
    size_t qk = (size_t)batch_size * num_heads * timestep * timestep;

    // Three independent linear projections.
    q_proj_buffer.set_size((size_t)batch_seq * q_output_size, batch_seq);
    q_proj_buffer.actual_size = q_output_size;
    q_proj_buffer.block_size = batch_seq;
    k_proj_buffer.set_size((size_t)batch_seq * k_output_size, batch_seq);
    k_proj_buffer.actual_size = k_output_size;
    k_proj_buffer.block_size = batch_seq;
    v_proj_buffer.set_size((size_t)batch_seq * v_output_size, batch_seq);
    v_proj_buffer.actual_size = v_output_size;
    v_proj_buffer.block_size = batch_seq;

    HiddenStateCuda *p_in = cu_in;
    HiddenStateCuda *p_q = &q_proj_buffer;
    HiddenStateCuda *p_k = &k_proj_buffer;
    HiddenStateCuda *p_v = &v_proj_buffer;
    linear_forward_cuda(p_in, p_q, d_mu_w_q, d_var_w_q, d_mu_b_q, d_var_b_q,
                        this->embed_dim, q_output_size, batch_seq, this->bias);
    linear_forward_cuda(p_in, p_k, d_mu_w_k, d_var_w_k, d_mu_b_k, d_var_b_k,
                        this->embed_dim, k_output_size, batch_seq, this->bias);
    linear_forward_cuda(p_in, p_v, d_mu_w_v, d_var_w_v, d_mu_b_v, d_var_b_v,
                        this->embed_dim, v_output_size, batch_seq, this->bias);

    // Reshape projections into [B,H,T,D].
    reshape_proj_to_heads_kernel<<<blocks_for((int)comp), THREADS>>>(
        q_proj_buffer.d_mu_a, q_proj_buffer.d_var_a, batch_size, num_heads,
        timestep, head_dim, attn_states.d_mu_q, attn_states.d_var_q);
    reshape_proj_to_heads_kernel<<<blocks_for((int)comp), THREADS>>>(
        k_proj_buffer.d_mu_a, k_proj_buffer.d_var_a, batch_size, num_heads,
        timestep, head_dim, attn_states.d_mu_k, attn_states.d_var_k);
    reshape_proj_to_heads_kernel<<<blocks_for((int)comp), THREADS>>>(
        v_proj_buffer.d_mu_a, v_proj_buffer.d_var_a, batch_size, num_heads,
        timestep, head_dim, attn_states.d_mu_v, attn_states.d_var_v);

    // Optional RoPE on Q and K.
    float *q_for_qk_mu = attn_states.d_mu_q;
    float *q_for_qk_var = attn_states.d_var_q;
    float *k_for_qk_mu = attn_states.d_mu_k;
    float *k_for_qk_var = attn_states.d_var_k;
    if (need_pe) {
        int rope_total = batch_size * num_heads * timestep * (head_dim / 2);
        apply_rope_kernel<<<blocks_for(rope_total), THREADS>>>(
            attn_states.d_mu_q, attn_states.d_var_q, d_cos_cache, d_sin_cache,
            batch_size, num_heads, timestep, head_dim, attn_states.d_mu_q_pe,
            attn_states.d_var_q_pe);
        apply_rope_kernel<<<blocks_for(rope_total), THREADS>>>(
            attn_states.d_mu_k, attn_states.d_var_k, d_cos_cache, d_sin_cache,
            batch_size, num_heads, timestep, head_dim, attn_states.d_mu_k_pe,
            attn_states.d_var_k_pe);
        q_for_qk_mu = attn_states.d_mu_q_pe;
        q_for_qk_var = attn_states.d_var_q_pe;
        k_for_qk_mu = attn_states.d_mu_k_pe;
        k_for_qk_var = attn_states.d_var_k_pe;
    }

    // QK^T scaled.
    query_key_kernel<<<blocks_for((int)qk), THREADS>>>(
        q_for_qk_mu, q_for_qk_var, k_for_qk_mu, k_for_qk_var, batch_size,
        num_heads, timestep, head_dim, attn_states.d_mu_qk,
        attn_states.d_var_qk);

    // Remax input (mask in-place if causal).
    remax_input.set_size(qk, (size_t)batch_size * num_heads * timestep);
    remax_input.actual_size = timestep;
    remax_input.block_size = (size_t)batch_size * num_heads * timestep;
    remax_input.seq_len = 1;
    cudaMemcpy(remax_input.d_mu_a, attn_states.d_mu_qk, qk * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(remax_input.d_var_a, attn_states.d_var_qk, qk * sizeof(float),
               cudaMemcpyDeviceToDevice);
    if (this->use_causal_mask) {
        apply_causal_mask_pre_remax_kernel<<<blocks_for((int)qk), THREADS>>>(
            remax_input.d_mu_a, remax_input.d_var_a, batch_size, num_heads,
            timestep);
    }

    bool fire = this->debug &&
                (this->_debug_step % std::max(1, this->debug_interval) == 0);
    if (fire) {
        std::printf(
            "[attn-diag] MHAv2Cuda forward step=%d (rope=%s, mask=%d)\n",
            this->_debug_step, this->pos_emb.c_str(),
            (int)this->use_causal_mask);
        std::vector<float> h_mu, h_var;
        d2h(h_mu, d_mu_w_q, num_weights_q);
        d2h(h_var, d_var_w_q, num_weights_q);
        print_magnitude_stats("W_q", h_mu, h_var);
        d2h(h_mu, d_mu_w_k, num_weights_k);
        d2h(h_var, d_var_w_k, num_weights_k);
        print_magnitude_stats("W_k", h_mu, h_var);
        d2h(h_mu, d_mu_w_v, num_weights_v);
        d2h(h_var, d_var_w_v, num_weights_v);
        print_magnitude_stats("W_v", h_mu, h_var);
        if (this->bias) {
            d2h(h_mu, d_mu_b_q, num_biases_q);
            d2h(h_var, d_var_b_q, num_biases_q);
            print_magnitude_stats("b_q", h_mu, h_var);
            d2h(h_mu, d_mu_b_k, num_biases_k);
            d2h(h_var, d_var_b_k, num_biases_k);
            print_magnitude_stats("b_k", h_mu, h_var);
            d2h(h_mu, d_mu_b_v, num_biases_v);
            d2h(h_var, d_var_b_v, num_biases_v);
            print_magnitude_stats("b_v", h_mu, h_var);
        }
        if (need_pe) {
            d2h(h_mu, attn_states.d_mu_q_pe, comp);
            d2h(h_var, attn_states.d_var_q_pe, comp);
            print_magnitude_stats("Q(rope)", h_mu, h_var);
            d2h(h_mu, attn_states.d_mu_k_pe, comp);
            d2h(h_var, attn_states.d_var_k_pe, comp);
            print_magnitude_stats("K(rope)", h_mu, h_var);
        } else {
            d2h(h_mu, attn_states.d_mu_q, comp);
            d2h(h_var, attn_states.d_var_q, comp);
            print_magnitude_stats("Q", h_mu, h_var);
            d2h(h_mu, attn_states.d_mu_k, comp);
            d2h(h_var, attn_states.d_var_k, comp);
            print_magnitude_stats("K", h_mu, h_var);
        }
        d2h(h_mu, attn_states.d_mu_v, comp);
        d2h(h_var, attn_states.d_var_v, comp);
        print_magnitude_stats("V", h_mu, h_var);
        d2h(h_mu, remax_input.d_mu_a, qk);
        d2h(h_var, remax_input.d_var_a, qk);
        if (this->use_causal_mask) {
            print_magnitude_stats_causal("QK", h_mu, h_var, batch_size,
                                         num_heads, timestep);
        } else {
            print_magnitude_stats("QK", h_mu, h_var);
        }
    }

    // Remax.
    remax_output.set_size(qk, (size_t)batch_size * num_heads * timestep);
    remax_output.actual_size = timestep;
    remax_output.block_size = (size_t)batch_size * num_heads * timestep;
    remax_output.seq_len = 1;
    remax_layer->forward(remax_input, remax_output, remax_temp);

    if (this->use_causal_mask) {
        apply_causal_mask_post_remax_kernel<<<blocks_for((int)qk), THREADS>>>(
            remax_output.d_mu_a, remax_output.d_var_a, remax_output.d_jcb,
            batch_size, num_heads, timestep);
    }
    copy_remax_output(remax_output, attn_states.d_mu_att_score,
                      attn_states.d_var_att_score, attn_states.d_j_mqk, qk);

    // SV and output projection.
    att_score_value_kernel<<<blocks_for((int)comp), THREADS>>>(
        attn_states.d_mu_att_score, attn_states.d_var_att_score,
        attn_states.d_mu_v, attn_states.d_var_v, batch_size, num_heads,
        timestep, head_dim, attn_states.d_mu_sv, attn_states.d_var_sv);
    project_output_forward_kernel<<<blocks_for((int)comp), THREADS>>>(
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
    this->_debug_step++;
}

void MultiheadAttentionV2Cuda::backward(BaseDeltaStates &input_delta_states,
                                        BaseDeltaStates &output_delta_states,
                                        BaseTempStates &temp_states,
                                        bool state_udapte) {
    DeltaStateCuda *cu_in_delta =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_out_delta =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    int batch_size = cu_in_delta->block_size;
    int timestep = (int)this->seq_len;
    int num_heads = (int)this->num_heads;
    int head_dim = (int)this->head_dim;
    int batch_seq = batch_size * timestep;

    if (batch_seq <= 0 || num_heads <= 0 || head_dim <= 0) return;

    bool need_pe = (this->pos_emb == "rope");
    attn_delta_states.set_size(batch_size, num_heads, timestep, head_dim,
                               need_pe);

    size_t comp = (size_t)batch_size * num_heads * timestep * head_dim;
    size_t qk = (size_t)batch_size * num_heads * timestep * timestep;
    size_t input_flat = (size_t)batch_seq * this->embed_dim;

    // Project [B,T,H,D] -> [B,H,T,D] into delta_buffer.
    // (B=batch_size, T=timestep, H=num_heads, D=head_dim).
    project_output_backward_kernel<<<blocks_for((int)comp), THREADS>>>(
        cu_in_delta->d_delta_mu, cu_in_delta->d_delta_var, batch_size,
        num_heads, timestep, head_dim, attn_delta_states.d_delta_mu_buffer,
        attn_delta_states.d_delta_var_buffer);

    // delta_v.
    mha_delta_value_kernel<<<blocks_for((int)comp), THREADS>>>(
        attn_states.d_mu_att_score, attn_delta_states.d_delta_mu_buffer,
        attn_delta_states.d_delta_var_buffer, batch_size, num_heads, timestep,
        head_dim, attn_delta_states.d_delta_mu_v,
        attn_delta_states.d_delta_var_v);

    // delta_att_score.
    mha_delta_score_kernel<<<blocks_for((int)qk), THREADS>>>(
        attn_states.d_mu_v, attn_delta_states.d_delta_mu_buffer,
        attn_delta_states.d_delta_var_buffer, batch_size, num_heads, timestep,
        head_dim, attn_delta_states.d_delta_mu_att_score,
        attn_delta_states.d_delta_var_att_score);

    // delta_q, delta_k.
    if (need_pe) {
        mha_delta_query_kernel<<<blocks_for((int)comp), THREADS>>>(
            attn_states.d_mu_k_pe, attn_delta_states.d_delta_mu_att_score,
            attn_delta_states.d_delta_var_att_score, attn_states.d_j_mqk,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_q_pe,
            attn_delta_states.d_delta_var_q_pe);
        mha_delta_key_kernel<<<blocks_for((int)comp), THREADS>>>(
            attn_states.d_mu_q_pe, attn_delta_states.d_delta_mu_att_score,
            attn_delta_states.d_delta_var_att_score, attn_states.d_j_mqk,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_k_pe,
            attn_delta_states.d_delta_var_k_pe);
        int rope_total = batch_size * num_heads * timestep * (head_dim / 2);
        rope_backward_kernel<<<blocks_for(rope_total), THREADS>>>(
            attn_delta_states.d_delta_mu_q_pe,
            attn_delta_states.d_delta_var_q_pe, d_cos_cache, d_sin_cache,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_q, attn_delta_states.d_delta_var_q);
        rope_backward_kernel<<<blocks_for(rope_total), THREADS>>>(
            attn_delta_states.d_delta_mu_k_pe,
            attn_delta_states.d_delta_var_k_pe, d_cos_cache, d_sin_cache,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_k, attn_delta_states.d_delta_var_k);
    } else {
        mha_delta_query_kernel<<<blocks_for((int)comp), THREADS>>>(
            attn_states.d_mu_k, attn_delta_states.d_delta_mu_att_score,
            attn_delta_states.d_delta_var_att_score, attn_states.d_j_mqk,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_q, attn_delta_states.d_delta_var_q);
        mha_delta_key_kernel<<<blocks_for((int)comp), THREADS>>>(
            attn_states.d_mu_q, attn_delta_states.d_delta_mu_att_score,
            attn_delta_states.d_delta_var_att_score, attn_states.d_j_mqk,
            batch_size, num_heads, timestep, head_dim,
            attn_delta_states.d_delta_mu_k, attn_delta_states.d_delta_var_k);
    }

    // Reshape [B,H,T,D] -> [B*T, H*D] back into proj-shaped delta buffers.
    ensure_delta_size(dq_proj_buffer, (size_t)batch_seq * q_output_size,
                      batch_seq);
    ensure_delta_size(dk_proj_buffer, (size_t)batch_seq * k_output_size,
                      batch_seq);
    ensure_delta_size(dv_proj_buffer, (size_t)batch_seq * v_output_size,
                      batch_seq);
    reshape_heads_to_proj_kernel<<<blocks_for((int)comp), THREADS>>>(
        attn_delta_states.d_delta_mu_q, attn_delta_states.d_delta_var_q,
        batch_size, num_heads, timestep, head_dim, dq_proj_buffer.d_delta_mu,
        dq_proj_buffer.d_delta_var);
    reshape_heads_to_proj_kernel<<<blocks_for((int)comp), THREADS>>>(
        attn_delta_states.d_delta_mu_k, attn_delta_states.d_delta_var_k,
        batch_size, num_heads, timestep, head_dim, dk_proj_buffer.d_delta_mu,
        dk_proj_buffer.d_delta_var);
    reshape_heads_to_proj_kernel<<<blocks_for((int)comp), THREADS>>>(
        attn_delta_states.d_delta_mu_v, attn_delta_states.d_delta_var_v,
        batch_size, num_heads, timestep, head_dim, dv_proj_buffer.d_delta_mu,
        dv_proj_buffer.d_delta_var);

    BackwardStateCuda *cu_bwd =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());

    // State backward: sum delta_z from Q,K,V projections into output_delta.
    if (state_udapte) {
        // Q -> output_delta directly.
        DeltaStateCuda *p_q = &dq_proj_buffer;
        DeltaStateCuda *p_out = cu_out_delta;
        linear_state_backward_cuda(p_q, p_out, cu_bwd, d_mu_w_q,
                                   this->embed_dim, q_output_size, batch_seq);
        // K -> scratch buffer, then add to output_delta.
        ensure_delta_size(dz_kv_scratch, input_flat, batch_seq);
        DeltaStateCuda *p_k = &dk_proj_buffer;
        DeltaStateCuda *p_scratch = &dz_kv_scratch;
        linear_state_backward_cuda(p_k, p_scratch, cu_bwd, d_mu_w_k,
                                   this->embed_dim, k_output_size, batch_seq);
        add_delta_inplace_kernel<<<blocks_for((int)input_flat), THREADS>>>(
            dz_kv_scratch.d_delta_mu, dz_kv_scratch.d_delta_var,
            (int)input_flat, cu_out_delta->d_delta_mu,
            cu_out_delta->d_delta_var);
        // V -> scratch buffer, then add to output_delta.
        DeltaStateCuda *p_v = &dv_proj_buffer;
        linear_state_backward_cuda(p_v, p_scratch, cu_bwd, d_mu_w_v,
                                   this->embed_dim, v_output_size, batch_seq);
        add_delta_inplace_kernel<<<blocks_for((int)input_flat), THREADS>>>(
            dz_kv_scratch.d_delta_mu, dz_kv_scratch.d_delta_var,
            (int)input_flat, cu_out_delta->d_delta_mu,
            cu_out_delta->d_delta_var);
        cu_out_delta->seq_len = timestep;
    }

    if (this->param_update) {
        DeltaStateCuda *p_q = &dq_proj_buffer;
        DeltaStateCuda *p_k = &dk_proj_buffer;
        DeltaStateCuda *p_v = &dv_proj_buffer;
        DeltaStateCuda *p_out = cu_out_delta;
        linear_weight_backward_cuda(p_q, p_out, cu_bwd, d_var_w_q,
                                    this->embed_dim, q_output_size, batch_seq,
                                    d_delta_mu_w_q, d_delta_var_w_q);
        linear_weight_backward_cuda(p_k, p_out, cu_bwd, d_var_w_k,
                                    this->embed_dim, k_output_size, batch_seq,
                                    d_delta_mu_w_k, d_delta_var_w_k);
        linear_weight_backward_cuda(p_v, p_out, cu_bwd, d_var_w_v,
                                    this->embed_dim, v_output_size, batch_seq,
                                    d_delta_mu_w_v, d_delta_var_w_v);

        if (this->bias) {
            constexpr unsigned int B_THREADS = 16;
            dim3 block_b(B_THREADS, B_THREADS);
            unsigned int blk_q = (q_output_size + B_THREADS - 1) / B_THREADS;
            unsigned int blk_k = (k_output_size + B_THREADS - 1) / B_THREADS;
            unsigned int blk_v = (v_output_size + B_THREADS - 1) / B_THREADS;
            dim3 grid_q(1, blk_q), grid_k(1, blk_k), grid_v(1, blk_v);
            linear_bwd_delta_b<<<grid_q, block_b>>>(
                d_var_b_q, dq_proj_buffer.d_delta_mu,
                dq_proj_buffer.d_delta_var, this->embed_dim, q_output_size,
                batch_seq, d_delta_mu_b_q, d_delta_var_b_q);
            linear_bwd_delta_b<<<grid_k, block_b>>>(
                d_var_b_k, dk_proj_buffer.d_delta_mu,
                dk_proj_buffer.d_delta_var, this->embed_dim, k_output_size,
                batch_seq, d_delta_mu_b_k, d_delta_var_b_k);
            linear_bwd_delta_b<<<grid_v, block_b>>>(
                d_var_b_v, dv_proj_buffer.d_delta_mu,
                dv_proj_buffer.d_delta_var, this->embed_dim, v_output_size,
                batch_seq, d_delta_mu_b_v, d_delta_var_b_v);
        }
    }
    CHECK_LAST_CUDA_ERROR();
}

void MultiheadAttentionV2Cuda::copy_v2_params_from(
    const MultiheadAttentionV2 &source) {
    mu_w_q = source.mu_w_q;
    var_w_q = source.var_w_q;
    mu_w_k = source.mu_w_k;
    var_w_k = source.var_w_k;
    mu_w_v = source.mu_w_v;
    var_w_v = source.var_w_v;
    mu_b_q = source.mu_b_q;
    var_b_q = source.var_b_q;
    mu_b_k = source.mu_b_k;
    var_b_k = source.var_b_k;
    mu_b_v = source.mu_b_v;
    var_b_v = source.var_b_v;

    cudaSetDevice(this->device_idx);
    realloc_pair(&d_mu_w_q, &d_var_w_q, num_weights_q);
    realloc_pair(&d_mu_w_k, &d_var_w_k, num_weights_k);
    realloc_pair(&d_mu_w_v, &d_var_w_v, num_weights_v);
    if (this->bias) {
        realloc_pair(&d_mu_b_q, &d_var_b_q, num_biases_q);
        realloc_pair(&d_mu_b_k, &d_var_b_k, num_biases_k);
        realloc_pair(&d_mu_b_v, &d_var_b_v, num_biases_v);
    }
    this->params_to_device();
}

std::unique_ptr<BaseLayer> MultiheadAttentionV2Cuda::to_host() {
    auto host = std::make_unique<MultiheadAttentionV2>(
        this->embed_dim, this->num_heads, this->num_kv_heads, this->seq_len,
        this->bias, this->gain_w, this->gain_b, this->init_method,
        this->pos_emb, this->rope_theta, this->max_seq_len,
        this->use_causal_mask);
    this->params_to_host();
    host->mu_w_q = this->mu_w_q;
    host->var_w_q = this->var_w_q;
    host->mu_w_k = this->mu_w_k;
    host->var_w_k = this->var_w_k;
    host->mu_w_v = this->mu_w_v;
    host->var_w_v = this->var_w_v;
    host->mu_b_q = this->mu_b_q;
    host->var_b_q = this->var_b_q;
    host->mu_b_k = this->mu_b_k;
    host->var_b_k = this->var_b_k;
    host->mu_b_v = this->mu_b_v;
    host->var_b_v = this->var_b_v;
    return host;
}

ParameterMap MultiheadAttentionV2Cuda::get_parameters_as_map(
    std::string suffix) {
    std::string key = this->get_layer_name();
    if (!suffix.empty()) {
        key += "." + suffix;
    }

    this->params_to_host();

    auto concat3 = [](const std::vector<float> &a, const std::vector<float> &b,
                      const std::vector<float> &c) {
        std::vector<float> r;
        r.reserve(a.size() + b.size() + c.size());
        r.insert(r.end(), a.begin(), a.end());
        r.insert(r.end(), b.begin(), b.end());
        r.insert(r.end(), c.begin(), c.end());
        return r;
    };

    std::vector<float> mu_b_all, var_b_all;
    if (this->bias) {
        mu_b_all = concat3(mu_b_q, mu_b_k, mu_b_v);
        var_b_all = concat3(var_b_q, var_b_k, var_b_v);
    }

    ParameterTuple parameters = std::make_tuple(
        concat3(mu_w_q, mu_w_k, mu_w_v), concat3(var_w_q, var_w_k, var_w_v),
        std::move(mu_b_all), std::move(var_b_all));

    return {{key, parameters}};
}

void MultiheadAttentionV2Cuda::load_parameters_from_map(
    const ParameterMap &param_map, const std::string &suffix) {
    std::string key = this->get_layer_name();
    if (!suffix.empty()) {
        key += "." + suffix;
    }

    auto it = param_map.find(key);
    if (it == param_map.end()) {
        LOG(LogLevel::ERROR, "Key " + key + " not found in parameter map.");
        return;
    }

    const auto &params = it->second;
    auto split3 = [](const std::vector<float> &src, std::vector<float> &q,
                     std::vector<float> &k, std::vector<float> &v, size_t nq,
                     size_t nk, size_t nv) {
        q.assign(src.begin(), src.begin() + nq);
        k.assign(src.begin() + nq, src.begin() + nq + nk);
        v.assign(src.begin() + nq + nk, src.begin() + nq + nk + nv);
    };

    split3(std::get<0>(params), mu_w_q, mu_w_k, mu_w_v, num_weights_q,
           num_weights_k, num_weights_v);
    split3(std::get<1>(params), var_w_q, var_w_k, var_w_v, num_weights_q,
           num_weights_k, num_weights_v);
    if (this->bias) {
        split3(std::get<2>(params), mu_b_q, mu_b_k, mu_b_v, num_biases_q,
               num_biases_k, num_biases_v);
        split3(std::get<3>(params), var_b_q, var_b_k, var_b_v, num_biases_q,
               num_biases_k, num_biases_v);
    }

    this->params_to_device();
}

AttentionScores MultiheadAttentionV2Cuda::get_attention_scores() {
    AttentionScores s;
    s.batch_size = attn_states.allocated_batch_size;
    s.num_heads = attn_states.allocated_num_heads;
    s.timestep = attn_states.allocated_timestep;
    if (s.batch_size <= 0 || s.num_heads <= 0 || s.timestep <= 0) return s;
    size_t total = (size_t)s.batch_size * s.num_heads * s.timestep * s.timestep;
    s.mu.resize(total);
    s.var.resize(total);
    cudaMemcpy(s.mu.data(), attn_states.d_mu_att_score, total * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(s.var.data(), attn_states.d_var_att_score, total * sizeof(float),
               cudaMemcpyDeviceToHost);
    return s;
}
