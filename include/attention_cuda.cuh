#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "activation_cuda.cuh"
#include "attention.h"
#include "base_layer.h"
#include "base_layer_cuda.cuh"
#include "data_struct.h"
#include "data_struct_cuda.cuh"

// Device mirror of AttentionStates. The masked scores are not stored: the mask
// is applied in place on the Remax input buffer. RoPE buffers are allocated
// only when use_rope is set.
struct AttentionStatesCuda {
    int allocated_batch_size = 0, allocated_num_heads = 0;
    int allocated_timestep = 0, allocated_head_dim = 0;
    bool allocated_rope = false;

    float *d_mu_q = nullptr, *d_var_q = nullptr;
    float *d_mu_k = nullptr, *d_var_k = nullptr;
    float *d_mu_v = nullptr, *d_var_v = nullptr;
    float *d_mu_q_rope = nullptr, *d_var_q_rope = nullptr;
    float *d_mu_k_rope = nullptr, *d_var_k_rope = nullptr;
    float *d_mu_qk = nullptr, *d_var_qk = nullptr;
    float *d_mu_att_score = nullptr, *d_var_att_score = nullptr;
    float *d_j_mqk = nullptr;
    float *d_mu_sv = nullptr, *d_var_sv = nullptr;

    AttentionStatesCuda() = default;
    ~AttentionStatesCuda();
    AttentionStatesCuda(const AttentionStatesCuda &) = delete;
    AttentionStatesCuda &operator=(const AttentionStatesCuda &) = delete;

    void set_size(int batch_size, int num_heads, int timestep, int head_dim,
                  bool use_rope);
    void deallocate();
};

// Device mirror of AttentionDeltaStates.
struct AttentionDeltaStatesCuda {
    int allocated_batch_size = 0, allocated_num_heads = 0;
    int allocated_timestep = 0, allocated_head_dim = 0;
    bool allocated_rope = false;

    float *d_delta_mu_buffer = nullptr, *d_delta_var_buffer = nullptr;
    float *d_delta_mu_v = nullptr, *d_delta_var_v = nullptr;
    float *d_delta_mu_att_score = nullptr, *d_delta_var_att_score = nullptr;
    float *d_delta_mu_q = nullptr, *d_delta_var_q = nullptr;
    float *d_delta_mu_k = nullptr, *d_delta_var_k = nullptr;
    float *d_delta_mu_q_rope = nullptr, *d_delta_var_q_rope = nullptr;
    float *d_delta_mu_k_rope = nullptr, *d_delta_var_k_rope = nullptr;

    AttentionDeltaStatesCuda() = default;
    ~AttentionDeltaStatesCuda();
    AttentionDeltaStatesCuda(const AttentionDeltaStatesCuda &) = delete;
    AttentionDeltaStatesCuda &operator=(const AttentionDeltaStatesCuda &) =
        delete;

    void set_size(int batch_size, int num_heads, int timestep, int head_dim,
                  bool use_rope);
    void deallocate();
};

class MultiheadAttentionCuda : public BaseLayerCuda {
   public:
    size_t num_heads;
    size_t num_kv_heads;
    size_t embed_dim;
    float gain_w;
    float gain_b;
    std::string init_method;
    size_t head_dim;

    AttentionStatesCuda attn_states;
    AttentionDeltaStatesCuda attn_delta_states;

    // Fused QKV projection output and its delta.
    HiddenStateCuda in_proj_buffer;
    DeltaStateCuda d_in_proj_buffer;

    std::unique_ptr<RemaxCuda> remax_layer;
    HiddenStateCuda remax_input;
    HiddenStateCuda remax_output;
    BaseTempStates remax_temp;

    bool use_rope;
    float rope_theta;
    size_t max_seq_len;
    bool use_causal_mask;
    bool center_score_delta;
    float *d_cos_cache = nullptr, *d_sin_cache = nullptr;

    MultiheadAttentionCuda(size_t embed_dim, size_t num_heads,
                           size_t num_kv_heads, size_t seq_len_ = 1,
                           bool bias = true, float gain_w = 1.0f,
                           float gain_b = 1.0f,
                           std::string init_method = "Xavier",
                           bool use_rope = true, float rope_theta = 10000.0f,
                           size_t max_seq_len = 2048,
                           bool use_causal_mask = true,
                           bool center_score_delta = false, int device_idx = 0);

    ~MultiheadAttentionCuda();

    MultiheadAttentionCuda(const MultiheadAttentionCuda &) = delete;
    MultiheadAttentionCuda &operator=(const MultiheadAttentionCuda &) = delete;

    std::string get_layer_info() const override;
    std::string get_layer_name() const override;
    LayerType get_layer_type() const override;

    void init_weight_bias() override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    std::unique_ptr<BaseLayer> to_host() override;

    AttentionScores get_attention_scores() const;
};
