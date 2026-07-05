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

// Device mirror of AttentionStates.
// `set_size` (re)allocates only the buffers needed for the active path:
// `need_pe` is false when pos_emb != "rope".
// We never store the masked-QK matrix separately; the mask is applied in-place
// inside the remax_input buffer (see attention_cuda.cu).
// The `in_proj` buffer is held on the layer (not in this struct) since V2
// doesn't need it at all.
struct AttentionStatesCuda {
    int allocated_batch_size = 0, allocated_num_heads = 0;
    int allocated_timestep = 0, allocated_head_dim = 0;
    bool alloc_pe = false;

    float *d_mu_q = nullptr, *d_var_q = nullptr;
    float *d_mu_k = nullptr, *d_var_k = nullptr;
    float *d_mu_v = nullptr, *d_var_v = nullptr;
    float *d_mu_q_pe = nullptr, *d_var_q_pe = nullptr;
    float *d_mu_k_pe = nullptr, *d_var_k_pe = nullptr;
    float *d_mu_qk = nullptr, *d_var_qk = nullptr;
    float *d_mu_att_score = nullptr, *d_var_att_score = nullptr;
    float *d_j_mqk = nullptr;
    float *d_mu_sv = nullptr, *d_var_sv = nullptr;

    AttentionStatesCuda() = default;
    ~AttentionStatesCuda();
    AttentionStatesCuda(const AttentionStatesCuda &) = delete;
    AttentionStatesCuda &operator=(const AttentionStatesCuda &) = delete;

    void set_size(int batch_size, int num_heads, int timestep, int head_dim,
                  bool need_pe);
    void deallocate();
};

// Device mirror of AttentionDeltaStates.
struct AttentionDeltaStatesCuda {
    int allocated_batch_size = 0, allocated_num_heads = 0;
    int allocated_timestep = 0, allocated_head_dim = 0;
    bool alloc_pe = false;

    float *d_delta_mu_buffer = nullptr, *d_delta_var_buffer = nullptr;
    float *d_delta_mu_v = nullptr, *d_delta_var_v = nullptr;
    float *d_delta_mu_att_score = nullptr, *d_delta_var_att_score = nullptr;
    float *d_delta_mu_q = nullptr, *d_delta_var_q = nullptr;
    float *d_delta_mu_k = nullptr, *d_delta_var_k = nullptr;
    float *d_delta_mu_q_pe = nullptr, *d_delta_var_q_pe = nullptr;
    float *d_delta_mu_k_pe = nullptr, *d_delta_var_k_pe = nullptr;

    AttentionDeltaStatesCuda() = default;
    ~AttentionDeltaStatesCuda();
    AttentionDeltaStatesCuda(const AttentionDeltaStatesCuda &) = delete;
    AttentionDeltaStatesCuda &operator=(const AttentionDeltaStatesCuda &) =
        delete;

    void set_size(int batch_size, int num_heads, int timestep, int head_dim,
                  bool need_pe);
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
    size_t num_reps;

    AttentionStatesCuda attn_states;
    AttentionDeltaStatesCuda attn_delta_states;

    // QKV linear projection scratch (forward + backward).
    HiddenStateCuda in_proj_buffer;
    DeltaStateCuda d_in_proj_buffer;

    std::unique_ptr<RemaxCuda> remax_layer;
    HiddenStateCuda remax_input;
    HiddenStateCuda remax_output;
    BaseTempStates remax_temp;

    std::string pos_emb;
    float rope_theta;
    size_t max_seq_len;
    bool use_causal_mask;
    bool debug = false;
    int debug_interval = 1;
    int _debug_step = 0;
    bool center_score_delta = false;

    std::vector<float> cos_cache;
    std::vector<float> sin_cache;
    float *d_cos_cache = nullptr, *d_sin_cache = nullptr;

    MultiheadAttentionCuda(size_t embed_dim, size_t num_heads,
                           size_t num_kv_heads, size_t seq_len_ = 1,
                           bool bias = true, float gain_w = 1.0f,
                           float gain_b = 1.0f,
                           std::string init_method = "Xavier",
                           std::string pos_emb = "rope",
                           float rope_theta = 10000.0f,
                           size_t max_seq_len = 2048,
                           bool use_causal_mask = true, int device_idx = 0);

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

    AttentionScores get_attention_scores();
};

class MultiheadAttentionV2Cuda : public BaseLayerCuda {
   public:
    size_t num_heads;
    size_t num_kv_heads;
    size_t embed_dim;
    float gain_w;
    float gain_b;
    std::string init_method;
    size_t head_dim;
    size_t num_reps;

    // Separate Q, K, V parameter buffers (host + device).
    std::vector<float> mu_w_q, var_w_q, mu_w_k, var_w_k, mu_w_v, var_w_v;
    std::vector<float> mu_b_q, var_b_q, mu_b_k, var_b_k, mu_b_v, var_b_v;
    std::vector<float> delta_mu_w_q, delta_var_w_q;
    std::vector<float> delta_mu_w_k, delta_var_w_k;
    std::vector<float> delta_mu_w_v, delta_var_w_v;
    std::vector<float> delta_mu_b_q, delta_var_b_q;
    std::vector<float> delta_mu_b_k, delta_var_b_k;
    std::vector<float> delta_mu_b_v, delta_var_b_v;

    float *d_mu_w_q = nullptr, *d_var_w_q = nullptr;
    float *d_mu_w_k = nullptr, *d_var_w_k = nullptr;
    float *d_mu_w_v = nullptr, *d_var_w_v = nullptr;
    float *d_mu_b_q = nullptr, *d_var_b_q = nullptr;
    float *d_mu_b_k = nullptr, *d_var_b_k = nullptr;
    float *d_mu_b_v = nullptr, *d_var_b_v = nullptr;
    float *d_delta_mu_w_q = nullptr, *d_delta_var_w_q = nullptr;
    float *d_delta_mu_w_k = nullptr, *d_delta_var_w_k = nullptr;
    float *d_delta_mu_w_v = nullptr, *d_delta_var_w_v = nullptr;
    float *d_delta_mu_b_q = nullptr, *d_delta_var_b_q = nullptr;
    float *d_delta_mu_b_k = nullptr, *d_delta_var_b_k = nullptr;
    float *d_delta_mu_b_v = nullptr, *d_delta_var_b_v = nullptr;

    size_t num_weights_q = 0, num_weights_k = 0, num_weights_v = 0;
    size_t num_biases_q = 0, num_biases_k = 0, num_biases_v = 0;
    size_t q_output_size = 0, k_output_size = 0, v_output_size = 0;

    AttentionStatesCuda attn_states;
    AttentionDeltaStatesCuda attn_delta_states;

    // Persistent reshaped Q/K/V projection buffers (host + device).
    HiddenStateCuda q_proj_buffer;
    HiddenStateCuda k_proj_buffer;
    HiddenStateCuda v_proj_buffer;
    DeltaStateCuda dq_proj_buffer;
    DeltaStateCuda dk_proj_buffer;
    DeltaStateCuda dv_proj_buffer;
    DeltaStateCuda dz_kv_scratch;  // scratch DeltaStateCuda for K/V delta_z

    std::unique_ptr<RemaxCuda> remax_layer;
    HiddenStateCuda remax_input;
    HiddenStateCuda remax_output;
    BaseTempStates remax_temp;

    std::string pos_emb;
    float rope_theta;
    size_t max_seq_len;
    bool use_causal_mask;
    bool debug = false;
    int debug_interval = 1;
    int _debug_step = 0;

    std::vector<float> cos_cache;
    std::vector<float> sin_cache;
    float *d_cos_cache = nullptr, *d_sin_cache = nullptr;

    MultiheadAttentionV2Cuda(size_t embed_dim, size_t num_heads,
                             size_t num_kv_heads, size_t seq_len_ = 1,
                             bool bias = true, float gain_w = 1.0f,
                             float gain_b = 1.0f,
                             std::string init_method = "Xavier",
                             std::string pos_emb = "rope",
                             float rope_theta = 10000.0f,
                             size_t max_seq_len = 2048,
                             bool use_causal_mask = true, int device_idx = 0);

    ~MultiheadAttentionV2Cuda();

    MultiheadAttentionV2Cuda(const MultiheadAttentionV2Cuda &) = delete;
    MultiheadAttentionV2Cuda &operator=(const MultiheadAttentionV2Cuda &) =
        delete;

    std::string get_layer_info() const override;
    std::string get_layer_name() const override;
    LayerType get_layer_type() const override;

    void init_weight_bias() override;
    void allocate_param_delta() override;
    void params_to_device() override;
    void params_to_host() override;
    void update_weights() override;
    void update_biases() override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    ParameterMap get_parameters_as_map(std::string suffix = "") override;
    void load_parameters_from_map(const ParameterMap &param_map,
                                  const std::string &suffix) override;

    std::unique_ptr<BaseLayer> to_host() override;

    // Copy weights from a host MultiheadAttentionV2 into the CUDA layer.
    void copy_v2_params_from(const MultiheadAttentionV2 &source);

    AttentionScores get_attention_scores();
};
