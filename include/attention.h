#pragma once
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base_layer.h"
#include "data_struct.h"
#include "param_init.h"

struct AttentionScores {
    int batch_size = 0;
    int num_heads = 0;
    int timestep = 0;
    std::vector<float> mu;
    std::vector<float> var;
};

struct AttentionStates {
    std::vector<float> mu_in_proj, var_in_proj;
    std::vector<float> mu_q, var_q, mu_k, var_k, mu_v, var_v;
    std::vector<float> mu_q_pe, var_q_pe, mu_k_pe, var_k_pe;
    std::vector<float> mu_qk, var_qk;
    std::vector<float> mu_mqk, var_mqk, j_mqk;
    std::vector<float> mu_att_score, var_att_score;
    std::vector<float> mu_sv, var_sv;

    void set_size(int batch_size, int num_heads, int timestep, int head_size);
};

struct AttentionDeltaStates {
    std::vector<float> delta_mu_buffer, delta_var_buffer;
    std::vector<float> delta_mu_v, delta_var_v;
    std::vector<float> delta_mu_att_score, delta_var_att_score;
    std::vector<float> delta_mu_q, delta_var_q;
    std::vector<float> delta_mu_k, delta_var_k;
    std::vector<float> delta_mu_q_pe, delta_var_q_pe;
    std::vector<float> delta_mu_k_pe, delta_var_k_pe;
    std::vector<float> delta_mu_in_proj, delta_var_in_proj;

    void set_size(int batch_size, int num_heads, int timestep, int head_size);
};

// Diagnostic helpers shared between CPU and CUDA forward paths.
void print_magnitude_stats(const char *name, const std::vector<float> &mu,
                           const std::vector<float> &var);

void print_magnitude_stats_causal(const char *name,
                                  const std::vector<float> &mu,
                                  const std::vector<float> &var, int batch_size,
                                  int num_heads, int timestep);

void separate_input_projection_components(
    std::vector<float> &mu_embs, std::vector<float> &var_embs, int batch_size,
    int num_heads, int timestep, int head_size, std::vector<float> &mu_q,
    std::vector<float> &var_q, std::vector<float> &mu_k,
    std::vector<float> &var_k, std::vector<float> &mu_v,
    std::vector<float> &var_v);

void cat_intput_projection_components(
    std::vector<float> &mu_q, std::vector<float> &var_q,
    std::vector<float> &mu_k, std::vector<float> &var_k,
    std::vector<float> &mu_v, std::vector<float> &var_v, int batch_size,
    int num_heads, int timestep, int head_size, std::vector<float> &mu_embs,
    std::vector<float> &var_embs);

void query_key(std::vector<float> &mu_q, std::vector<float> &var_q,
               std::vector<float> &mu_k, std::vector<float> &var_k,
               int batch_size, int num_heads, int timestep, int head_size,
               std::vector<float> &mu_qk, std::vector<float> &var_qk);

void tagi_4d_matrix_mul(std::vector<float> &mu_a, std::vector<float> &var_a,
                        std::vector<float> &mu_b, std::vector<float> &var_b,
                        int N, int C, int H, int W, int D,
                        std::vector<float> &mu_ab, std::vector<float> &var_ab);

void project_output_forward(std::vector<float> &mu_in,
                            std::vector<float> &var_in, int batch_size,
                            int num_heads, int timestep, int head_size,
                            std::vector<float> &mu_out,
                            std::vector<float> &var_out);

void project_output_backward(std::vector<float> &mu_in,
                             std::vector<float> &var_in, int batch_size,
                             int num_heads, int timestep, int head_size,
                             std::vector<float> &mu_out,
                             std::vector<float> &var_out);

void mha_delta_score(std::vector<float> &mu_v, std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int batch_size,
                     int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_s,
                     std::vector<float> &delta_var_s);

void mha_delta_value(std::vector<float> &mu_s, std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int batch_size,
                     int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_v,
                     std::vector<float> &delta_var_v);

void mha_delta_query(std::vector<float> &var_q, std::vector<float> &mu_k,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, std::vector<float> &jcb_mqk,
                     int batch_size, int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_q,
                     std::vector<float> &delta_var_q);

void mha_delta_key(std::vector<float> &var_k, std::vector<float> &mu_q,
                   std::vector<float> &delta_mu, std::vector<float> &delta_var,
                   std::vector<float> &jcb_mqk, int batch_size, int num_heads,
                   int timestep, int head_size, std::vector<float> &delta_mu_k,
                   std::vector<float> &delta_var_k);

void generate_rope_cache(int max_seq_len, int head_dim, float theta,
                         std::vector<float> &cos_cache,
                         std::vector<float> &sin_cache);

void generate_sinusoidal_pe_cache(int max_seq_len, int head_dim,
                                  std::vector<float> &pe_cache);

void apply_positional_encoding(std::vector<float> &mu_in,
                               std::vector<float> &var_in,
                               std::vector<float> &pe_cache, int batch_size,
                               int num_heads, int timestep, int head_dim,
                               std::vector<float> &mu_out,
                               std::vector<float> &var_out);

void apply_rope(std::vector<float> &mu_in, std::vector<float> &var_in,
                std::vector<float> &cos_cache, std::vector<float> &sin_cache,
                int batch_size, int num_heads, int timestep, int head_dim,
                std::vector<float> &mu_out, std::vector<float> &var_out);

void rope_backward(std::vector<float> &delta_mu_in,
                   std::vector<float> &delta_var_in,
                   std::vector<float> &cos_cache, std::vector<float> &sin_cache,
                   int batch_size, int num_heads, int timestep, int head_dim,
                   std::vector<float> &delta_mu_out,
                   std::vector<float> &delta_var_out);

class Remax;
class Softmax;

class MultiheadAttention : public BaseLayer {
   public:
    size_t num_heads;
    size_t num_kv_heads;
    size_t embed_dim;
    float gain_w;
    float gain_b;
    std::string init_method;
    size_t head_dim;
    size_t num_reps;
    AttentionStates attn_states;
    AttentionDeltaStates attn_delta_states;

    std::unique_ptr<Remax> remax_layer;      // [REMAX]
    std::unique_ptr<Softmax> softmax_layer;  // [SOFTMAX]
    BaseHiddenStates remax_input;
    BaseHiddenStates remax_output;
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
    std::vector<float> pe_cache;

    MultiheadAttention(size_t embed_dim, size_t num_heads, size_t num_kv_heads,
                       size_t seq_len_ = 1, bool bias = true,
                       float gain_w = 1.0f, float gain_b = 1.0f,
                       std::string init_method = "Xavier",
                       std::string pos_emb = "rope",
                       float rope_theta = 10000.0f, size_t max_seq_len = 2048,
                       bool use_causal_mask = true, int device_idx = 0);

    ~MultiheadAttention();

    MultiheadAttention(const MultiheadAttention &) = delete;
    MultiheadAttention &operator=(const MultiheadAttention &) = delete;

    MultiheadAttention(MultiheadAttention &&) = default;
    MultiheadAttention &operator=(MultiheadAttention &&) = default;

    virtual std::string get_layer_info() const override;

    virtual std::string get_layer_name() const override;

    virtual LayerType get_layer_type() const override;

    void init_weight_bias() override;

    virtual void forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states) override;

    virtual void backward(BaseDeltaStates &input_delta_states,
                          BaseDeltaStates &output_delta_states,
                          BaseTempStates &temp_states,
                          bool state_udapte = true) override;

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif

    AttentionScores get_attention_scores();
};

class MultiheadAttentionV2 : public BaseLayer {
   public:
    size_t num_heads;
    size_t num_kv_heads;
    size_t embed_dim;
    float gain_w;
    float gain_b;
    std::string init_method;
    size_t head_dim;
    size_t num_reps;
    AttentionStates attn_states;
    AttentionDeltaStates attn_delta_states;

    // Separate Q, K, V projection weights
    std::vector<float> mu_w_q, var_w_q, mu_w_k, var_w_k, mu_w_v, var_w_v;
    std::vector<float> delta_mu_w_q, delta_var_w_q;
    std::vector<float> delta_mu_w_k, delta_var_w_k;
    std::vector<float> delta_mu_w_v, delta_var_w_v;
    // Separate biases
    std::vector<float> mu_b_q, var_b_q, mu_b_k, var_b_k, mu_b_v, var_b_v;
    std::vector<float> delta_mu_b_q, delta_var_b_q;
    std::vector<float> delta_mu_b_k, delta_var_b_k;
    std::vector<float> delta_mu_b_v, delta_var_b_v;

    size_t num_weights_q, num_weights_k, num_weights_v;
    size_t num_biases_q, num_biases_k, num_biases_v;
    size_t q_output_size, k_output_size, v_output_size;

    std::unique_ptr<Remax> remax_layer;
    BaseHiddenStates remax_input;
    BaseHiddenStates remax_output;
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

    // Buffers for reshaping Q/K/V linear outputs
    std::vector<float> mu_q_proj, var_q_proj;
    std::vector<float> mu_k_proj, var_k_proj;
    std::vector<float> mu_v_proj, var_v_proj;

    MultiheadAttentionV2(size_t embed_dim, size_t num_heads,
                         size_t num_kv_heads, size_t seq_len_ = 1,
                         bool bias = true, float gain_w = 1.0f,
                         float gain_b = 1.0f,
                         std::string init_method = "Xavier",
                         std::string pos_emb = "rope",
                         float rope_theta = 10000.0f, size_t max_seq_len = 2048,
                         bool use_causal_mask = true, int device_idx = 0);

    ~MultiheadAttentionV2();

    MultiheadAttentionV2(const MultiheadAttentionV2 &) = delete;
    MultiheadAttentionV2 &operator=(const MultiheadAttentionV2 &) = delete;

    MultiheadAttentionV2(MultiheadAttentionV2 &&) = default;
    MultiheadAttentionV2 &operator=(MultiheadAttentionV2 &&) = default;

    std::string get_layer_info() const override;
    std::string get_layer_name() const override;
    LayerType get_layer_type() const override;

    void init_weight_bias() override;
    void allocate_param_delta() override;
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

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif

    AttentionScores get_attention_scores();
};
