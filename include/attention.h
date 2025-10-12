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

class AttentionStates {
   public:
    std::vector<float> mu_in_proj, var_in_proj;
    std::vector<float> mu_q, var_q, mu_k, var_k, mu_v, var_v;
    std::vector<float> mu_qk, var_qk, J_qk;
    std::vector<float> mu_mqk, var_mqk, J_mqk;
    std::vector<float> mu_att_score, var_att_score;
    std::vector<float> mu_sv, var_sv;
    std::vector<float> mu_out_proj, var_out_proj, J_out_proj;

    AttentionStates();
    ~AttentionStates() = default;
    void set_size(int batch_size, int num_heads, int timestep, int head_size);
};

class AttentionDeltaStates {
   public:
    std::vector<float> delta_mu_buffer, delta_var_buffer;
    std::vector<float> delta_mu_out_proj, delta_var_out_proj;
    std::vector<float> delta_mu_v, delta_var_v;
    std::vector<float> delta_mu_att_score, delta_var_att_score;
    std::vector<float> delta_mu_r, delta_var_r;
    std::vector<float> delta_mu_q, delta_var_q;
    std::vector<float> delta_mu_k, delta_var_k;
    std::vector<float> delta_mu_in_proj, delta_var_in_proj;

    AttentionDeltaStates();
    ~AttentionDeltaStates() = default;
    void set_size(int batch_size, int num_heads, int timestep, int head_size);
};

void separate_input_projection_components(
    std::vector<float> &mu_embs, std::vector<float> &var_embs, int emb_pos,
    int qkv_pos, int batch_size, int num_heads, int timestep, int head_size,
    std::vector<float> &mu_q, std::vector<float> &var_q,
    std::vector<float> &mu_k, std::vector<float> &var_k,
    std::vector<float> &mu_v, std::vector<float> &var_v);

void cat_intput_projection_components(
    std::vector<float> &mu_q, std::vector<float> &var_q,
    std::vector<float> &mu_k, std::vector<float> &var_k,
    std::vector<float> &mu_v, std::vector<float> &var_v, int qkv_pos,
    int emb_pos, int batch_size, int num_heads, int timestep, int head_size,
    std::vector<float> &mu_embs, std::vector<float> &var_embs);

void query_key(std::vector<float> &mu_q, std::vector<float> &var_q,
               std::vector<float> &mu_k, std::vector<float> &var_k, int qkv_pos,
               int batch_size, int num_heads, int timestep, int head_size,
               std::vector<float> &mu_qk, std::vector<float> &var_qk);

void mask_query_key(std::vector<float> &mu_qk, std::vector<float> &var_qk,
                    int batch_size, int num_heads, int timestep, int head_size,
                    std::vector<float> &mu_mqk, std::vector<float> &var_mqk);

void tagi_4d_matrix_mul(std::vector<float> &mu_a, std::vector<float> &var_a,
                        std::vector<float> &mu_b, std::vector<float> &var_b,
                        int a_pos, int b_pos, int ab_pos, int N, int C, int H,
                        int W, int D, std::vector<float> &mu_ab,
                        std::vector<float> &var_ab);

void project_output_forward(std::vector<float> &mu_in,
                            std::vector<float> &var_in, int in_pos, int out_pos,
                            int batch_size, int num_heads, int timestep,
                            int head_size, std::vector<float> &mu_out,
                            std::vector<float> &var_out);

void project_output_backward(std::vector<float> &mu_in,
                             std::vector<float> &var_in, int in_pos,
                             int out_pos, int batch_size, int num_heads,
                             int timestep, int head_size,
                             std::vector<float> &mu_out,
                             std::vector<float> &var_out);

void mha_delta_score(std::vector<float> &mu_v, std::vector<float> &var_s,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int qkv_pos, int att_pos,
                     int batch_size, int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_s,
                     std::vector<float> &delta_var_s);

void mha_delta_value(std::vector<float> &mu_s, std::vector<float> &var_v,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int qkv_pos, int att_pos,
                     int batch_size, int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_v,
                     std::vector<float> &delta_var_v);

void mha_delta_query(std::vector<float> &var_q, std::vector<float> &mu_k,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int qkv_pos, int att_pos,
                     int batch_size, int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_q,
                     std::vector<float> &delta_var_q);

void mha_delta_key(std::vector<float> &var_k, std::vector<float> &mu_q,
                   std::vector<float> &delta_mu, std::vector<float> &delta_var,
                   int qkv_pos, int att_pos, int batch_size, int num_heads,
                   int timestep, int head_size, std::vector<float> &delta_mu_k,
                   std::vector<float> &delta_var_k);

class Remax;

class SelfAttention : public BaseLayer {
   public:
    size_t num_heads;
    size_t num_kv_heads;
    size_t embed_dim;
    float gain_w;
    float gain_b;
    std::string init_method;
    size_t timestep;
    size_t head_dim;
    size_t num_reps;
    AttentionStates attn_states;
    AttentionDeltaStates attn_delta_states;

    std::unique_ptr<Remax> remax_layer;
    BaseHiddenStates remax_input;
    BaseHiddenStates remax_output;
    BaseTempStates remax_temp;

    SelfAttention(size_t embed_dim, size_t num_heads, size_t num_kv_heads,
                  bool bias = true, float gain_w = 1.0f, float gain_b = 1.0f,
                  std::string init_method = "Xavier", int device_idx = 0);

    ~SelfAttention();

    SelfAttention(const SelfAttention &) = delete;
    SelfAttention &operator=(const SelfAttention &) = delete;

    SelfAttention(SelfAttention &&) = default;
    SelfAttention &operator=(SelfAttention &&) = default;

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
};
