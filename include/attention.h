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

struct AttentionStates {
    std::vector<float> mu_in_proj, var_in_proj;
    std::vector<float> mu_q, var_q, mu_k, var_k, mu_v, var_v;
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
    std::vector<float> delta_mu_in_proj, delta_var_in_proj;

    void set_size(int batch_size, int num_heads, int timestep, int head_size);
};

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

void mask_query_key(std::vector<float> &mu_qk, std::vector<float> &var_qk,
                    int batch_size, int num_heads, int timestep, int head_size,
                    std::vector<float> &mu_mqk, std::vector<float> &var_mqk);

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

void mha_delta_score(std::vector<float> &mu_v, std::vector<float> &var_s,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int batch_size,
                     int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_s,
                     std::vector<float> &delta_var_s);

void mha_delta_value(std::vector<float> &mu_s, std::vector<float> &var_v,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var, int batch_size,
                     int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_v,
                     std::vector<float> &delta_var_v);

void mha_delta_query(std::vector<float> &var_q, std::vector<float> &mu_k,
                     std::vector<float> &delta_mu,
                     std::vector<float> &delta_var,
                     std::vector<float> &jcb_att_score, int batch_size,
                     int num_heads, int timestep, int head_size,
                     std::vector<float> &delta_mu_q,
                     std::vector<float> &delta_var_q);

void mha_delta_key(std::vector<float> &var_k, std::vector<float> &mu_q,
                   std::vector<float> &delta_mu, std::vector<float> &delta_var,
                   std::vector<float> &jcb_att_score, int batch_size,
                   int num_heads, int timestep, int head_size,
                   std::vector<float> &delta_mu_k,
                   std::vector<float> &delta_var_k);

class Remax;

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

    std::unique_ptr<Remax> remax_layer;
    BaseHiddenStates remax_input;
    BaseHiddenStates remax_output;
    BaseTempStates remax_temp;
    size_t seq_len = 1;

    MultiheadAttention(size_t embed_dim, size_t num_heads, size_t num_kv_heads,
                       bool bias = true, float gain_w = 1.0f,
                       float gain_b = 1.0f, std::string init_method = "Xavier",
                       int device_idx = 0);

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
};
