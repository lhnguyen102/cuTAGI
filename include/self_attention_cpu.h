#pragma once
#include <cmath>
#include <iostream>
#include <vector>

#include "activation_fun_cpu.h"
#include "data_transfer_cpu.h"
#include "fc_layer_cpu.h"
#include "struct_var.h"

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

void self_attention_forward_cpu(Network &net_prop, NetState &state,
                                Param &theta, int l);

void update_self_attention_state(Network &net_prop, NetState &state,
                                 Param &theta, DeltaState &d_state, int k);

void update_self_attention_state(Network &net_prop, NetState &state,
                                 Param &theta, DeltaState &d_state, int k);

void update_self_attention_param(Network &net_prop, Param &theta,
                                 NetState &state, DeltaState &d_state,
                                 DeltaParam &d_theta, int k_layer);