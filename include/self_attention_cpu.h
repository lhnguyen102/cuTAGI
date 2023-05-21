///////////////////////////////////////////////////////////////////////////////
// File:         self_attention_cpu.h
// Description:  Header of CPU version for self attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 13, 2023
// Updated:      May 21, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
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