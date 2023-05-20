///////////////////////////////////////////////////////////////////////////////
// File:         self_attention_cpu.h
// Description:  Header of CPU version for self attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 13, 2023
// Updated:      May 20, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
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