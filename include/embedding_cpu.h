///////////////////////////////////////////////////////////////////////////////
// File:         embedding_cpu.h
// Description:  Header file for embeddings layer
//               (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 22, 2023
// Updated:      August 23, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

struct EmbeddingProp {
    std::vector<int> num_categories, num_emb_sizes;
};
std::tuple<std::vector<float>, std::vector<float>> initialize_embedding_values(
    std::vector<int> num_classes, std::vector<int> num_weights, float scale,
    unsigned int *seed = nullptr);
void forward(std::vector<float> &ma, std::vector<float> &mu_w,
             std::vector<float> &var_w, std::vector<int> &cat_sizes,
             std::vector<int> &emb_sizes, int num_cat, int batch_size,
             int w_pos_in, int z_pos_in, int z_pos_out, int num_states,
             std::vector<float> &mu_z, std::vector<float> &var_z);
void param_backward(std::vector<float> &delta_mz, int z_pos_out,
                    int num_states);
int calculate_embedding_size(int num_categories);
