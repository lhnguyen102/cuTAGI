#pragma once
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

struct EmbeddingProp {
    std::vector<size_t> cat_sizes, emb_sizes, num_bags, bag_sizes;
};

std::tuple<std::vector<float>, std::vector<float>> get_embedding_values(
    size_t num_classes, size_t emb_size, float scale,
    unsigned int *seed = nullptr);

std::tuple<std::vector<float>, std::vector<float>> initialize_embedding_values(
    std::vector<size_t> &cat_sizes, std::vector<size_t> &emb_sizes,
    int num_cat_var, float scale, unsigned int *seed = nullptr);

void bag_forward(std::vector<float> &mu_a, std::vector<float> &mu_w,
                 std::vector<float> &var_w, std::vector<size_t> &cat_sizes,
                 std::vector<size_t> &emb_sizes, std::vector<size_t> &num_bags,
                 std::vector<size_t> &bag_sizes, int num_cat, int batch_size,
                 int w_pos_in, int z_pos_in, int z_pos_out,
                 std::vector<float> &mu_z, std::vector<float> &var_z);

void param_backward(std::vector<float> &ma, std::vector<float> &var_w,
                    std::vector<float> &delta_mu, std::vector<float> &delta_var,
                    std::vector<size_t> &cat_sizes,
                    std::vector<size_t> &emb_sizes, int num_cat, int batch_size,
                    int z_pos_in, int z_pos_out, int w_pos_in,
                    std::vector<float> &delta_mu_w,
                    std::vector<float> &delta_var_w);

void bag_forward(std::vector<float> &mu_a, std::vector<float> &mu_w,
                 std::vector<float> &var_w, std::vector<size_t> &cat_sizes,
                 std::vector<size_t> &emb_sizes, std::vector<size_t> &num_bags,
                 std::vector<size_t> &bag_sizes, int num_cat, int batch_size,
                 int w_pos_in, int z_pos_in, int z_pos_out,
                 std::vector<float> &mu_z, std::vector<float> &var_z);

void bag_param_backward(
    std::vector<float> &mu_a, std::vector<float> &var_w,
    std::vector<float> &delta_mu, std::vector<float> &delta_var,
    std::vector<size_t> &cat_sizes, std::vector<size_t> &emb_sizes,
    std::vector<size_t> &num_bags, std::vector<size_t> &bag_sizes, int num_cat,
    int batch_size, int z_pos_in, int w_pos_in, int z_pos_out,
    std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w);

int calculate_embedding_size(int num_categories);
