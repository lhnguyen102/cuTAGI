#pragma once
#include <vector>

#include "struct_var.h"

struct GatedResidualProp {
    int input_size;
    int hidden_size;
    int output_size;
    int context_size;
    int num_states;
    int num_weights = 0;
    int num_biases = 0;
    bool residual = false;
    std::vector<int> w_pos = {0};
    std::vector<int> b_pos = {0};
    std::vector<int> z_pos = {0};
    std::vector<float> mu_z_1;
    std::vector<float> Sz_z_1;
    std::vector<float> ma_z_1;
    std::vector<float> Sa_z_1;
    std::vector<float> mu_z_2;
    std::vector<float> Sz_z_2;
    std::vector<float> ma_z_2;
    std::vector<float> Sa_z_2;
};
