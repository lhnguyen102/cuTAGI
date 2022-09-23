///////////////////////////////////////////////////////////////////////////////
// File:         test_lstm_cpu.cpp
// Description:  Test LSTM layer
//               (cpu version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 04, 2022
// Updated:      September 04, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "test_lstm_cpu.h"

void generate_random_number(float &min_v, float &max_v, std::vector<float> &v) {
    for (int i = 0; i < v.size(); i++) {
        v[i] = min_v + static_cast<float>(rand()) /
                           (static_cast<float>(RAND_MAX / (max_v - min_v)));
    }
}

void test_delta_mean_var_z() {
    // Seed
    srand(10);

    // Data
    int ni = 3;
    int no = 4;
    int seq_len = 3;
    int B = 1;
    float max_v = 1.0f;
    float min_v = -1.0f;

    // Initialization
    int z_pos_i = 0;
    int z_pos_o = 0;
    int z_pos_o_lstm = 0;
    int w_pos_f = 0;
    int w_pos_i = 0;
    int w_pos_c = 0;
    int w_pos_o = 0;
    std::vector<float> mw((no + ni) * no, 0);
    std::vector<float> Sz((no + ni) * seq_len * B, 0);
    std::vector<float> Jf_ga(no * seq_len * B, 0);
    std::vector<float> mi_ga(no * seq_len * B, 0);
    std::vector<float> Ji_ga(no * seq_len * B, 0);
    std::vector<float> mc_ga(no * seq_len * B, 0);
    std::vector<float> Jc_ga(no * seq_len * B, 0);
    std::vector<float> mo_ga(no * seq_len * B, 0);
    std::vector<float> Jo_ga(no * seq_len * B, 0);
    std::vector<float> mc_prev(no * seq_len * B, 0);
    std::vector<float> mca(no * seq_len * B, 0);
    std::vector<float> Jca(no * seq_len * B, 0);
    std::vector<float> delta_m(no * seq_len * B, 0);
    std::vector<float> delta_S(no * seq_len * B, 0);
    std::vector<float> delta_mz(ni * seq_len * B, 0);
    std::vector<float> delta_Sz(ni * seq_len * B, 0);

    // Generate data
    generate_random_number(min_v, max_v, mw);
    generate_random_number(min_v, max_v, Sz);
    generate_random_number(min_v, max_v, Jf_ga);
    generate_random_number(min_v, max_v, mi_ga);
    generate_random_number(min_v, max_v, Ji_ga);
    generate_random_number(min_v, max_v, mc_ga);
    generate_random_number(min_v, max_v, Jc_ga);
    generate_random_number(min_v, max_v, mo_ga);
    generate_random_number(min_v, max_v, Jo_ga);
    generate_random_number(min_v, max_v, mc_prev);
    generate_random_number(min_v, max_v, mca);
    generate_random_number(min_v, max_v, Jca);
    generate_random_number(min_v, max_v, delta_m);
    generate_random_number(min_v, max_v, delta_S);

    // Inovation vector
    lstm_delta_mean_var_z(Sz, mw, Jf_ga, mi_ga, Ji_ga, mc_ga, Jc_ga, mo_ga,
                          Jo_ga, mc_prev, mca, Jca, delta_m, delta_S, z_pos_i,
                          z_pos_o, z_pos_o_lstm, w_pos_f, w_pos_i, w_pos_c,
                          w_pos_o, no, ni, seq_len, B, delta_mz, delta_Sz);
    int check = 0;
}

int test_lstm_cpu() {
    test_delta_mean_var_z();
    return 0;
}