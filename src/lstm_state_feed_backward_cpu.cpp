///////////////////////////////////////////////////////////////////////////////
// File:         lstm_state_feed_backward_cpu.cpp
// Description:  Long-Short Term Memory (LSTM) state backward pass in TAGI
//               (cpu version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2022
// Updated:      August 07, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#include "../include/lstm_state_feed_backward_cpu.h"

void lstm_delta_mean_var(std::vector<float> &Sz, std::vector<float> &mw,
                         std::vector<float> &Jf_ga, std::vector<float> &mi_f,
                         std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
                         std::vector<float> &Jc_ga, std::vector<float> &mo_ga,
                         std::vector<float> &Jo_ga, std::vector<float> &mc_prev,
                         std::vector<float> &Jc, std::vector<float> &delta_m,
                         std::vector<float> &delta_S, int w_pos_f, int w_pos_i,
                         int w_pos_c, int w_pos_o, int no, int ni, int n_seq,
                         int B, std::vector<float> &delta_mz,
                         std::vector<float> &delta_Sz)
/*Compute the updated quatitites of the mean of the hidden states for lstm
   layer*/
{
    // TO BE COMPLETED
}