///////////////////////////////////////////////////////////////////////////////
// File:         lstm_state_feed_backward_cpu.h
// Description:  Hearder file for Long-Short Term Memory (LSTM) backward pass
//               (cpu version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2022
// Updated:      September 16, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cmath>
#include <thread>
#include <vector>

#include "common.h"
#include "data_transfer_cpu.h"
#include "feed_forward_cpu.h"
#include "lstm_feed_forward_cpu.h"
#include "net_prop.h"
#include "struct_var.h"

void lstm_state_update_cpu(Network &net, NetState &state, Param &theta,
                           DeltaState &d_state, int l);

void lstm_parameter_update_cpu(Network &net, NetState &state, Param &theta,
                               DeltaState &d_state, DeltaParam &d_theta, int l);

void lstm_delta_mean_var_z(std::vector<float> &Sz, std::vector<float> &mw,
                           std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
                           std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
                           std::vector<float> &Jc_ga, std::vector<float> &mo_ga,
                           std::vector<float> &Jo_ga,
                           std::vector<float> &mc_prev, std::vector<float> &mca,
                           std::vector<float> &Jca, std::vector<float> &delta_m,
                           std::vector<float> &delta_S, int z_pos_i,
                           int z_pos_o, int z_pos_o_lstm, int w_pos_f,
                           int w_pos_i, int w_pos_c, int w_pos_o, int no,
                           int ni, int seq_len, int B,
                           std::vector<float> &delta_mz,
                           std::vector<float> &delta_Sz);

void lstm_delta_mean_var_w(std::vector<float> &Sw, std::vector<float> &mha,
                           std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
                           std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
                           std::vector<float> &Jc_ga, std::vector<float> &mo_ga,
                           std::vector<float> &Jo_ga,
                           std::vector<float> &mc_prev, std::vector<float> &mca,
                           std::vector<float> &Jc, std::vector<float> &delta_m,
                           std::vector<float> &delta_S, int z_pos_o,
                           int z_pos_o_lstm, int w_pos_f, int w_pos_i,
                           int w_pos_c, int w_pos_o, int no, int ni,
                           int seq_len, int B, std::vector<float> &delta_mw,
                           std::vector<float> &delta_Sw);

void lstm_delta_mean_var_b(std::vector<float> &Sb, std::vector<float> &Jf_ga,
                           std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
                           std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
                           std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
                           std::vector<float> &mc_prev, std::vector<float> &mca,
                           std::vector<float> &Jc, std::vector<float> &delta_m,
                           std::vector<float> &delta_S, int z_pos_o,
                           int z_pos_o_lstm, int b_pos_f, int b_pos_i,
                           int b_pos_c, int b_pos_o, int no, int seq_len, int B,
                           std::vector<float> &delta_mb,
                           std::vector<float> &delta_Sb);
