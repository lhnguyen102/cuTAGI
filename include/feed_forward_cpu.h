///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.h
// Description:  Header file for CPU forward pass
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 17, 2022
// Updated:      September 11, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <iostream>
#include <thread>

#include "activation_fun_cpu.h"
#include "common.h"
#include "derivative_calcul_cpu.h"
#include "lstm_feed_forward_cpu.h"
#include "net_prop.h"
#include "struct_var.h"

void fc_mean_cpu(std::vector<float> &mw, std::vector<float> &mb,
                 std::vector<float> &ma, int w_pos, int b_pos, int z_pos_in,
                 int z_pos_out, int m, int n, int k, std::vector<float> &mz);

void fc_var_cpu(std::vector<float> &mw, std::vector<float> &Sw,
                std::vector<float> &Sb, std::vector<float> &ma,
                std::vector<float> &Sa, int w_pos, int b_pos, int z_pos_in,
                int z_pos_out, int m, int n, int k, std::vector<float> &Sz);

void fc_mean_var_multithreading(std::vector<float> &mw, std::vector<float> &Sw,
                                std::vector<float> &mb, std::vector<float> &Sb,
                                std::vector<float> &ma, std::vector<float> &Sa,
                                int w_pos, int b_pos, int z_pos_in,
                                int z_pos_out, int m, int n, int k,
                                unsigned int NUM_THREADS,
                                std::vector<float> &mz, std::vector<float> &Sz);

void initialize_states_cpu(std::vector<float> &x, std::vector<float> &Sx,
                           std::vector<float> &Sx_f, int ni, int B,
                           NetState &state);
void initialize_full_states_cpu(
    std::vector<float> &mz_init, std::vector<float> &Sz_init,
    std::vector<float> &ma_init, std::vector<float> &Sa_init,
    std::vector<float> &J_init, std::vector<float> &mz, std::vector<float> &Sz,
    std::vector<float> &ma, std::vector<float> &Sa, std::vector<float> &J);

void initialize_states_multithreading(std::vector<float> &x,
                                      std::vector<float> &Sx, int niB,
                                      unsigned int NUM_THREADS,
                                      NetState &state);

void feed_forward_cpu(Network &net, Param &theta, IndexOut &idx,
                      NetState &state);
