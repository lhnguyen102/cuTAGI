///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.h
// Description:  Header file for CPU forward pass
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 17, 2022
// Updated:      August 17, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <iostream>
#include <thread>

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

void tanh_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz, int zpos,
                       int n, std::vector<float> &ma, std::vector<float> &J,
                       std::vector<float> &Sa);

void sigmoid_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                          int zpos, int n, std::vector<float> &ma,
                          std::vector<float> &J, std::vector<float> &Sa);

void initialize_states_cpu(std::vector<float> &x, std::vector<float> &Sx,
                           std::vector<float> &Sx_f, int ni, int B,
                           NetState &state);
void initialize_states_multithreading(std::vector<float> &x,
                                      std::vector<float> &Sx, int niB,
                                      unsigned int NUM_THREADS,
                                      NetState &state);

void feed_forward_cpu(Network &net, Param &theta, IndexOut &idx,
                      NetState &state);
