///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_forward_cpu.h
// Description:  Hearder file for Long-Short Term Memory (LSTM) forward pass
//               (cpu version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 03, 2022
// Updated:      September 01, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
// #include <algorithm>
#include <vector>

#include "common.h"
#include "feed_forward_cpu.h"
#include "net_prop.h"
#include "struct_var.h"

void lstm_state_forward_cpu(Network &net, NetState &state, Param &theta, int l);

void to_prev_states(std::vector<float> &curr, std::vector<float> &prev);

void cat_activations_and_prev_states(std::vector<float> &a,
                                     std::vector<float> &b, int n, int m,
                                     int seq_len, int z_pos_a, int z_pos_b,
                                     std::vector<float> &c);