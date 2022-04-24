///////////////////////////////////////////////////////////////////////////////
// File:         net_prop.h
// Description:  Header file for net_prop.cpp
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 15, 2022
// Updated:      April 20, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>
//#include <filesystem>
#include <map>
#include <random>
#include <sstream>
#include <string>

#include "struct_var.h"

std::tuple<int, int> compute_downsample_img_size(int kernel, int stride, int wi,
                                                 int hi, int pad, int pad_type);

std::tuple<int, int> compute_upsample_img_size(int kernel, int stride, int wi,
                                               int hi, int pad, int pad_type);

std::tuple<int, int> get_number_param_fc(int ni, int no, bool use_bias);

std::tuple<int, int> get_number_param_conv(int kernel, int fi, int fo,
                                           bool use_bias);

std::tuple<int, int> get_number_param_norm(int n);

void get_similar_layer(Network &net);

void set_idx_to_similar_layer(std::vector<int> &similar_layers,
                              std::vector<int> &idx);

float he_init(float fan_in);

float xavier_init(float fan_in, float fan_out);

std::tuple<std::vector<float>, std::vector<float>> gaussian_param_init(
    float scale, float gain, int N);

void get_net_props(Network &net);
void net_default(Network &net);
NetState initialize_net_states(Network &net);

Param initialize_param(Network &net);

Network load_cfg(std::string net_file);
