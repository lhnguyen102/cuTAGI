///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_forward.cuh
// Description:  Header file for Long-Short Term Memory (LSTM) forward pass
//               in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 05, 2022
// Updated:      September 11, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "activation_fun.cuh"
#include "data_transfer.cuh"
#include "feed_forward.cuh"
#include "net_prop.h"
#include "struct_var.h"

__global__ void cat_activations_and_prev_states(float const *a, float const *b,
                                                int n, int m, int seq_len,
                                                int B, int z_pos_a, int z_pos_b,
                                                float *c);

void save_prev_states(Network &net, StateGPU &state);

void lstm_state_forward(Network &net, StateGPU &state, ParamGPU &theta, int l);