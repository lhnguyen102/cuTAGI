///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.h
// Description:  Header file for CPU forward pass
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 17, 2022
// Updated:      July 30, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <iostream>
#include <thread>

#include "common.h"
#include "derivative_calcul_cpu.h"
#include "net_prop.h"
#include "struct_var.h"

void initialize_states_cpu(std::vector<float> &x, std::vector<float> &Sx,
                           std::vector<float> &Sx_f, int ni, int B,
                           NetState &state);
void initialize_states_multithreading(std::vector<float> &x,
                                      std::vector<float> &Sx, int niB,
                                      unsigned int NUM_THREADS,
                                      NetState &state);

void feed_forward_cpu(Network &net, Param &theta, IndexOut &idx,
                      NetState &state);
