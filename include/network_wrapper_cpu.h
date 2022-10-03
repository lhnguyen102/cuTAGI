///////////////////////////////////////////////////////////////////////////////
// File:         network_wrapper_cpu.h
// Description:  Header file for Python binding of cutagi code
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 03, 2022
// Updated:      October 03, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <third_party/pybind11/pybind11.h>
#include <third_party/pybind11/stl.h>

#include <iostream>
#include <vector>

#include "data_transfer_cpu.h"
#include "feature_availability.h"
#include "feed_forward_cpu.h"
#include "indices.h"
#include "net_init.h"
#include "net_prop.h"
#include "struct_var.h"

class CpuNetworkWrapper {
   public:
    NetConfig config;
    Network net;
    IndexOut idx;
    NetState state;
    Param theta;
    DeltaState d_state;
    DeltaParam d_theta;
    Input net_input;
    Obs obs;
    CpuNetworkWrapper(NetConfig &config);
    ~CpuNetworkWrapper();
    void feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                      std::vector<float> &Sx_f_batch);
    void state_feed_backward(std::vector<float> &y, std::vector<float> &Sy,
                             std::vector<int> &idx_ud_batch);
    void param_feed_backward();
}