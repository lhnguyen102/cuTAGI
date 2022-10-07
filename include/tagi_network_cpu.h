///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network_cpu.h
// Description:  Header file for tagi network including feed forward & backward
//               (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 03, 2022
// Updated:      October 07, 2022
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
#include "param_feed_backward_cpu.h"
#include "state_feed_backward_cpu.h"
#include "struct_var.h"

class TagiNetworkCPU {
   public:
    IndexOut idx;
    NetState state;
    Param theta;
    DeltaState d_state;
    DeltaParam d_theta;
    Input net_input;
    Obs obs;
    int num_weights, num_biases, num_weights_sc, nun_biases_sc;
    TagiNetworkCPU(Network &net);
    TagiNetworkCPU();
    ~TagiNetworkCPU();
    std::tuple<std::vector<float>, std::vector<float>> feed_forward(
        std::vector<float> &x, std::vector<float> &Sx,
        std::vector<float> &Sx_f_batch);
    void state_feed_backward(std::vector<float> &y, std::vector<float> &Sy,
                             std::vector<int> &idx_ud_batch);
    void param_feed_backward();

   private:
    void init_net();
}