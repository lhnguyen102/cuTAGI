///////////////////////////////////////////////////////////////////////////////
// File:         network_wrapper.h
// Description:  Header file for Python binding of cutagi code
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 05, 2022
// Updated:      October 05, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <third_party/pybind11/pybind11.h>
#include <third_party/pybind11/stl.h>

#include <iostream>
#include <vector>

#include "data_transfer.cuh"
#include "data_transfer_cpu.h"
#include "feature_availability.h"
#include "feed_forward.cuh"
#include "feed_forward_cpu.h"
#include "indices.h"
#include "net_init.h"
#include "net_prop.h"
#include "param_feed_backward.cuh"
#include "param_feed_backward_cpu.h"
#include "state_feed_backward.cuh"
#include "state_feed_backward_cpu.h"
#include "struct_var.h"

class NetworkWrapper {
   public:
    IndexOut idx;
    NetState state;
    Param theta;
    DeltaState d_state;
    DeltaParam d_theta;
    Input net_input;
    Obs obs;

    IndexGPU idx_gpu;
    StateGPU state_gpu;
    ParamGPU theta_gpu;
    DeltaStateGPU d_state_gpu;
    DeltaParamGPU d_theta_gpu;

    int num_weights, num_biases, num_weights_sc, nun_biases_sc;

    NetworkWrapper(Network &net);
    ~NetworkWrapper();
    std::tuple<std::vector<float>, std::vector<float>> feed_forward(
        std::vector<float> &x, std::vector<float> &Sx,
        std::vector<float> &Sx_f_batch);
    void state_feed_backward(std::vector<float> &y, std::vector<float> &Sy,
                             std::vector<int> &idx_ud_batch);
    void param_feed_backward();

   private:
    void feed_forward_cpu(std::vector<float> &x, std::vector<float> &Sx,
                          std::vector<float> &Sx_f);
    void state_feed_backward_cpu(std::vector<float> &y, std::vector<float> &Sy,
                                 std::vector<int> &idx_ud);
    void param_feed_backward_cpu();

    void feed_forward_cuda(std::vector<float> &x, std::vector<float> &Sx,
                           std::vector<float> &Sx_f);

    void state_feed_backward_cuda(std::vector<float> &y, std::vector<float> &Sy,
                                  std::vector<int> &idx_ud);
    void param_feed_backward_cuda();

    void state_to_host();

    void param_to_host();

    std::tuple<std::vector<float>, std::vector<float>> get_weights();
}
