///////////////////////////////////////////////////////////////////////////////
// File:         python_api_cpu.h
// Description:  API for Python bindings of C++/CUDA
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 19, 2022
// Updated:      October 19, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "struct_var.h"
#include "tagi_network_base.h"
#include "tagi_network_cpu.h"
#include "utility_wrapper.h"

class NetworkWrapper {
   public:
    std::unique_ptr<TagiNetworkBase> tagi_net;
    NetworkWrapper(Network &net);
    ~NetworkWrapper();
    void feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                      std::vector<float> &Sx_f);
    void state_feed_backward(std::vector<float> &y, std::vector<float> &Sy,
                             std::vector<int> &idx_ud);
    void param_feed_backward();
    std::tuple<std::vector<float>, std::vector<float>> get_network_outputs();
    void set_parameters(Param &init_theta);
    Param get_parameters();
};
