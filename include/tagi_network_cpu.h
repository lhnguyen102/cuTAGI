///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network_cpu.h
// Description:  Header file for tagi network including feed forward & backward
//               (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 03, 2022
// Updated:      October 08, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "tagi_network_base.h"

class TagiNetworkCPU : public TagiNetworkBase {
   public:
    TagiNetworkCPU(Network &net);
    ~TagiNetworkCPU();
    void feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                      std::vector<float> &Sx_f);
    void state_feed_backward(std::vector<float> &y, std::vector<float> &Sy,
                             std::vector<int> &idx_ud);
    void param_feed_backward();
    void get_network_outputs();

   private:
    void init_net();
};