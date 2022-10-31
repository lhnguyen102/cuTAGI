///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network_base.h
// Description:  header file for tagi network base
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 08, 2022
// Updated:      October 31, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <vector>

#include "data_transfer_cpu.h"
#include "feature_availability.h"
#include "feed_forward_cpu.h"
#include "global_param_update_cpu.h"
#include "indices.h"
#include "net_init.h"
#include "net_prop.h"
#include "param_feed_backward_cpu.h"
#include "state_feed_backward_cpu.h"
#include "struct_var.h"

class TagiNetworkBase {
   public:
    std::vector<float> ma, Sa, mz, Sz, J, ma_init, Sa_init, mz_init, Sz_init,
        J_init;
    Network prop;
    IndexOut idx;
    NetState state;
    Param theta;
    DeltaState d_state;
    DeltaParam d_theta;
    Input net_input;
    Obs obs;

    int num_weights, num_biases, num_weights_sc, num_biases_sc;
    TagiNetworkBase();
    virtual ~TagiNetworkBase();
    virtual void feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                              std::vector<float> &Sx_f);
    virtual void connected_feed_forward(std::vector<float> &ma,
                                        std::vector<float> &Sa,
                                        std::vector<float> &mz,
                                        std::vector<float> &Sz,
                                        std::vector<float> &J);

    virtual void state_feed_backward(std::vector<float> &y,
                                     std::vector<float> &Sy,
                                     std::vector<int> &idx_ud);

    virtual void param_feed_backward();
    virtual void get_network_outputs();
    virtual void get_all_network_outputs();
    virtual void get_all_network_inputs();
    virtual std::tuple<std::vector<float>, std::vector<float>>
    get_inovation_mean_var(int layer);
    virtual std::tuple<std::vector<float>, std::vector<float>>
    get_state_delta_mean_var();
    virtual void set_parameters(Param &init_theta);
    virtual Param get_parameters();
};