///////////////////////////////////////////////////////////////////////////////
// File:         network_wrapper_cpu.cpp
// Description:  Python binding of cutagi code
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 03, 2022
// Updated:      October 03, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/network_wrapper_cpu.h"

CpuNetworkWrapper::CpuNetworkWrapper(NetConfig &config) {
    this->config = config;
    map_config_to_prop(this->config, this->net);
    net_default(this->net);
    get_net_props(this->net);
    get_similar_layer(this->net);

    // Check feature availability
    check_feature_availability(net);

    // Indices
    tagi_idx(this->idx, net);
    index_default(this->idx);  // TODO: To be removed
    this->theta = initialize_param(net);
    this->state = initialize_net_states(net);

    // Update quantities
    this->d_state.set_values(this->net.n_state, this->state.msc.size(),
                             this > state.mdsc.size(), this->net.n_max_state);
    this->d_theta.set_values(this->theta.mw.size(), this->theta.mb.size(),
                             this->theta.mw_sc.size(),
                             this->theta.mb_sc.size());
}

CpuNetworkWrapper::~CpuNetworkWrapper() {}

CpuNetworkWrapper::feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                                std::vector<float> &Sx_f_batch) {
    this->ip.set_values(x_batch, Sx_batch, Sx_f_batch);
    this->op.set_values(y_batch, V_batch, idx_ud_batch);

    // Initialize input
    initialize_states_cpu(this->ip.x_batch, this->ip.Sx_batch,
                          this->> ip.Sx_f_batch, this->net.n_x,
                          this->net.batch_size, this->state);

    // Feed forward
    feed_forward_cpu(this->net, this->theta, this->idx, this->state);
}