///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network_cpu.cpp
// Description:  TAGI network including feed forward & backward (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 03, 2022
// Updated:      October 09, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/tagi_network_cpu.h"

TagiNetworkCPU::TagiNetworkCPU(Network &net_prop) {
    this->prop = net_prop;
    init_net();
}

TagiNetworkCPU::~TagiNetworkCPU() {}

void TagiNetworkCPU::feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                                  std::vector<float> &Sx_f) {
    // Set input data
    this->net_input.set_values(x, Sx, Sx_f);

    // Initialize input
    initialize_states_cpu(this->net_input.x_batch, this->net_input.Sx_batch,
                          this->net_input.Sx_f_batch, this->prop.n_x,
                          this->prop.batch_size, this->state);

    // Feed forward
    feed_forward_cpu(this->prop, this->theta, this->idx, this->state);
}

void TagiNetworkCPU::state_feed_backward(std::vector<float> &y,
                                         std::vector<float> &Sy,
                                         std::vector<int> &idx_ud) {
    // Set output data
    this->obs.set_values(y, Sy, idx_ud);

    // Compute update quantities for hidden states
    state_backward_cpu(this->prop, this->theta, this->state, this->idx,
                       this->obs, this->d_state);
}

void TagiNetworkCPU::param_feed_backward() {
    // Feed backward for parameters
    param_backward_cpu(this->prop, this->theta, this->state, this->d_state,
                       this->idx, this->d_theta);

    // Update model parameters
    global_param_update_cpu(this->d_theta, this->num_weights, this->num_biases,
                            this->num_weights_sc, this->num_biases_sc,
                            this->theta);
}

void TagiNetworkCPU::init_net() {
    net_default(this->prop);
    get_net_props(this->prop);
    get_similar_layer(this->prop);

    // Check feature availability
    check_feature_availability(this->prop);

    // Indices
    tagi_idx(this->idx, this->prop);
    index_default(this->idx);  // TODO: To be removed
    this->theta = initialize_param(this->prop);
    this->state = initialize_net_states(this->prop);

    // Update quantities
    this->d_state.set_values(this->prop.n_state, this->state.msc.size(),
                             this->state.mdsc.size(), this->prop.n_max_state);
    this->d_theta.set_values(this->theta.mw.size(), this->theta.mb.size(),
                             this->theta.mw_sc.size(),
                             this->theta.mb_sc.size());

    this->num_weights = theta.mw.size();
    this->num_biases = theta.mb.size();
    this->num_weights_sc = theta.mw_sc.size();
    this->num_biases_sc = theta.mb_sc.size();
    this->ma.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->Sa.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
}

void TagiNetworkCPU::get_network_outputs() {
    // Last layer's hidden state
    int num_outputs = this->prop.nodes.back() * this->prop.batch_size;
    for (int i = 0; i < num_outputs; i++) {
        this->ma[i] = this->state.ma[this->prop.z_pos.back() + i];
        this->Sa[i] = this->state.Sa[this->prop.z_pos.back() + i];
    }
}