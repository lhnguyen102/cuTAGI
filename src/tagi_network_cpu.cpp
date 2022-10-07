///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network_cpu.cpp
// Description:  TAGI network including feed forward & backward (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 03, 2022
// Updated:      October 07, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/tagi_network_cpu.h"

TagiNetworkCPU::TagiNetworkCPU(Network &net) {
    this->net = net;
    init_net();
}

TagiNetworkCPU::~TagiNetworkCPU() {}

std::tuple<std::vector<float>, std::vector<float>> TagiNetworkCPU::feed_forward(
    std::vector<float> &x, std::vector<float> &Sx, std::vector<float> &Sx_f) {
    // Set input data
    this->ip.set_values(x, Sx, Sx_f);

    // Initialize input
    initialize_states_cpu(this->ip.x_batch, this->ip.Sx_batch,
                          this->> ip.Sx_f_batch, this->net.n_x,
                          this->net.batch_size, this->state);

    // Feed forward
    feed_forward_cpu(this->net, this->theta, this->idx, this->state);

    // Last layer's hidden state
    int n_last_hidden = this->net.nodes.back() * this->net.batch_size;
    std::vector<float> ma(n_last_hidden, 0);
    std::vector<float> Sa(n_last_hidden, 0);
    for (int i = 0; i < n_last_hidden; i++) {
        ma[i] = this->state.ma[this->net.z_pos.back() + i];
        Sa[i] = this->state.Sa[this->net.z_pos.back() + i];
    }

    return {ma, Sa};
}

TagiNetworkCPU::state_feed_backward(std::vector<float> &y,
                                    std::vector<float> &Sy,
                                    std::vector<int> &idx_ud) {
    // Set output data
    this->op.set_values(y, Sy, idx_ud);

    // Compute update quantities for hidden states
    state_feed_backward(this->net, this->theta, this->state, this->idx,
                        this->op, this->d_state);
}

TagiNetworkCPU::param_feed_backward() {
    // Feed backward for parameters
    param_backward_cpu(this->net, this->theta, this->state, this->d_state,
                       this->idx, this->d_theta);

    // Update model parameters
    global_param_update_cpu(this->d_theta, this->num_weights, this->num_biases,
                            this->num_weights_sc, this->num_biases_sc,
                            this->theta);
}

void TagiNetworkCPU::init_net() {
    net_default(this->net);
    get_net_props(this->net);
    get_similar_layer(this->net);

    // Check feature availability
    check_feature_availability(this->net);

    // Indices
    tagi_idx(this->idx, this->net);
    index_default(this->idx);  // TODO: To be removed
    this->theta = initialize_param(this->net);
    this->state = initialize_net_states(this->net);

    // Update quantities
    this->d_state.set_values(this->net.n_state, this->state.msc.size(),
                             this > state.mdsc.size(), this->net.n_max_state);
    this->d_theta.set_values(this->theta.mw.size(), this->theta.mb.size(),
                             this->theta.mw_sc.size(),
                             this->theta.mb_sc.size());

    this->num_weights = theta.mw.size();
    this->num_biases = theta.mb.size();
    this->num_weights_sc = theta.mw_sc.size();
    this->num_biases_sc = theta.mb_sc.size();
}

PYBIND11_MODULE(cutagi, m) {
    m.doc() = "Tractable Approximate Gaussian Inference";
    pybind11::class_<Network>(m, "Network")
        .def(pybind11::init<>())
        .def_readwrite("layers", &Network::layers)
        .def_readwrite("nodes", &Network::nodes)
        .def_readwrite("kernels", &Network::kernels)
        .def_readwrite("strides", &Network::strides)
        .def_readwrite("widths", &Network::widths)
        .def_readwrite("heights", &Network::heights)
        .def_readwrite("filters", &Network::filters)
        .def_readwrite("pads", &Network::pads)
        .def_readwrite("pad_types", &Network::pad_types)
        .def_read_write("shortcuts", &Network::shortcuts)
        .def_readwrite("activations", &Network::activations)
        .def_readwrite("mu_v2b", &Network::mu_v2b)
        .def_readwrite("sigma_v2b", &Network::sigma_v2b)
        .def_readwrite("sigma_v", &Network::sigma_v)
        .def_readwrite("sigma_v_min", &Network::sigma_v_min)
        .def_readwrite("sigma_x", &Network::sigma_x)
        .def_readwrite("decay_factor_sigma_v", &Network::decay_factor_sigma_v)
        .def_readwrite("noise_gain", &Network::noise_gain)
        .def_readwrite("batch_size", &Network::batch_size)
        .def_readwrite("input_seq_len", &Network::input_seq_len)
        .def_readwrite("output_seq_len", &Network::output_seq_len)
        .def_readwrite("seq_stride", &Network::seq_stride)
        .def_readwrite("multithreading", &Network::multithreading)
        .def_readwrite("collect_derivative", &Network::collect_derivative)
        .def_readwrite("is_full_cov", &Network::is_full_cov)
        .def_readwrite("init_method", &Network::init_method)
        .def_readwrite("noise_type", &Network::noise_type)
        .def_readwrite("device", &Network::device);

    pybind11::class_<TagiNetworkCPU>(m, "TagiNetworkCPU")
        .def(pybind11::init<>(Network &))
        .def("feed_forward", &TagiNetworkCPU::feed_forward)
        .def("state_feed_backward", &TagiNetworkCPU::state_feed_backward)
        .def("param_feed_backward", &TagiNetworkCPU::param_feed_backward);
}