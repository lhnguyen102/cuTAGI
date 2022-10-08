///////////////////////////////////////////////////////////////////////////////
// File:         network_wrapper.cu
// Description:  Python wrapper for C++/CUDA code
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 07, 2022
// Updated:      October 08, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/network_wrapper.cuh"

NetworkWrapper::NetworkWrapper(Network &net) {
    if (net.device.compare("cuda") == 0) {
        this->tagi_net = std::make_unique<TagiNetwork>(net);
    } else if (net.device.compare("cpu") == 0) {
        this->tagi_net = std::make_unique<TagiNetworkCPU>(net);
    } else {
        throw std::invalid_argument(
            "Device is invalid. Device is either cpu or cuda");
    }
}
NetworkWrapper::~NetworkWrapper();

std::tuple<std::vector<float>, std::vector<float>> NetworkWrapper::feed_forward(
    std::vector<float> &x, std::vector<float> &Sx, std::vector<float> &Sx_f) {
    this->tagi_net.feed_forward(x, Sx, Sx_f);
}

void NetworkWrapper::state_feed_backward(std::vector<float> &y,
                                         std::vector<float> &Sy,
                                         std::vector<int> &idx_ud) {
    this->tagi_net.state_feed_backward(y, Sy, idx_ud);
}

void NetworkWrapper::param_feed_backward() {
    this->tagi_net.param_feed_backward();
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

    pybind11::class_<NetworkWrapper>(m, "NetworkWrapper")
        .def(pybind11::init<>(Network &))
        .def("feed_forward", &NetworkWrapper::feed_forward)
        .def("state_feed_backward", &NetworkWrapper::state_feed_backward)
        .def("param_feed_backward", &NetworkWrapper::param_feed_backward)
        .def("get_output_outputs", &NetworkWrapper::get_output_outputs);
}