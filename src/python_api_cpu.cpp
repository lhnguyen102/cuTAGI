///////////////////////////////////////////////////////////////////////////////
// File:         python_api_cpu.cpp
// Description:  API for Python bindings of C++/CUDA
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 19, 2022
// Updated:      October 19, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/python_api_cpu.h"

NetworkWrapper::NetworkWrapper(Network &net) {
    this->tagi_net = std::make_unique<TagiNetworkCPU>(net);
}
NetworkWrapper::~NetworkWrapper(){};

void NetworkWrapper::feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                                  std::vector<float> &Sx_f) {
    this->tagi_net->feed_forward(x, Sx, Sx_f);
}
void NetworkWrapper::state_feed_backward(std::vector<float> &y,
                                         std::vector<float> &Sy,
                                         std::vector<int> &idx_ud) {
    this->tagi_net->state_feed_backward(y, Sy, idx_ud);
}
void NetworkWrapper::param_feed_backward() {
    this->tagi_net->param_feed_backward();
}

std::tuple<std::vector<float>, std::vector<float>>
NetworkWrapper::get_network_outputs() {
    this->tagi_net->get_network_outputs();

    return {this->tagi_net->ma, this->tagi_net->Sa};
}

void NetworkWrapper::set_parameters(Param &init_theta) {
    this->tagi_net->set_parameters(init_theta);
}

Param NetworkWrapper::get_parameters() { return this->tagi_net->theta; }

PYBIND11_MODULE(pytagi, m) {
    m.doc() = "Tractable Approximate Gaussian Inference";

    pybind11::class_<Param>(m, "Param")
        .def(pybind11::init<>())
        .def_readwrite("mw", &Param::mw)
        .def_readwrite("Sw", &Param::Sw)
        .def_readwrite("mb", &Param::mb)
        .def_readwrite("Sb", &Param::Sb)
        .def_readwrite("mw_sc", &Param::mw_sc)
        .def_readwrite("Sw_sc", &Param::Sw_sc)
        .def_readwrite("mb_sc", &Param::mb_sc)
        .def_readwrite("Sb_sc", &Param::Sb_sc);

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
        .def_readwrite("shortcuts", &Network::shortcuts)
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

    pybind11::class_<UtilityWrapper>(m, "UtilityWrapper")
        .def(pybind11::init<>())
        .def("hierarchical_softmax", &UtilityWrapper::hierarchical_softmax)
        .def("load_mnist_dataset", &UtilityWrapper::load_mnist_dataset)
        .def("load_cifar_dataset", &UtilityWrapper::load_cifar_dataset);

    pybind11::class_<NetworkWrapper>(m, "NetworkWrapper")
        .def(pybind11::init<Network &>())
        .def("feed_forward", &NetworkWrapper::feed_forward)
        .def("state_feed_backward", &NetworkWrapper::state_feed_backward)
        .def("param_feed_backward", &NetworkWrapper::param_feed_backward)
        .def("get_network_outputs", &NetworkWrapper::get_network_outputs)
        .def("set_parameters", &NetworkWrapper::set_parameters)
        .def("get_parameters", &NetworkWrapper::get_parameters);
}