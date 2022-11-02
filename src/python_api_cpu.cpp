///////////////////////////////////////////////////////////////////////////////
// File:         python_api_cpu.cpp
// Description:  API for Python bindings of C++/CUDA
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 19, 2022
// Updated:      November 02, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/python_api_cpu.h"

NetworkWrapper::NetworkWrapper(Network &net) {
    this->tagi_net = std::make_unique<TagiNetworkCPU>(net);
}
NetworkWrapper::~NetworkWrapper(){};

void NetworkWrapper::feed_forward_wrapper(std::vector<float> &x,
                                          std::vector<float> &Sx,
                                          std::vector<float> &Sx_f) {
    this->tagi_net->feed_forward(x, Sx, Sx_f);
}
void NetworkWrapper::connected_feed_forward_wrapper(std::vector<float> &ma,
                                                    std::vector<float> &Sa,
                                                    std::vector<float> &mz,
                                                    std::vector<float> &Sz,
                                                    std::vector<float> &J) {
    this->tagi_net->connected_feed_forward(ma, Sa, mz, Sz, J);
}
void NetworkWrapper::state_feed_backward_wrapper(std::vector<float> &y,
                                                 std::vector<float> &Sy,
                                                 std::vector<int> &idx_ud) {
    this->tagi_net->state_feed_backward(y, Sy, idx_ud);
}
void NetworkWrapper::param_feed_backward_wrapper() {
    this->tagi_net->param_feed_backward();
}

std::tuple<std::vector<float>, std::vector<float>>
NetworkWrapper::get_network_outputs_wrapper() {
    this->tagi_net->get_network_outputs();

    return {this->tagi_net->ma, this->tagi_net->Sa};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>, std::vector<float>>
NetworkWrapper::get_all_network_outputs_wrapper() {
    this->tagi_net->get_all_network_outputs();

    return {this->tagi_net->ma, this->tagi_net->Sa, this->tagi_net->mz,
            this->tagi_net->Sz, this->tagi_net->J};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>, std::vector<float>>
NetworkWrapper::get_all_network_inputs_wrapper() {
    this->tagi_net->get_all_network_inputs();

    return {this->tagi_net->ma_init, this->tagi_net->Sa_init,
            this->tagi_net->mz_init, this->tagi_net->Sz_init,
            this->tagi_net->J_init};
}

std::tuple<std::vector<float>, std::vector<float>>
NetworkWrapper::get_inovation_mean_var_wrapper(int layer) {
    std::vector<float> delta_m, delta_S;
    std::tie(delta_m, delta_S) = this->tagi_net->get_inovation_mean_var(layer);
    return {delta_m, delta_S};
}

std::tuple<std::vector<float>, std::vector<float>>
NetworkWrapper::get_state_delta_mean_var_wrapper() {
    std::vector<float> delta_mz, delta_Sz;
    std::tie(delta_mz, delta_Sz) = this->tagi_net->get_state_delta_mean_var();
    return {delta_mz, delta_Sz};
}

void NetworkWrapper::set_parameters_wrapper(Param &init_theta) {
    this->tagi_net->set_parameters(init_theta);
}

Param NetworkWrapper::get_parameters_wrapper() { return this->tagi_net->theta; }

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

    pybind11::class_<HrSoftmax>(m, "HrSoftmax")
        .def(pybind11::init<>())
        .def_readwrite("obs", &HrSoftmax::obs)
        .def_readwrite("idx", &HrSoftmax::idx)
        .def_readwrite("num_obs", &HrSoftmax::n_obs)
        .def_readwrite("length", &HrSoftmax::len);

    pybind11::class_<UtilityWrapper>(m, "UtilityWrapper")
        .def(pybind11::init<>())
        .def("hierarchical_softmax_wrapper",
             &UtilityWrapper::hierarchical_softmax_wrapper)
        .def("load_mnist_dataset_wrapper",
             &UtilityWrapper::load_mnist_dataset_wrapper)
        .def("load_cifar_dataset_wrapper",
             &UtilityWrapper::load_cifar_dataset_wrapper)
        .def("get_labels_wrapper", &UtilityWrapper::get_labels_wrapper)
        .def("label_to_obs_wrapper", &UtilityWrapper::label_to_obs_wrapper)
        .def("obs_to_label_prob_wrapper",
             &UtilityWrapper::obs_to_label_prob_wrapper)
        .def("get_error_wrapper", &UtilityWrapper::get_error_wrapper);

    pybind11::class_<NetworkWrapper>(m, "NetworkWrapper")
        .def(pybind11::init<Network &>())
        .def("feed_forward_wrapper", &NetworkWrapper::feed_forward_wrapper)
        .def("connected_feed_forward_wrapper",
             &NetworkWrapper::connected_feed_forward_wrapper)
        .def("state_feed_backward_wrapper",
             &NetworkWrapper::state_feed_backward_wrapper)
        .def("param_feed_backward_wrapper",
             &NetworkWrapper::param_feed_backward_wrapper)
        .def("get_network_outputs_wrapper",
             &NetworkWrapper::get_network_outputs_wrapper)
        .def("get_all_network_outputs_wrapper",
             &NetworkWrapper::get_all_network_outputs_wrapper)
        .def("get_all_network_inputs_wrapper",
             &NetworkWrapper::get_all_network_inputs_wrapper)
        .def("get_inovation_mean_var_wrapper",
             &NetworkWrapper::get_inovation_mean_var_wrapper)
        .def("get_state_delta_mean_var_wrapper",
             &NetworkWrapper::get_state_delta_mean_var_wrapper)
        .def("set_parameters_wrapper", &NetworkWrapper::set_parameters_wrapper)
        .def("get_parameters_wrapper", &NetworkWrapper::get_parameters_wrapper);
}