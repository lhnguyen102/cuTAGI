////////////////////////////////////////////////////////////////////////////////
// File:         base_layer_binding.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 30, 2023
// Updated:      December 30, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/base_layer_bindings.h"

void bind_base_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<BaseLayer, std::shared_ptr<BaseLayer>>(modo, "BaseLayer")
        .def(pybind11::init<>())
        .def_readwrite("input_size", &BaseLayer::input_size)
        .def_readwrite("output_size", &BaseLayer::output_size)
        .def_readwrite("num_weights", &BaseLayer::num_weights)
        .def_readwrite("num_biases", &BaseLayer::num_biases)
        .def_readwrite("mu_w", &BaseLayer::mu_w)
        .def_readwrite("var_w", &BaseLayer::var_w)
        .def_readwrite("mu_b", &BaseLayer::mu_b)
        .def_readwrite("var_b", &BaseLayer::var_b)
        .def_readwrite("delta_mu_w", &BaseLayer::delta_mu_w)
        .def_readwrite("delta_var_w", &BaseLayer::delta_var_w)
        .def_readwrite("delta_mu_b", &BaseLayer::delta_mu_b)
        .def_readwrite("delta_var_b", &BaseLayer::delta_var_b)
        .def_readwrite("num_threads", &BaseLayer::num_threads)
        .def_readwrite("training", &BaseLayer::training)
        .def_readwrite("device", &BaseLayer::device)
        .def("to_cuda", &BaseLayer::to_cuda);
    // .def("get_layer_info", &BaseLayer::get_layer_info)
    // .def("get_layer_name", &BaseLayer::get_layer_name)
    // .def("update_weights", &BaseLayer::update_weights)
    // .def("update_biases", &BaseLayer::update_biases);
}