///////////////////////////////////////////////////////////////////////////////
// File:         slinear_layer_binding.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 21, 2024
// Updated:      August 21, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/slinear_layer_bindings.h"

void bind_slinear_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<SLinear, std::shared_ptr<SLinear>, BaseLayer>(modo,
                                                                   "SLinear")
        .def(pybind11::init<size_t, size_t, bool, float, float, std::string>(),
             pybind11::arg("ip_size"), pybind11::arg("op_size"),
             pybind11::arg("bias"), pybind11::arg("gain_weight") = 1.0f,
             pybind11::arg("gain_bias") = 1.0f, pybind11::arg("method") = "He")
        .def("get_layer_info", &SLinear::get_layer_info)
        .def("get_layer_name", &SLinear::get_layer_name)
        .def_readwrite("gain_w", &SLinear::gain_w)
        .def_readwrite("gain_b", &SLinear::gain_b)
        .def_readwrite("init_method", &SLinear::init_method)
        .def("init_weight_bias", &SLinear::init_weight_bias)
        .def("forward", &SLinear::forward)
        .def("state_backward", &SLinear::backward);
}