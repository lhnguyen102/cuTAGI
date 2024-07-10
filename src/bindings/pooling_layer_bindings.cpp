///////////////////////////////////////////////////////////////////////////////
// File:         pooling_layer_bindings.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 20, 2024
// Updated:      January 20, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/pooling_layer_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/pooling_layer.h"

void bind_avgpool2d_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<AvgPool2d, std::shared_ptr<AvgPool2d>, BaseLayer>(
        modo, "AvgPool2d")
        .def(pybind11::init<size_t, int, int, int>(),
             pybind11::arg("kernel_size"), pybind11::arg("stride") = -1,
             pybind11::arg("padding") = 0, pybind11::arg("padding_type") = 0)
        .def("get_layer_info", &AvgPool2d::get_layer_info)
        .def("get_layer_name", &AvgPool2d::get_layer_name)
        .def("forward", &AvgPool2d::forward)
        .def("state_backward", &AvgPool2d::backward);
    ;
}