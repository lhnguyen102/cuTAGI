///////////////////////////////////////////////////////////////////////////////
// File:         conv2d_layer_bindings.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 20, 2024
// Updated:      January 20, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/conv2d_layer_bindings.h"

#include "../include/conv2d_layer.h"

void bind_conv2d_layer(pybind11::module_& modo)
/**/
{
    pybind11::class_<Conv2d, std::shared_ptr<Conv2d>, BaseLayer>(modo, "Conv2d")
        .def(pybind11::init<size_t, size_t, size_t, int, int, int, size_t,
                            size_t, float, float, std::string, bool>(),
             pybind11::arg("in_channels"), pybind11::arg("out_channels"),
             pybind11::arg("kernel_size"), pybind11::arg("stride") = 1,
             pybind11::arg("padding") = 0, pybind11::arg("padding_type") = 1,
             pybind11::arg("in_width") = 0, pybind11::arg("in_height") = 0,
             pybind11::arg("gain_w") = 1.0f, pybind11::arg("gain_b") = 1.0f,
             pybind11::arg("init_method") = "He", pybind11::arg("bias") = true)
        .def("get_layer_info", &Conv2d::get_layer_info)
        .def("get_layer_name", &Conv2d::get_layer_name)
        .def_readwrite("gain_w", &Conv2d::gain_w)
        .def_readwrite("gain_b", &Conv2d::gain_b)
        .def_readwrite("init_method", &Conv2d::init_method)
        .def("init_weight_bias", &Conv2d::init_weight_bias)
        .def("forward", &Conv2d::forward)
        .def("state_backward", &Conv2d::state_backward)
        .def("param_backward", &Conv2d::param_backward);
}