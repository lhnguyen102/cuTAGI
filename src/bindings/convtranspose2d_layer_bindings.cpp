///////////////////////////////////////////////////////////////////////////////
// File:         convtranspose2d_layer_bindings.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 19, 2024
// Updated:      March 19, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/convtranspose2d_layer_bindings.h"

#include "../include/base_layer.h"
#include "../include/convtranspose2d_layer.h"

void bind_convtranspose2d_layer(pybind11::module_& modo)
/**/
{
    pybind11::class_<ConvTranspose2d, std::shared_ptr<ConvTranspose2d>,
                     BaseLayer>(modo, "ConvTranspose2d")
        .def(pybind11::init<size_t, size_t, size_t, bool, int, int, int, size_t,
                            size_t, float, float, std::string>(),
             pybind11::arg("in_channels"), pybind11::arg("out_channels"),
             pybind11::arg("kernel_size"), pybind11::arg("bias") = true,
             pybind11::arg("stride") = 1, pybind11::arg("padding") = 0,
             pybind11::arg("padding_type") = 1, pybind11::arg("in_width") = 0,
             pybind11::arg("in_height") = 0, pybind11::arg("gain_w") = 1.0f,
             pybind11::arg("gain_b") = 1.0f,
             pybind11::arg("init_method") = "He")
        .def("get_layer_info", &ConvTranspose2d::get_layer_info)
        .def("get_layer_name", &ConvTranspose2d::get_layer_name)
        .def_readwrite("gain_w", &ConvTranspose2d::gain_w)
        .def_readwrite("gain_b", &ConvTranspose2d::gain_b)
        .def_readwrite("init_method", &ConvTranspose2d::init_method)
        .def("init_weight_bias", &ConvTranspose2d::init_weight_bias)
        .def("forward", &ConvTranspose2d::forward)
        .def("backward", &ConvTranspose2d::backward);
}