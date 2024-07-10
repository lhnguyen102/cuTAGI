///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer_bindings.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 08, 2024
// Updated:      April 12, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/norm_layer_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/norm_layer.h"

void bind_layernorm_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<LayerNorm, std::shared_ptr<LayerNorm>, BaseLayer>(
        modo, "LayerNorm")
        .def(pybind11::init<const std::vector<int>, float, bool>(),
             pybind11::arg("normalized_shape"), pybind11::arg("eps") = 1e-4,
             pybind11::arg("bias") = true)
        .def("get_layer_info", &LayerNorm::get_layer_info)
        .def("get_layer_name", &LayerNorm::get_layer_name)
        .def("forward", &LayerNorm::forward)
        .def("backward", &LayerNorm::backward)
        .def("init_weight_bias", &LayerNorm::init_weight_bias);
    ;
}

void bind_batchnorm_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<BatchNorm2d, std::shared_ptr<BatchNorm2d>, BaseLayer>(
        modo, "BatchNorm2d")
        .def(pybind11::init<int, float, float, bool>(),
             pybind11::arg("num_features"), pybind11::arg("eps") = 1e-4,
             pybind11::arg("mometum") = 0.9, pybind11::arg("bias") = true)
        .def("get_layer_info", &BatchNorm2d::get_layer_info)
        .def("get_layer_name", &BatchNorm2d::get_layer_name)
        .def("forward", &BatchNorm2d::forward)
        .def("backward", &BatchNorm2d::backward)
        .def("init_weight_bias", &BatchNorm2d::init_weight_bias);
    ;
}