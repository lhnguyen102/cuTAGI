////////////////////////////////////////////////////////////////////////////////
// File:         base_layer_cuda_binding.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 30, 2023
// Updated:      December 30, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/linear_layer_cuda_bindings.h"

void base_layer_cuda_binding(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<BaseLayerCuda, BaseLayer>(modo, "BaseLayerCuda")
        .def(pybind11::init<>())
        .def("to_host", &Linear::to_host)
        .def("params_to_host", &Linear::params_to_host);
}