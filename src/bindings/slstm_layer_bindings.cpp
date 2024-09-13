
///////////////////////////////////////////////////////////////////////////////
// File:         slstm_layer_binding.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 21, 2024
// Updated:      August 21, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/slstm_layer_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/lstm_layer.h"

void bind_slstm_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<SLSTM, std::shared_ptr<SLSTM>, BaseLayer>(modo, "SLSTM")
        .def(pybind11::init<size_t, size_t, int, bool, float, float,
                            std::string>(),
             pybind11::arg("input_size"), pybind11::arg("output_size"),
             pybind11::arg("seq_len"), pybind11::arg("bias"),
             pybind11::arg("gain_weight") = 1.0f,
             pybind11::arg("gain_bias") = 1.0f, pybind11::arg("method") = "He")
        .def("get_layer_info", &SLSTM::get_layer_info)
        .def("get_layer_name", &SLSTM::get_layer_name)
        .def_readwrite("gain_w", &SLSTM::gain_w)
        .def_readwrite("gain_b", &SLSTM::gain_b)
        .def_readwrite("init_method", &SLSTM::init_method)
        .def("init_weight_bias", &SLSTM::init_weight_bias)
        .def("forward", &SLSTM::forward)
        .def("backward", &SLSTM::backward);
}