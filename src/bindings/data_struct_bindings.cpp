///////////////////////////////////////////////////////////////////////////////
// File:         data_struct_bindings.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 31, 2023
// Updated:      December 31, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/data_struct_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/data_struct.h"

void bind_base_hidden_states(pybind11::module_ &m) {
    pybind11::class_<BaseHiddenStates>(m, "BaseHiddenStates")
        .def(pybind11::init<size_t, size_t>())
        .def(pybind11::init<>())
        .def_readwrite("mu_z", &BaseHiddenStates::mu_z)
        .def_readwrite("var_z", &BaseHiddenStates::var_z)
        .def_readwrite("mu_a", &BaseHiddenStates::mu_a)
        .def_readwrite("var_a", &BaseHiddenStates::var_a)
        .def_readwrite("jcb", &BaseHiddenStates::jcb)
        .def_readwrite("size", &BaseHiddenStates::size)
        .def_readwrite("block_size", &BaseHiddenStates::block_size)
        .def_readwrite("actual_size", &BaseHiddenStates::actual_size)
        .def("set_input_x", &BaseHiddenStates::set_input_x)
        .def("get_name", &BaseHiddenStates::get_name);
}

void bind_base_delta_states(pybind11::module_ &m) {
    pybind11::class_<BaseDeltaStates>(m, "BaseDeltaStates")
        .def(pybind11::init<size_t, size_t>())
        .def(pybind11::init<>())
        .def_readwrite("delta_mu", &BaseDeltaStates::delta_mu)
        .def_readwrite("delta_var", &BaseDeltaStates::delta_var)
        .def_readwrite("size", &BaseDeltaStates::size)
        .def_readwrite("block_size", &BaseDeltaStates::block_size)
        .def_readwrite("actual_size", &BaseDeltaStates::actual_size)
        .def("get_name", &BaseDeltaStates::get_name);
}