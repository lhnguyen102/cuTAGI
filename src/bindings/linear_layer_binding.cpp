///////////////////////////////////////////////////////////////////////////////
// File:         linear_layer_binding.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 29, 2023
// Updated:      December 29, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/linear_layer_binding.h"

void bind_linear_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Linear>(modo, "Linear")
        .def(pybind11::init<>())
        .def_readwrite("mu_w", &Linear::mu_w)
        .def_readwrite("var_w", &Linear::var_w)
        .def_readwrite("mu_b", &Linear::mu_b)
        .def_readwrite("var_b", &Linear::var_b)
        .def_readwrite("delta_mu_w", &Linear::delta_mu_w)
        .def_readwrite("delta_var_w", &Linear::delta_var_w)
        .def_readwrite("delta_mu_b", &Linear::delta_mu_b)
        .def_readwrite("delta_var_b", &Linear::delta_var_b)
        .def_readwrite("num_threads", &Linear::num_threads)
        .def_readwrite("training", &Linear::training)
        .def("forward", &Linear::forward)
        .def("state_backward", &Linear::state_backward)
        .def("param_backward", &Linear::param_backward);
}

// PYBIND11_MODULE(neural_net_module, m) {
//     py::class_<Linear>(m, "Linear")
//         .def(py::init<int, int>())
//         // Binding the overloaded forward methods
//         .def("forward", (void (Linear::*)(BaseHiddenStates&,
//         BaseHiddenStates&, BaseTempStates&)) &Linear::forward)
//         .def("forward", (std::vector<float> (Linear::*)(const
//         std::vector<float>&)) &Linear::forward);
// }