///////////////////////////////////////////////////////////////////////////////
// File:         sequential_bindings.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 30, 2023
// Updated:      December 31, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/sequential_bindings.h"

void bind_sequential(pybind11::module_& m) {
    pybind11::class_<Sequential, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(pybind11::init<>())
        .def(pybind11::init(
            [](const std::vector<std::shared_ptr<BaseLayer>>& layers) {
                auto seq = std::make_shared<Sequential>();
                for (const auto& layer : layers) {
                    seq->add_layer(layer);
                }
                return seq;
            }))

        .def_readwrite("z_buffer_size", &Sequential::z_buffer_size)
        .def_readwrite("z_buffer_block_size", &Sequential::z_buffer_block_size)
        .def_readwrite("input_size", &Sequential::input_size)
        .def_readwrite("training", &Sequential::training)
        .def_readwrite("param_update", &Sequential::param_update)
        .def_readwrite("input_hidden_state_update",
                       &Sequential::input_hidden_state_update)
        .def_readwrite("num_threads", &Sequential::num_threads)
        .def_readwrite("device", &Sequential::device)
        .def("to_device", &Sequential::to_device)
        .def("set_threads", &Sequential::set_threads)
        .def("forward", &Sequential::forward_py)
        .def("backward", &Sequential::backward)
        .def("step", &Sequential::step)
        .def("output_to_host", &Sequential::output_to_host)
        .def("delta_z_to_host", &Sequential::delta_z_to_host)
        .def("get_layer_stack_info", &Sequential::get_layer_stack_info)
        .def("save", &Sequential::save)
        .def("load", &Sequential::load)
        .def("save_csv", &Sequential::save_csv)
        .def("load_csv", &Sequential::load_csv)
        .def("params_from", &Sequential::params_from);
}