///////////////////////////////////////////////////////////////////////////////
// File:         sequential_bindings.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 30, 2023
// Updated:      March 18, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/sequential_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <tuple>

#include "../include/base_layer.h"
#include "../include/data_struct.h"
#include "../include/sequential.h"

void bind_sequential(pybind11::module_& m) {
    pybind11::class_<Sequential, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(pybind11::init<>())
        .def(pybind11::init(
            [](const std::vector<std::shared_ptr<BaseLayer>>& layers) {
                auto seq = std::make_shared<Sequential>();
                for (const auto& layer : layers) {
                    seq->add_layer(layer);
                }

                // Perform the pre-computation of the network's parameters
                seq->add_layers();
                return seq;
            }))
        .def_readwrite("layers", &Sequential::layers)
        .def_readwrite("output_z_buffer", &Sequential::output_z_buffer)
        .def_readwrite("input_delta_z_buffer",
                       &Sequential::input_delta_z_buffer)
        .def_readwrite("output_delta_z_buffer",
                       &Sequential::output_delta_z_buffer)
        .def_readwrite("z_buffer_size", &Sequential::z_buffer_size)
        .def_readwrite("z_buffer_block_size", &Sequential::z_buffer_block_size)
        .def_readwrite("input_size", &Sequential::input_size)
        .def_readwrite("num_samples", &Sequential::num_samples)
        .def_readwrite("training", &Sequential::training)
        .def_readwrite("param_update", &Sequential::param_update)
        .def_readwrite("device", &Sequential::device)
        .def_readwrite("input_state_update", &Sequential::input_state_update)
        .def_readwrite("num_threads", &Sequential::num_threads)
        .def_readwrite("device", &Sequential::device)
        .def("to_device", &Sequential::to_device)
        .def("params_to_host", &Sequential::params_to_host)
        .def("params_to_device", &Sequential::params_to_device)
        .def("set_threads", &Sequential::set_threads)
        .def("train", &Sequential::train)
        .def("eval", &Sequential::eval)
        .def("forward", &Sequential::forward_py)
        .def("forward",
             [](Sequential& self, pybind11::object arg1,
                pybind11::object arg2 = pybind11::none()) {
                 if (pybind11::isinstance<pybind11::array_t<float>>(arg1)) {
                     pybind11::array_t<float> mu_a_np =
                         arg1.cast<pybind11::array_t<float>>();
                     pybind11::array_t<float> var_a_np =
                         arg2.is_none() ? pybind11::array_t<float>()
                                        : arg2.cast<pybind11::array_t<float>>();
                     self.forward_py(mu_a_np, var_a_np);
                 } else {
                     // Handle the case for BaseHiddenStates
                     BaseHiddenStates& input_states =
                         arg1.cast<BaseHiddenStates&>();
                     self.forward(input_states);
                 }
             })
        .def("backward", &Sequential::backward)
        .def("smoother", &Sequential::smoother)
        .def("step", &Sequential::step)
        .def("output_to_host", &Sequential::output_to_host)
        .def("delta_z_to_host", &Sequential::delta_z_to_host)
        .def("get_layer_stack_info", &Sequential::get_layer_stack_info)
        .def("preinit_layer", &Sequential::preinit_layer)
        .def("save", &Sequential::save)
        .def("load", &Sequential::load)
        .def("save_csv", &Sequential::save_csv)
        .def("load_csv", &Sequential::load_csv)
        .def("params_from", &Sequential::params_from)
        .def("parameters",
             [](Sequential& self) {
                 auto params = self.parameters();
                 pybind11::list py_params;
                 for (auto& param_ref : params) {
                     auto& param = param_ref.get();
                     py_params.append(pybind11::array_t<float>(
                         {static_cast<long>(param.size())}, {sizeof(float)},
                         param.data()));
                 }
                 return py_params;
             })
        .def("get_state_dict",
             [](Sequential& self) {
                 auto cpp_state_dict = self.get_state_dict();
                 pybind11::dict py_state_dict;
                 for (const auto& pair : cpp_state_dict) {
                     pybind11::dict layer_dict;
                     layer_dict["mu_w"] = std::get<0>(pair.second);
                     layer_dict["var_w"] = std::get<1>(pair.second);
                     layer_dict["mu_b"] = std::get<2>(pair.second);
                     layer_dict["var_b"] = std::get<3>(pair.second);
                     py_state_dict[pair.first.c_str()] = layer_dict;
                 }
                 return py_state_dict;
             })

        .def("load_state_dict",
             [](Sequential& self, const pybind11::dict& py_state_dict) {
                 std::map<std::string,
                          std::tuple<std::vector<float>, std::vector<float>,
                                     std::vector<float>, std::vector<float>>>
                     cpp_state_dict;
                 for (const auto& item : py_state_dict) {
                     std::string key = pybind11::cast<std::string>(item.first);
                     pybind11::dict layer_dict =
                         item.second.cast<pybind11::dict>();
                     cpp_state_dict[key] = std::make_tuple(
                         layer_dict["mu_w"].cast<std::vector<float>>(),
                         layer_dict["var_w"].cast<std::vector<float>>(),
                         layer_dict["mu_b"].cast<std::vector<float>>(),
                         layer_dict["var_b"].cast<std::vector<float>>());
                 }
                 self.load_state_dict(cpp_state_dict);
             })
        .def("get_outputs", &Sequential::get_outputs)
        .def("get_outputs_smoother", &Sequential::get_outputs_smoother)
        .def("get_input_states", &Sequential::get_input_states);
}