#include "../include/bindings/data_struct_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/data_struct.h"

void bind_base_hidden_states(pybind11::module_ &m) {
    pybind11::class_<BaseHiddenStates, std::shared_ptr<BaseHiddenStates>>(
        m, "BaseHiddenStates")
        .def(pybind11::init<size_t, size_t>())
        .def(pybind11::init<>())
        .def_property(
            "mu_a",
            [](BaseHiddenStates &self) {
                self.sync_to_host();
                return self.mu_a;
            },
            [](BaseHiddenStates &self, const std::vector<float> &v) {
                self.mu_a = v;
            })
        .def_property(
            "var_a",
            [](BaseHiddenStates &self) {
                self.sync_to_host();
                return self.var_a;
            },
            [](BaseHiddenStates &self, const std::vector<float> &v) {
                self.var_a = v;
            })
        .def_property(
            "jcb",
            [](BaseHiddenStates &self) {
                self.sync_to_host();
                return self.jcb;
            },
            [](BaseHiddenStates &self, const std::vector<float> &v) {
                self.jcb = v;
            })
        .def_readwrite("size", &BaseHiddenStates::size)
        .def_readwrite("block_size", &BaseHiddenStates::block_size)
        .def_readwrite("actual_size", &BaseHiddenStates::actual_size)
        .def_readwrite("width", &BaseHiddenStates::width)
        .def_readwrite("height", &BaseHiddenStates::height)
        .def_readwrite("depth", &BaseHiddenStates::depth)
        .def("set_input_x", &BaseHiddenStates::set_input_x)
        .def("get_name", &BaseHiddenStates::get_name)
        .def("set_size", &BaseHiddenStates::set_size)
        .def("swap", &BaseHiddenStates::swap)
        .def("copy_from", &BaseHiddenStates::copy_from);
}

void bind_base_delta_states(pybind11::module_ &m) {
    pybind11::class_<BaseDeltaStates, std::shared_ptr<BaseDeltaStates>>(
        m, "BaseDeltaStates")
        .def(pybind11::init<size_t, size_t>())
        .def(pybind11::init<>())
        .def_readwrite("delta_mu", &BaseDeltaStates::delta_mu)
        .def_readwrite("delta_var", &BaseDeltaStates::delta_var)
        .def_readwrite("size", &BaseDeltaStates::size)
        .def_readwrite("block_size", &BaseDeltaStates::block_size)
        .def_readwrite("actual_size", &BaseDeltaStates::actual_size)
        .def("get_name", &BaseDeltaStates::get_name)
        .def("reset_zeros", &BaseDeltaStates::reset_zeros)
        .def("copy_from", &BaseDeltaStates::copy_from, pybind11::arg("source"),
             pybind11::arg("num_data") = -1)
        .def("set_size", &BaseDeltaStates::set_size)
        .def("swap", &BaseDeltaStates::swap);
    ;
}

void bind_hrcsoftmax(pybind11::module_ &modo)
/*
 */
{
    pybind11::class_<HRCSoftmax>(modo, "HRCSoftmax")
        .def(pybind11::init<>())
        .def_readwrite("obs", &HRCSoftmax::obs)
        .def_readwrite("idx", &HRCSoftmax::idx)
        .def_readwrite("num_obs", &HRCSoftmax::n_obs)
        .def_readwrite("len", &HRCSoftmax::len);
}
