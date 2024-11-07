#pragma once
namespace pybind11 {
class module_;
}

void bind_base_hidden_states(pybind11::module_& modo);

void bind_base_delta_states(pybind11::module_& modo);

void bind_hrcsoftmax(pybind11::module_& modo);